"""Honcho-based session management for conversation history."""

from __future__ import annotations

import copy
import queue
import logging
import platform as _platform
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING

from plugins.memory.honcho.client import get_honcho_client, spawn_context_thread
from plugins.memory.honcho.session_auth import (
    HonchoAuthError,
    SessionAuthMixin,
    _is_auth_error,
)
from plugins.memory.honcho.session_context import SessionContextMixin
from plugins.memory.honcho.session_migration import SessionMigrationMixin
from plugins.memory.honcho.session_peers import SessionPeersMixin
from plugins.memory.honcho.semantic_safety import (
    SemanticPayloadRejected,
    is_safe_semantic_payload,
    require_safe_semantic_payload,
)

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

# Sentinel to signal the async writer thread to shut down
_ASYNC_SHUTDOWN = object()


class DeliveryState(str, Enum):
    """Terminal state of one direct Honcho delivery boundary call."""

    NOOP = "noop"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass(frozen=True)
class DeliveryOutcome:
    """Content-free result of a bounded, immediate Honcho delivery attempt."""

    state: DeliveryState
    attempted_count: int = 0
    delivered_count: int = 0
    pending_count: int = 0
    error_category: str | None = None
    http_status: int | None = None

    @classmethod
    def aggregate(cls, outcomes: list[DeliveryOutcome]) -> DeliveryOutcome:
        """Combine boundary results without losing any terminal failure."""
        if not outcomes:
            return cls(state=DeliveryState.NOOP)
        failures = [
            outcome for outcome in outcomes if outcome.state is DeliveryState.FAILED
        ]
        state = (
            DeliveryState.FAILED
            if failures
            else DeliveryState.DELIVERED
            if any(outcome.state is DeliveryState.DELIVERED for outcome in outcomes)
            else DeliveryState.NOOP
        )
        latest_failure = failures[-1] if failures else None
        return cls(
            state=state,
            attempted_count=sum(outcome.attempted_count for outcome in outcomes),
            delivered_count=sum(outcome.delivered_count for outcome in outcomes),
            pending_count=sum(outcome.pending_count for outcome in outcomes),
            error_category=latest_failure.error_category if latest_failure else None,
            http_status=latest_failure.http_status if latest_failure else None,
        )


def classify_delivery_error(exc: BaseException) -> tuple[str, int | None]:
    """Classify a delivery error without retaining or rendering its text."""
    raw_status = getattr(exc, "status_code", None)
    if raw_status is None:
        raw_status = getattr(exc, "status", None)
    status = (
        raw_status
        if isinstance(raw_status, int) and not isinstance(raw_status, bool)
        else None
    )
    if status is not None and not 100 <= status <= 599:
        status = None
    if _is_auth_error(exc):
        return "authentication", status
    code = getattr(exc, "code", None)
    if code == "timeout":
        return "timeout", status
    if code == "connection_error":
        return "connection", status
    if isinstance(exc, TimeoutError):
        return "timeout", status
    if isinstance(exc, ConnectionError):
        return "connection", status
    if status is not None:
        return "http_error", status
    return "sdk_error", None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_machine_identity(context: dict[str, Any]) -> str:
    for key in ("machine_id", "runtime_machine_id"):
        value = context.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    try:
        value = _platform.node().strip()
    except Exception:
        value = ""
    return value or "unknown-machine"


_SECRET_SHAPED_RE = re.compile(
    r"(?i)(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\beyJ[A-Za-z0-9_-]{10,}\.|"
    r"\b(?:api[_-]?key|token|password|secret)\s*[:=]|\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,})"
)
_SAFE_CREDENTIAL_REFERENCE_SUFFIXES = (
    "_path",
    "_ref",
    "_reference",
    "_name",
    "_id",
    "_status",
    "_count",
    "_policy",
    "_enabled",
)


def _normalize_metadata_key(key: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_sensitive_metadata_key(key: str) -> bool:
    normalized = _normalize_metadata_key(key)
    if normalized.endswith(_SAFE_CREDENTIAL_REFERENCE_SUFFIXES):
        return False
    parts = normalized.split("_")
    compact = "".join(parts)
    if any(
        part in {"password", "passwd", "secret", "token", "authorization"}
        for part in parts
    ):
        return True
    if any(
        marker in compact
        for marker in (
            "apikey",
            "privatekey",
            "clientsecret",
            "accesstoken",
            "refreshtoken",
            "authtoken",
            "password",
            "authorization",
        )
    ):
        return True
    return any(
        parts[index : index + 2] in (["api", "key"], ["private", "key"])
        for index in range(max(0, len(parts) - 1))
    )


def _redact_metadata_value(value: Any) -> Any:
    """Force-redact metadata recursively and fail closed for suspicious values."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            raw_key = str(key)
            safe_key = str(_redact_metadata_value(raw_key))
            result[safe_key] = (
                "«redacted-metadata»"
                if _is_sensitive_metadata_key(raw_key)
                else _redact_metadata_value(item)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_metadata_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        return "«redacted-metadata»"
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(value, force=True, redact_url_credentials=True)
    except Exception:
        return "«redacted-metadata»"
    return (
        "«redacted-metadata»"
        if redacted == value and _SECRET_SHAPED_RE.search(value)
        else redacted
    )


@dataclass
class HonchoSession:
    """A conversation session backed by Honcho: a local message cache that syncs to Honcho."""

    key: str  # channel:chat_id
    user_peer_id: str  # Honcho peer ID for the user
    assistant_peer_id: str  # Honcho peer ID for the assistant
    honcho_session_id: str  # Honcho session ID
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the local cache."""
        created_at = kwargs.pop("created_at", None) or _utc_now()
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": _rfc3339(created_at),
            "created_at": created_at,
            **kwargs,
        })
        self.updated_at = _utc_now()


class HonchoSessionManager(
    SessionAuthMixin, SessionPeersMixin, SessionContextMixin, SessionMigrationMixin
):
    """Conversation sessions backed by Honcho, alongside hermes' SQLite state and file memory.
    Auth retry, peer-ID resolution, recall and memory-file migration live in the mixins."""

    def _authed_semantic_write(self, label: str, payload: Any, fn: Any) -> Any:
        """Validate exact payload before every authenticated write attempt."""
        def _validated_attempt() -> Any:
            require_safe_semantic_payload(payload)
            return fn()

        return self._authed_call(label, _validated_attempt)

    def __init__(
        self,
        honcho: Honcho | None = None,
        context_tokens: int | None = None,
        config: Any | None = None,
        runtime_user_peer_name: str | None = None,
        runtime_user_peer_name_alt: str | None = None,
        provenance_context: dict[str, Any] | None = None,
        source_session_id: str | None = None,
    ):
        """``honcho`` defaults to the per-identity cached client; ``context_tokens`` caps
        context() calls (None = Honcho default); the runtime peer names are the gateway
        user identity (and a stable alternate) for per-user memory scoping."""
        self._honcho = honcho
        self._context_tokens = context_tokens
        self._config = config
        self._runtime_user_peer_name = runtime_user_peer_name
        self._runtime_user_peer_name_alt = runtime_user_peer_name_alt
        self._provenance_context = dict(provenance_context or {})
        self._source_session_id = str(source_session_id or "").strip()
        configured_metadata = getattr(config, "message_metadata", {}) if config else {}
        self._message_metadata = (
            copy.deepcopy(configured_metadata)
            if isinstance(configured_metadata, dict)
            else {}
        )
        self._cache: dict[str, HonchoSession] = {}
        self._cache_lock = threading.RLock()
        self._delivery_lock = threading.RLock()
        self._peers_cache: dict[str, Any] = {}
        self._sessions_cache: dict[str, Any] = {}
        # Bumped (under _cache_lock) whenever _force_reauth rebuilds the client, so an
        # in-flight resolver never stores an object bound to the discarded client.
        self._client_generation = 0

        # Set when a call still fails auth after a forced token refresh; cleared on the next success.
        self._auth_failure: str | None = None
        self._auth_notice_emitted = False

        # Behavior knobs copied from config (HonchoClientConfig defaults when absent); the
        # observation booleans map 1:1 to Honcho's SessionPeerConfig toggles.
        for name, default in (
            ("write_frequency", "async"),
            ("dialectic_reasoning_level", "low"),
            ("dialectic_dynamic", True),
            ("dialectic_max_chars", 600),
            ("dialectic_max_input_chars", 10000),
            ("user_observe_me", True),
            ("user_observe_others", True),
            ("ai_observe_me", True),
            ("ai_observe_others", True),
        ):
            setattr(self, f"_{name}", getattr(config, name) if config else default)
        self._turn_counter: int = 0
        self._last_delivery_outcome = DeliveryOutcome(state=DeliveryState.NOOP)

        # Prefetch cache: session_key -> last context result (consumed once per turn).
        # Dialectic results are cached on the plugin side (HonchoMemoryProvider._prefetch_result)
        # so session-start prewarm and turn-driven fires share one source of truth.
        self._context_cache: dict[str, dict] = {}
        self._prefetch_cache_lock = threading.Lock()

        # Async write queue — the writer thread starts lazily on first enqueue
        # (_ensure_async_writer): constructing a manager must not spawn background
        # work or touch the network (unit tests build managers with mocked clients).
        self._async_queue: queue.Queue | None = (
            queue.Queue() if self._write_frequency == "async" else None
        )
        self._async_thread: threading.Thread | None = None
        self._async_thread_lock = threading.Lock()

    @property
    def last_delivery_outcome(self) -> DeliveryOutcome:
        """Latest content-free direct delivery result, including async failures."""
        return self._last_delivery_outcome

    def _retain_delivery_outcome(self, outcome: DeliveryOutcome) -> DeliveryOutcome:
        self._last_delivery_outcome = outcome
        return outcome

    @staticmethod
    def new_source_record_id() -> str:
        return str(uuid.uuid4())

    def _build_message_metadata(
        self,
        session: HonchoSession,
        role: str,
        *,
        created_at: datetime,
        source_record_id: str,
        source_session_id: str | None,
        chunk_index: int,
        chunk_count: int,
        secret_rejected: bool = False,
    ) -> dict[str, Any]:
        context = self._provenance_context
        extensions = copy.deepcopy(self._message_metadata.get("extensions", {}))
        if not isinstance(extensions, dict):
            extensions = {}
        static_hermes = extensions.get("hermes")
        hermes_extension = (
            dict(static_hermes) if isinstance(static_hermes, dict) else {}
        )
        hermes_extension.update({
            "platform": str(context.get("platform") or "cli"),
            "agent_context": str(context.get("agent_context") or "primary"),
            "agent_identity": str(context.get("agent_identity") or "default"),
            "agent_workspace": str(context.get("agent_workspace") or "hermes"),
            "user_id": context.get("user_id"),
            "user_id_alt": context.get("user_id_alt"),
            "user_name": context.get("user_name"),
            "chat_id": context.get("chat_id"),
            "chat_name": context.get("chat_name"),
            "chat_type": context.get("chat_type"),
            "thread_id": context.get("thread_id"),
            "gateway_session_key": context.get("gateway_session_key"),
            "session_title": context.get("session_title"),
            "honcho_session_id": session.honcho_session_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
        })
        extensions["hermes"] = hermes_extension
        metadata = copy.deepcopy(self._message_metadata)
        profile = str(context.get("agent_identity") or "default")
        agent_context = str(context.get("agent_context") or "primary")
        metadata.update({
            "schema": metadata.get("schema") or "hermes.provenance/v1",
            "event_id": str(uuid.uuid4()),
            "record_kind": "source_event",
            "source_kind": "user_statement"
            if role == "user"
            else "assistant_statement",
            "human_peer": session.user_peer_id,
            "ai_peer": session.assistant_peer_id,
            "interface": str(context.get("platform") or "cli"),
            "machine": _resolve_machine_identity(context),
            "runtime": f"hermes:{profile}:{agent_context}",
            "session_id": str(
                source_session_id or self._source_session_id or session.key
            ),
            "channel_id": context.get("chat_id"),
            "source_record_id": source_record_id,
            "observed_at": _rfc3339(created_at),
            "effective_at": metadata.get("effective_at"),
            "authority": session.user_peer_id
            if role == "user"
            else session.assistant_peer_id,
            "evidence_refs": metadata.get("evidence_refs")
            if isinstance(metadata.get("evidence_refs"), list)
            else [],
            "confidence": metadata.get("confidence"),
            "verification_state": metadata.get("verification_state") or "unverified",
            "review_state": metadata.get("review_state") or "captured",
            "sensitivity": "secret_rejected"
            if secret_rejected
            else metadata.get("sensitivity") or "private",
            "retention_class": metadata.get("retention_class") or "semantic",
            "derived_from": metadata.get("derived_from")
            if isinstance(metadata.get("derived_from"), list)
            else [],
            "supersedes": metadata.get("supersedes")
            if isinstance(metadata.get("supersedes"), list)
            else [],
            "superseded_by": metadata.get("superseded_by")
            if isinstance(metadata.get("superseded_by"), list)
            else [],
            "deletion_request_id": metadata.get("deletion_request_id"),
            "deleted_at": metadata.get("deleted_at"),
            "target_artifacts": metadata.get("target_artifacts")
            if isinstance(metadata.get("target_artifacts"), list)
            else [],
            "extensions": extensions,
        })
        return _redact_metadata_value(metadata)

    def add_source_message(
        self,
        session: HonchoSession,
        role: str,
        content: str,
        *,
        source_record_id: str | None = None,
        source_session_id: str | None = None,
        chunk_index: int = 0,
        chunk_count: int = 1,
        secret_rejected: bool = False,
    ) -> None:
        created_at = _utc_now()
        metadata = self._build_message_metadata(
            session,
            role,
            created_at=created_at,
            source_record_id=source_record_id or self.new_source_record_id(),
            source_session_id=source_session_id,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            secret_rejected=secret_rejected,
        )
        session.add_message(role, content, metadata=metadata, created_at=created_at)

    @property
    def honcho(self) -> Honcho:
        """The Honcho client, refreshing a near-expiry OAuth token in place. Always goes through
        ``get_honcho_client`` WITH this manager's bound config: a long session can't outlive its
        1h access token, and daemon threads can't see the ambient ContextVar profile, so a bare
        ``get_honcho_client()`` would migrate them onto the first-built profile.

        See #69123, #74065.
        """
        self._honcho = get_honcho_client(self._config)
        return self._honcho

    # ----- SDK object caches (generation-guarded against client rebuilds) -----

    def _cached_sdk_object(self, cache: dict[str, Any], key: str, fetch: Any) -> Any:
        """Get-or-fetch from ``cache``; a fetch that straddles a client rebuild is not cached."""
        while True:
            with self._cache_lock:
                if key in cache:
                    return cache[key]
                generation = self._client_generation
            obj = fetch()
            with self._cache_lock:
                if self._client_generation == generation:
                    return cache.setdefault(key, obj)
            # Client rebuilt mid-resolve: this object holds the discarded transport. Retry.

    def _sdk_session(self, session_id: str) -> Any:
        """Get or create the SDK session (cached until a client rebuild clears the cache)."""
        return self._cached_sdk_object(
            self._sessions_cache, session_id, lambda: self.honcho.session(session_id)
        )

    def _get_or_create_peer(self, peer_id: str) -> Any:
        """Get or create a Honcho peer (one get-or-create API call, then cached)."""
        return self._cached_sdk_object(
            self._peers_cache,
            peer_id,
            lambda: self._authed_call("peer setup", lambda: self.honcho.peer(peer_id)),
        )

    # ----- Session creation -----

    def _configure_session_peers(
        self, session_id: str, user_peer: Any, assistant_peer: Any
    ) -> bool:
        """add_peers with the local observation config, then adopt the server's effective
        config (set via the Honcho UI, it wins over local defaults). Observation booleans are
        manager-scoped, so the last session init wins. Returns False when auth died mid-way
        (already recorded by _authed_call)."""
        peers = (("user", user_peer), ("ai", assistant_peer))
        try:
            from honcho.session import SessionPeerConfig

            peer_entries = [
                (
                    peer,
                    SessionPeerConfig(
                        observe_me=getattr(self, f"_{kind}_observe_me"),
                        observe_others=getattr(self, f"_{kind}_observe_others"),
                    ),
                )
                for kind, peer in peers
            ]
            self._authed_call(
                "session peer setup",
                lambda: self._sdk_session(session_id).add_peers(peer_entries),
            )

            def _adopt_server_config() -> None:
                server_cfgs = self._authed_call(
                    "peer configuration read",
                    lambda: [
                        self._sdk_session(session_id).get_peer_configuration(peer)
                        for _, peer in peers
                    ],
                )
                for (kind, _), server_cfg in zip(peers, server_cfgs):
                    for field_name in ("observe_me", "observe_others"):
                        value = getattr(server_cfg, field_name)
                        if value is not None:
                            setattr(self, f"_{kind}_{field_name}", value)
                logger.debug(
                    "Honcho observation synced from server: user(me=%s,others=%s) ai(me=%s,others=%s)",
                    self._user_observe_me,
                    self._user_observe_others,
                    self._ai_observe_me,
                    self._ai_observe_others,
                )

            self._guarded(
                _adopt_server_config,
                None,
                logging.DEBUG,
                "Honcho get_peer_configuration failed (using local config): %s",
            )
        except HonchoAuthError:
            return False
        except Exception as e:
            logger.warning(
                "Honcho session '%s' add_peers failed (non-fatal): %s", session_id, e
            )
        return True

    def _load_existing_messages(self, session_id: str) -> list:
        """Load prior messages via context() (one call for messages + metadata), oldest first."""
        try:
            ctx = self._authed_call(
                "session context load",
                lambda: self._sdk_session(session_id).context(
                    summary=True, tokens=self._context_tokens
                ),
            )
            existing_messages = ctx.messages or []
            if len(existing_messages) > 1:
                timestamps = [m.created_at for m in existing_messages if m.created_at]
                if timestamps and timestamps != sorted(timestamps):
                    logger.warning(
                        "Honcho messages not chronologically ordered for session '%s', sorting",
                        session_id,
                    )
                    existing_messages = sorted(
                        existing_messages, key=lambda m: m.created_at or datetime.min
                    )
            if existing_messages:
                logger.info(
                    "Honcho session '%s' retrieved (%d existing messages)",
                    session_id,
                    len(existing_messages),
                )
            else:
                logger.info("Honcho session '%s' created (new)", session_id)
            return existing_messages
        except HonchoAuthError:
            logger.warning(
                "Honcho session '%s' loaded without server context: auth failed",
                session_id,
            )
        except Exception as e:
            logger.warning(
                "Honcho session '%s' loaded (failed to fetch context: %s)",
                session_id,
                e,
            )
        return []

    def _get_or_create_honcho_session(
        self, session_id: str, user_peer: Any, assistant_peer: Any
    ) -> tuple[Any, list]:
        """(honcho_session, existing_messages) with peers configured; a cached session yields no messages."""
        with self._cache_lock:
            if session_id in self._sessions_cache:
                logger.debug("Honcho session '%s' retrieved from cache", session_id)
                return self._sessions_cache[session_id], []

        self._authed_call("session setup", lambda: self._sdk_session(session_id))
        existing_messages: list = (
            self._load_existing_messages(session_id)
            if self._configure_session_peers(session_id, user_peer, assistant_peer)
            else []
        )

        with self._cache_lock:
            honcho_session = self._sessions_cache.get(session_id)
        if honcho_session is None:
            # A mid-init client rebuild dropped the cached session; resolve a fresh one.
            honcho_session = self._authed_call(
                "session setup", lambda: self._sdk_session(session_id)
            )
        return honcho_session, existing_messages

    def get_or_create(self, key: str) -> HonchoSession:
        """Get an existing session or create a new one for ``key`` (usually channel:chat_id)."""
        with self._cache_lock:
            if key in self._cache:
                logger.debug("Local session cache hit: %s", key)
                return self._cache[key]

        # Gateway sessions normally use the platform-native runtime identity so multi-user
        # bots scope memory per user; config can alias/prefix it, or pinPeerName pins all
        # identities to peerName for single-user deployments (see _resolve_user_peer_id).
        # Determine peer IDs — no lock needed (read-only, no shared state mutation). See #14984.
        user_peer_id = self._resolve_user_peer_id(key)
        assistant_peer_id = self._sanitize_id(
            self._config.ai_peer if self._config else "hermes-assistant"
        )

        # All expensive I/O outside the lock — Honcho's persistence is source of truth.
        honcho_session_id = self._sanitize_id(key)
        user_peer = self._get_or_create_peer(user_peer_id)
        assistant_peer = self._get_or_create_peer(assistant_peer_id)
        _, existing_messages = self._get_or_create_honcho_session(
            honcho_session_id, user_peer, assistant_peer
        )

        session = HonchoSession(
            key=key,
            user_peer_id=user_peer_id,
            assistant_peer_id=assistant_peer_id,
            honcho_session_id=honcho_session_id,
            messages=[
                {
                    "role": "assistant" if msg.peer_id == assistant_peer_id else "user",
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else "",
                    "created_at": msg.created_at,
                    "metadata": copy.deepcopy(getattr(msg, "metadata", None) or {}),
                    "_synced": True,
                }
                for msg in existing_messages
            ],
        )
        with self._cache_lock:
            self._cache[key] = session
        return session

    # ----- Writes -----

    def _flush_session(self, session: HonchoSession) -> DeliveryOutcome:
        """The single structured boundary for direct Honcho message delivery."""
        with self._delivery_lock:
            return self._flush_session_locked(session)

    def _flush_session_locked(self, session: HonchoSession) -> DeliveryOutcome:
        """Deliver one session while holding the manager-wide delivery lock."""
        if not session.messages:
            return self._retain_delivery_outcome(
                DeliveryOutcome(state=DeliveryState.NOOP)
            )

        new_messages = [m for m in session.messages if not m.get("_synced")]
        if not new_messages:
            return self._retain_delivery_outcome(
                DeliveryOutcome(state=DeliveryState.NOOP)
            )

        # Backward-compatible guard for callers that populated the local cache
        # directly. Assign once before the first attempt so retries keep IDs.
        for message in new_messages:
            if not isinstance(message.get("metadata"), dict) or not message[
                "metadata"
            ].get("event_id"):
                created_at = message.get("created_at")
                if not isinstance(created_at, datetime):
                    created_at = _utc_now()
                    message["created_at"] = created_at
                message["metadata"] = self._build_message_metadata(
                    session,
                    message["role"],
                    created_at=created_at,
                    source_record_id=self.new_source_record_id(),
                    source_session_id=None,
                    chunk_index=0,
                    chunk_count=1,
                )

        # Validate exact content + metadata together before SDK message construction.
        semantic_payload = [
            {"content": message.get("content"), "metadata": message.get("metadata")}
            for message in new_messages
        ]
        if not is_safe_semantic_payload(semantic_payload):
            for message in new_messages:
                message["_synced"] = False
            logger.warning(
                "Honcho semantic payload rejected before direct delivery count=%d",
                len(new_messages),
            )
            return self._retain_delivery_outcome(
                DeliveryOutcome(
                    state=DeliveryState.FAILED,
                    attempted_count=len(new_messages),
                    pending_count=len(new_messages),
                    error_category="semantic_payload_rejected",
                )
            )

        # Resolved inside the operation so a retry after a client rebuild gets fresh objects.
        def _sync_messages() -> int:
            user_peer = self._get_or_create_peer(session.user_peer_id)
            assistant_peer = self._get_or_create_peer(session.assistant_peer_id)
            honcho_session = self._sessions_cache.get(session.honcho_session_id)
            if honcho_session is None:
                honcho_session, _ = self._get_or_create_honcho_session(
                    session.honcho_session_id, user_peer, assistant_peer
                )
            honcho_messages = [
                (user_peer if m["role"] == "user" else assistant_peer).message(
                    m["content"],
                    metadata=_redact_metadata_value(m.get("metadata") or {}),
                    created_at=m.get("created_at"),
                )
                for m in new_messages
            ]
            honcho_session.add_messages(honcho_messages)
            return len(honcho_messages)

        try:
            synced = self._authed_semantic_write(
                "message sync", semantic_payload, _sync_messages
            )
            for msg in new_messages:
                msg["_synced"] = True
            pending = sum(
                1 for message in session.messages if not message.get("_synced")
            )
            logger.debug("Honcho direct delivery succeeded count=%d", synced)
            with self._cache_lock:
                self._cache[session.key] = session
            return self._retain_delivery_outcome(
                DeliveryOutcome(
                    state=DeliveryState.DELIVERED,
                    attempted_count=len(new_messages),
                    delivered_count=synced,
                    pending_count=pending,
                )
            )
        except Exception as exc:
            for msg in new_messages:
                msg["_synced"] = False
            pending = sum(
                1 for message in session.messages if not message.get("_synced")
            )
            category, status = classify_delivery_error(exc)
            logger.error(
                "Honcho direct delivery terminal failure category=%s status=%s attempted=%d pending=%d",
                category,
                status if status is not None else "none",
                len(new_messages),
                pending,
            )
            with self._cache_lock:
                self._cache[session.key] = session
            return self._retain_delivery_outcome(
                DeliveryOutcome(
                    state=DeliveryState.FAILED,
                    attempted_count=len(new_messages),
                    delivered_count=0,
                    pending_count=pending,
                    error_category=category,
                    http_status=status,
                )
            )

    def _try_flush(self, session: HonchoSession, level: int, msg: str) -> bool:
        """_flush_session that logs (never raises) a failure; False when the batch didn't land."""
        try:
            if self._flush_session(session):
                return True
            logger.log(level, msg)
        except Exception as e:
            logger.log(level, msg + ": %s", e)
        return False

    def _async_writer_loop(self) -> None:
        """Background daemon thread: drains the async write queue."""
        while True:
            try:
                item = self._async_queue.get(timeout=5)
                if item is _ASYNC_SHUTDOWN:
                    break

                try:
                    outcome = self._retain_delivery_outcome(self._flush_session(item))
                except Exception as exc:
                    category, status = classify_delivery_error(exc)
                    pending = sum(
                        1 for message in item.messages if not message.get("_synced")
                    )
                    outcome = self._retain_delivery_outcome(
                        DeliveryOutcome(
                            state=DeliveryState.FAILED,
                            attempted_count=pending,
                            delivered_count=0,
                            pending_count=pending,
                            error_category=category,
                            http_status=status,
                        )
                    )
                if outcome.state is DeliveryState.FAILED:
                    logger.error(
                        "Honcho async direct delivery terminal failure category=%s status=%s "
                        "attempted=%d pending=%d; messages remain retryable",
                        outcome.error_category,
                        outcome.http_status
                        if outcome.http_status is not None
                        else "none",
                        outcome.attempted_count,
                        outcome.pending_count,
                    )
            except queue.Empty:
                continue
            except Exception as exc:
                category, status = classify_delivery_error(exc)
                logger.error(
                    "Honcho async writer error category=%s status=%s",
                    category,
                    status if status is not None else "none",
                )

    def save(self, session: HonchoSession) -> DeliveryOutcome | None:
        """Save messages to Honcho, respecting write_frequency.

        write_frequency modes:
          "async"   — enqueue for background thread (zero blocking, zero token cost)
          "turn"    — flush synchronously every turn
          "session" — defer until flush_session() is called explicitly
          N (int)   — flush every N turns
        """
        self._turn_counter += 1
        wf = self._write_frequency

        if wf == "async":
            if self._async_queue is not None:
                self._ensure_async_writer()
                self._async_queue.put(session)
        elif wf == "turn":
            return self._flush_session(session)
        elif wf == "session":
            # Accumulate; caller must call flush_all() at session end
            pass
        elif isinstance(wf, int) and wf > 0:
            if self._turn_counter % wf == 0:
                return self._flush_session(session)
        return None

    def flush_all(self) -> DeliveryOutcome:
        """Flush all pending unsynced messages for all cached sessions.

        Called at session end for "session" write_frequency, or to force
        a sync before process exit regardless of mode.
        """
        with self._cache_lock:
            sessions = list(self._cache.values())

        # Drain queued references before delivery, then de-duplicate sessions.
        if self._async_queue is not None:
            while not self._async_queue.empty():
                try:
                    item = self._async_queue.get_nowait()
                    if item is not _ASYNC_SHUTDOWN:
                        sessions.append(item)
                except queue.Empty:
                    break

        outcomes = []
        seen: set[int] = set()
        for session in sessions:
            if id(session) in seen:
                continue
            seen.add(id(session))
            try:
                outcomes.append(self._flush_session(session))
            except Exception as exc:
                category, status = classify_delivery_error(exc)
                pending = sum(
                    1 for message in session.messages if not message.get("_synced")
                )
                outcomes.append(
                    self._retain_delivery_outcome(
                        DeliveryOutcome(
                            state=DeliveryState.FAILED,
                            attempted_count=pending,
                            delivered_count=0,
                            pending_count=pending,
                            error_category=category,
                            http_status=status,
                        )
                    )
                )
                logger.error(
                    "Honcho flush_all terminal failure category=%s status=%s pending=%d",
                    category,
                    status if status is not None else "none",
                    pending,
                )
        return self._retain_delivery_outcome(DeliveryOutcome.aggregate(outcomes))

    def _ensure_async_writer(self) -> None:
        """Start the async writer on first enqueue (idempotent, thread-safe)."""
        if self._async_thread is not None and self._async_thread.is_alive():
            return
        with self._async_thread_lock:
            if self._async_thread is None or not self._async_thread.is_alive():
                self._async_thread = spawn_context_thread(
                    self._async_writer_loop, name="honcho-async-writer"
                )
                self._async_thread.start()

    def stop_async_writer(self) -> None:
        """Join the async writer WITHOUT flushing (saveMessages: false must still exit cleanly)."""
        if (
            self._async_queue is not None
            and self._async_thread is not None
            and self._async_thread.is_alive()
        ):
            self._async_queue.put(_ASYNC_SHUTDOWN)
            self._async_thread.join(timeout=10)

    def shutdown(self) -> None:
        """Flush everything, then stop the async writer thread."""
        if self._async_queue is not None:
            self.flush_all()
            self.stop_async_writer()

    # ----- Prefetch cache -----

    def prefetch_context(
        self, session_key: str, user_message: str | None = None
    ) -> None:
        """Fire get_prefetch_context in a background thread; consumed next turn via pop_context_result()."""

        def _run():
            result = self.get_prefetch_context(session_key, user_message)
            if result:
                self.set_context_result(session_key, result)

        spawn_context_thread(_run, name="honcho-context-prefetch").start()

    def set_context_result(self, session_key: str, result: dict[str, str]) -> None:
        """Store a prefetched context result in a thread-safe way."""
        if not result:
            return
        with self._prefetch_cache_lock:
            self._context_cache[session_key] = result

    def pop_context_result(self, session_key: str) -> dict[str, str]:
        """Return and clear the cached context result ({} if none ready yet)."""
        with self._prefetch_cache_lock:
            return self._context_cache.pop(session_key, {})


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Callable  # noqa: F401,E402
from pathlib import Path  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import re  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
