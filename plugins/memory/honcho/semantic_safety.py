"""Fail-closed protection for payloads crossing Honcho semantic-storage APIs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

from agent.memory_manager import sanitize_context
from agent.redact import redact_sensitive_text


SECRET_REJECTED_CONTENT = (
    "[Secret-shaped content rejected before Honcho semantic storage.]"
)
_MAX_DEPTH = 12
_MAX_NODES = 512
_MAX_SCALARS = 256
_MAX_TOTAL_CHARS = 131_072
_MAX_SCALAR_CHARS = 65_536
_MAX_PAIR_EDGE_CHARS = 256
_REDACTED_METADATA_SENTINEL = "«redacted-metadata»"
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

# Prefixes which may be separated from an opaque body by whitespace or JSON
# field boundaries. Requiring a >=10-character token body preserves the same
# false-positive floor as the core redactor (so prose such as "the sk- prefix"
# remains safe).
_SPLIT_PREFIX_RE = re.compile(
    r"(?i)(?:sk-|sk_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|xox[baprs]-|"
    r"AKIA|ASIA|AIza|hf_|npm_|pypi-|rk_live_|sk_live_|sk_test_|rk_test_|"
    r"dop_v1_|doo_v1_|am_|tvly-|exa_|gsk_|syt_|retaindb_|hsk-|mem0_|brv_|xai-)"
    r"\s+([A-Za-z0-9_-]{10,})"
)
_PREFIX_AT_END_RE = re.compile(
    r"(?i)(?:sk-|sk_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|xox[baprs]-|"
    r"AKIA|ASIA|AIza|hf_|npm_|pypi-|rk_live_|sk_live_|sk_test_|rk_test_|"
    r"dop_v1_|doo_v1_|am_|tvly-|exa_|gsk_|syt_|retaindb_|hsk-|mem0_|brv_|xai-)$"
)


def _normalize_key(key: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_credential_shaped_key(key: str) -> bool:
    """Match secret-bearing fields, but not references or benign counters."""
    normalized = _normalize_key(key)
    if normalized.endswith(_SAFE_CREDENTIAL_REFERENCE_SUFFIXES):
        return False
    parts = normalized.split("_")
    compact = "".join(parts)
    return (
        any(
            part in {"password", "passwd", "secret", "token", "authorization"}
            for part in parts
        )
        or any(
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
        )
        or any(
            parts[index : index + 2] in (["api", "key"], ["private", "key"])
            for index in range(max(0, len(parts) - 1))
        )
    )


class SemanticPayloadRejected(ValueError):
    """Content-free signal that a semantic payload is unsafe or uninspectable."""

    def __init__(self) -> None:
        super().__init__("semantic payload rejected")


def protect_semantic_content(content: str) -> tuple[str, bool]:
    """Replace a whole semantic record when mandatory redaction would change it."""
    clean = sanitize_context(content or "").strip()
    if not clean:
        return "", False
    redacted = redact_sensitive_text(clean, force=True, redact_url_credentials=True)
    return (SECRET_REJECTED_CONTENT, True) if redacted != clean else (clean, False)


def _collect_scalars(
    value: Any,
) -> tuple[list[str], list[str], list[tuple[str, str]]] | None:
    """Return bounded deterministic tokens and key/value relations; fail closed."""
    scalars: list[str] = []
    values: list[str] = []
    relationships: list[tuple[str, str]] = []
    active: set[int] = set()
    nodes = 0
    total_chars = 0

    def visit(item: Any, depth: int, *, is_key: bool = False) -> bool:
        nonlocal nodes, total_chars
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            return False
        if item is None or isinstance(item, (bool, int, float)):
            return True
        if isinstance(item, bytes):
            if len(item) > _MAX_SCALAR_CHARS:
                return False
            try:
                text = item.decode("utf-8")
            except UnicodeDecodeError:
                return False
        elif isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                return False
            # Each entry necessarily visits at least its key and value. Reject
            # oversized mappings before materializing or sorting their entries so
            # work and memory remain bounded by the global node budget.
            try:
                entry_count = len(item)
            except Exception:
                return False
            if entry_count > (_MAX_NODES - nodes) // 2:
                return False
            active.add(identity)
            try:
                remaining_entries = (_MAX_NODES - nodes) // 2
                iterator = iter(item.items())
                bounded_entries = list(islice(iterator, remaining_entries + 1))
                if len(bounded_entries) > remaining_entries:
                    return False
                entries: list[tuple[tuple[str, str], Any, Any]] = []
                for key, nested in bounded_entries:
                    if isinstance(key, bytes):
                        try:
                            key_text = key.decode("utf-8")
                        except UnicodeDecodeError:
                            return False
                    elif isinstance(key, str):
                        key_text = key
                    elif key is None or isinstance(key, (bool, int, float)):
                        key_text = repr(key)
                    else:
                        return False
                    if (
                        isinstance(key, (str, bytes))
                        and _is_credential_shaped_key(key_text)
                        and nested != _REDACTED_METADATA_SENTINEL
                    ):
                        return False
                    if isinstance(nested, bytes):
                        try:
                            nested_text = nested.decode("utf-8")
                        except UnicodeDecodeError:
                            return False
                    elif isinstance(nested, str):
                        nested_text = nested
                    else:
                        nested_text = None
                    if nested_text is not None:
                        relationships.append((key_text, nested_text))
                    entries.append(((type(key).__name__, key_text), key, nested))
                for _order, key, nested in sorted(entries, key=lambda entry: entry[0]):
                    if not visit(key, depth + 1, is_key=True) or not visit(
                        nested, depth + 1
                    ):
                        return False
            finally:
                active.remove(identity)
            return True
        elif isinstance(item, Sequence):
            identity = id(item)
            if identity in active:
                return False
            active.add(identity)
            try:
                if not all(visit(nested, depth + 1) for nested in item):
                    return False
            finally:
                active.remove(identity)
            return True
        else:
            # SDK/domain objects are not a safe semantic payload surface. Callers
            # must project the transmitted content/metadata into plain JSON first.
            return False
        if len(text) > _MAX_SCALAR_CHARS or len(scalars) >= _MAX_SCALARS:
            return False
        total_chars += len(text)
        if total_chars > _MAX_TOTAL_CHARS:
            return False
        scalars.append(text)
        if not is_key:
            values.append(text)
        return True

    return (scalars, values, relationships) if visit(value, 0) else None


def _redactor_changes(text: str) -> bool:
    if not text:
        return False

    def close_split(match: re.Match[str]) -> str:
        # Whitespace after a documented prefix is common prose. A digit in the
        # opaque body controls false positives such as "sk- placeholders".
        return (
            re.sub(r"\s+", "", match.group(0))
            if any(character.isdigit() for character in match.group(1))
            else match.group(0)
        )

    normalized = _SPLIT_PREFIX_RE.sub(close_split, text)
    return (
        redact_sensitive_text(normalized, force=True, redact_url_credentials=True)
        != normalized
    )


def is_safe_semantic_payload(payload: Any) -> bool:
    """Inspect one complete semantic payload, including cross-field adjacency.

    The function is total: malformed, cyclic, unsupported, or over-limit values
    are rejected rather than raising or being partially inspected. It never
    returns, formats, or logs any payload material.
    """
    try:
        collected = _collect_scalars(payload)
        if collected is None:
            return False
        scalars, values, relationships = collected
        if any(_redactor_changes(item) for item in scalars):
            return False

        # The bounded traversal above already checks each scalar, each direct
        # key/value relationship, and the canonical value sequence. Do not walk
        # the original object again: custom Mapping implementations may return a
        # different or unbounded iterator on a second pass.

        # Canonical value adjacency covers ordered multi-fragment splits, while
        # direct key/value relationships ensure mapping keys are inspected too.
        if _redactor_changes("".join(values)) or _redactor_changes(
            " ".join(values)
        ):
            return False

        # Check every direct mapping relationship, then every ordered pairing
        # whose left side is a recognized secret prefix. This covers key/value
        # fragments and insertion-reordered prefix/body values without treating
        # unrelated metadata fields as if every possible permutation were
        # adjacent (which would create both quadratic noise and false positives).
        candidates = list(relationships)
        candidates.extend(
            (left, right)
            for left_index, left in enumerate(values)
            if _PREFIX_AT_END_RE.search(left)
            for right_index, right in enumerate(values)
            if left_index != right_index
        )
        for left, right in candidates:
            left_suffix = left[-_MAX_PAIR_EDGE_CHARS:]
            right_prefix = right[:_MAX_PAIR_EDGE_CHARS]
            if _redactor_changes(left_suffix + right_prefix) or _redactor_changes(
                left_suffix + " " + right_prefix
            ):
                return False
        return True
    except Exception:
        return False


def require_safe_semantic_payload(payload: Any) -> None:
    """Raise a content-free exception unless the whole payload is safe."""
    if not is_safe_semantic_payload(payload):
        raise SemanticPayloadRejected()


def contains_secret_shaped_content(value: Any) -> bool:
    """Compatibility predicate: unsafe includes secrets and uninspectable input."""
    return not is_safe_semantic_payload(value)
