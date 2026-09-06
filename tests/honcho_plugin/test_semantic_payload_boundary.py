"""Adversarial regressions for Honcho's total pre-network payload guard."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.semantic_safety import is_safe_semantic_payload
from plugins.memory.honcho.session import DeliveryState, HonchoSession, HonchoSessionManager

_BODY = "AGADORFAKESECRET1234567890ABCDEF"
_SECRET = "sk-" + _BODY
_REDACTED = "«redacted-metadata»"


def _manager() -> tuple[HonchoSessionManager, HonchoSession, MagicMock, MagicMock, MagicMock]:
    mgr = HonchoSessionManager(
        honcho=MagicMock(),
        config=HonchoClientConfig(enabled=True, api_key="test-key", write_frequency="turn"),
    )
    session = HonchoSession(
        key="reviewer-path",
        user_peer_id="reviewer-user",
        assistant_peer_id="reviewer-assistant",
        honcho_session_id="reviewer-session",
    )
    mgr._cache[session.key] = session
    sdk_session = MagicMock()
    user_peer = MagicMock()
    assistant_peer = MagicMock()
    mgr._sessions_cache[session.honcho_session_id] = sdk_session
    mgr._peers_cache[session.user_peer_id] = user_peer
    mgr._peers_cache[session.assistant_peer_id] = assistant_peer
    return mgr, session, sdk_session, user_peer, assistant_peer


@pytest.mark.parametrize(
    "payload",
    [
        ["sk-", _BODY],
        {"prefix": "sk-", "body": _BODY},
        {"outer": [{"prefix": "sk-"}, {"body": _BODY}]},
        "sk- " + _BODY,
        {"content": "safe", "metadata": {"pieces": ["sk-", _BODY]}},
        {"content": "safe", "source": "sk-", "filename": _BODY},
    ],
)
def test_split_secret_payloads_are_rejected_as_one_semantic_unit(payload):
    assert is_safe_semantic_payload(payload) is False


@pytest.mark.parametrize(
    "payload",
    [
        {"sk-": _BODY},
        {"body": _BODY, "prefix": "sk-"},
        {"z_prefix": "sk-", "a_body": _BODY},
    ],
)
def test_mapping_keys_and_reordered_values_cannot_reconstruct_a_secret(payload):
    assert is_safe_semantic_payload(payload) is False


@pytest.mark.parametrize(
    "credential_key",
    ["api_key", "password", "token", "client_secret", "authorization"],
)
def test_exact_internal_redacted_sentinel_is_safe_for_credential_keys(
    credential_key,
):
    assert is_safe_semantic_payload({credential_key: _REDACTED}) is True


@pytest.mark.parametrize(
    "credential_key",
    ["api_key", "password", "token", "client_secret", "authorization"],
)
@pytest.mark.parametrize(
    "lookalike",
    [
        _REDACTED + "suffix",
        "prefix" + _REDACTED,
        "redacted-metadata",
        "«REDACTED-METADATA»",
        "<redacted-metadata>",
    ],
)
def test_redacted_sentinel_concatenations_and_lookalikes_are_not_exempt(
    credential_key, lookalike
):
    assert is_safe_semantic_payload({credential_key: lookalike}) is False


@pytest.mark.parametrize(
    "payload",
    [
        ["sk-", "short"],
        {"prefix": "sk-", "description": "OpenAI key prefix documentation"},
        {"token_count": 42, "secret_reference": "vault://honcho/key"},
        {"content": "Use sk- placeholders in docs", "filename": "SOUL.md"},
    ],
)
def test_false_positive_controls_allow_benign_secret_vocabulary(payload):
    assert is_safe_semantic_payload(payload) is True


def test_total_validator_rejects_cycles_and_excessive_payloads_without_raising():
    recursive: list[object] = []
    recursive.append(recursive)
    assert is_safe_semantic_payload(recursive) is False
    assert is_safe_semantic_payload([str(index) for index in range(1000)]) is False
    assert is_safe_semantic_payload({str(index): "safe" for index in range(1000)}) is False


def test_mapping_that_underreports_length_is_bounded_before_sorting():
    class UnderreportedMapping(dict):
        visited = 0

        def __len__(self):
            return 0

        def items(self):
            for item in super().items():
                self.visited += 1
                yield item

    payload = UnderreportedMapping({str(index): "safe" for index in range(10_000)})
    assert is_safe_semantic_payload(payload) is False
    assert payload.visited <= 256


def test_custom_mapping_is_traversed_only_once():
    class SinglePassMapping(dict):
        calls = 0

        def items(self):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("semantic validator traversed mapping twice")
            return super().items()

    payload = SinglePassMapping({"content": "safe", "source": "user"})
    assert is_safe_semantic_payload(payload) is True
    assert payload.calls == 1


def test_direct_create_conclusion_rejects_before_sdk_call(caplog):
    mgr, session, _sdk_session, _user_peer, assistant_peer = _manager()
    scope = MagicMock()
    assistant_peer.conclusions_of.return_value = scope

    assert mgr.create_conclusion(session.key, _SECRET) is False
    scope.create.assert_not_called()
    assert _SECRET not in caplog.text


def test_direct_set_peer_card_rejects_split_secret_before_sdk_call(caplog):
    mgr, session, _sdk_session, _user_peer, assistant_peer = _manager()

    assert mgr.set_peer_card(session.key, ["sk-", _BODY]) is None
    assistant_peer.set_card.assert_not_called()
    assert _SECRET not in caplog.text


def test_raw_cached_flush_rejects_content_and_metadata_together_before_sdk_call(caplog):
    mgr, session, sdk_session, user_peer, _assistant_peer = _manager()
    session.messages.append(
        {
            "role": "user",
            "content": "sk-",
            "metadata": {"event_id": "event", "fragment": _BODY},
            "_synced": False,
        }
    )

    outcome = mgr._flush_session(session)

    assert outcome.state is DeliveryState.FAILED
    assert outcome.error_category == "semantic_payload_rejected"
    assert session.messages[0]["_synced"] is False
    user_peer.message.assert_not_called()
    sdk_session.add_messages.assert_not_called()
    assert _SECRET not in caplog.text


@pytest.mark.parametrize(
    "metadata",
    [
        {"event_id": "legacy-1", "sk-": _BODY},
        {"event_id": "legacy-2", "body": _BODY, "prefix": "sk-"},
    ],
)
def test_legacy_cached_flush_rejects_key_value_and_reordered_fragments_without_network(
    metadata, caplog
):
    mgr, session, sdk_session, user_peer, _assistant_peer = _manager()
    session.messages.append(
        {
            "role": "user",
            "content": "legacy cached message",
            "metadata": metadata,
            "_synced": False,
        }
    )

    outcome = mgr._flush_session(session)

    assert outcome.state is DeliveryState.FAILED
    assert outcome.error_category == "semantic_payload_rejected"
    assert session.messages[0]["_synced"] is False
    user_peer.message.assert_not_called()
    sdk_session.add_messages.assert_not_called()
    assert _SECRET not in caplog.text


def test_identity_seed_validates_source_filename_before_interpolation(caplog):
    mgr, session, sdk_session, _user_peer, assistant_peer = _manager()

    assert mgr.seed_ai_identity(session.key, "Benign identity", source=_SECRET) is False
    assistant_peer.message.assert_not_called()
    sdk_session.add_messages.assert_not_called()
    assert _SECRET not in caplog.text


def test_identity_seed_rejects_secret_split_between_source_and_content():
    mgr, session, sdk_session, _user_peer, assistant_peer = _manager()

    assert mgr.seed_ai_identity(session.key, _BODY, source="sk-") is False
    assistant_peer.message.assert_not_called()
    sdk_session.add_messages.assert_not_called()


def test_migration_validates_filename_and_metadata_before_network(tmp_path, monkeypatch):
    import plugins.memory.honcho.session_migration as migration

    mgr, session, sdk_session, _user_peer, _assistant_peer = _manager()
    assert mgr._config is not None
    mgr._config.peer_name = session.user_peer_id
    monkeypatch.setattr(
        migration,
        "_MEMORY_FILES",
        ((_SECRET, "safe-upload.md", "Benign description", "user"),),
    )
    (tmp_path / _SECRET).write_text("Benign migration content", encoding="utf-8")

    assert mgr.migrate_memory_files(session.key, str(tmp_path)) is False
    sdk_session.upload_file.assert_not_called()
