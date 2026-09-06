"""Secret-shaped content must not cross the Honcho semantic-storage boundary."""

from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


SECRET = "sk-" + "AGADORFAKESECRET1234567890ABCDEF"
REJECTED = "[Secret-shaped content rejected before Honcho semantic storage.]"


def _ready_provider() -> tuple[HonchoMemoryProvider, MagicMock, MagicMock]:
    provider = HonchoMemoryProvider()
    provider._session_initialized = True
    provider._session_key = "agador-secret-boundary"
    provider._manager = MagicMock()
    provider._cron_skipped = False
    provider._recall_mode = "hybrid"
    provider._config = HonchoClientConfig(
        enabled=True,
        api_key="test-key",
        message_max_chars=25000,
        save_messages=True,
    )
    session = MagicMock()
    provider._manager.get_or_create.return_value = session
    provider._manager.new_source_record_id.return_value = "source-record"
    return provider, provider._manager, session


def test_sync_turn_replaces_entire_secret_side_with_non_reusable_marker():
    provider, manager, session = _ready_provider()

    provider.sync_turn(
        f"Please remember this credential {SECRET}",
        "I will not retain credentials.",
        session_id="source-session",
    )
    assert provider._sync_thread is not None
    provider._sync_thread.join(timeout=2)

    calls = manager.add_source_message.call_args_list
    assert calls[0].args == (session, "user", REJECTED)
    assert calls[0].kwargs["secret_rejected"] is True
    assert calls[1].args == (session, "assistant", "I will not retain credentials.")
    assert calls[1].kwargs["secret_rejected"] is False
    assert SECRET not in repr(calls)


def test_rejected_source_event_is_labeled_secret_rejected():
    config = HonchoClientConfig(
        enabled=True,
        api_key="test-key",
        message_metadata={
            "schema": "agador.provenance/v1",
            "sensitivity": "private",
        },
    )
    manager = HonchoSessionManager(config=config)
    session = HonchoSession(
        key="secret-boundary",
        user_peer_id="bgx4k3p",
        assistant_peer_id="hermes-infra",
        honcho_session_id="secret-boundary",
    )

    manager.add_source_message(
        session,
        "user",
        REJECTED,
        source_record_id="source-record",
        source_session_id="source-session",
        secret_rejected=True,
    )

    stored = session.messages[0]
    assert stored["content"] == REJECTED
    assert stored["metadata"]["sensitivity"] == "secret_rejected"


def test_secret_conclusion_is_rejected_before_manager_call():
    provider, manager, _session = _ready_provider()

    result = provider.handle_tool_call(
        "honcho_conclude",
        {"conclusion": f"Store this token {SECRET}"},
    )

    assert "secret-shaped content" in result.lower()
    manager.create_conclusion.assert_not_called()
    assert SECRET not in result


def test_secret_peer_card_is_rejected_before_manager_call():
    provider, manager, _session = _ready_provider()

    result = provider.handle_tool_call(
        "honcho_profile",
        {"card": ["Role: infrastructure", f"Credential: {SECRET}"]},
    )

    assert "secret-shaped content" in result.lower()
    manager.set_peer_card.assert_not_called()
    assert SECRET not in result


def test_secret_builtin_memory_mirror_is_not_written_to_honcho():
    provider, manager, _session = _ready_provider()

    provider.on_memory_write(
        "add",
        "user",
        f"User credential is {SECRET}",
    )

    assert getattr(provider, "_memwrite_thread", None) is None
    manager.create_conclusion.assert_not_called()
