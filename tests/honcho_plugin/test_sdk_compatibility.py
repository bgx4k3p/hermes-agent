"""Honcho SDK pin consistency and the production API surface we depend on."""

from __future__ import annotations

import inspect
import re
import tomllib
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HONCHO_VERSION = "2.4.0"
HONCHO_SPEC = f"honcho-ai=={HONCHO_VERSION}"


def test_authoritative_honcho_sdk_pins_are_consistent():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["optional-dependencies"]["honcho"] == [HONCHO_SPEC]

    authoritative_surfaces = [
        "plugins/memory/honcho/cli.py",
        "plugins/memory/honcho/client.py",
        "plugins/memory/honcho/plugin.yaml",
        "plugins/memory/honcho/README.md",
        "optional-skills/autonomous-ai-agents/honcho/SKILL.md",
        "cli-config.yaml.example",
        "hermes_cli/doctor_state.py",
        "website/docs/user-guide/features/memory-providers.md",
        "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/features/memory-providers.md",
    ]
    for relative_path in authoritative_surfaces:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert HONCHO_SPEC in text, f"stale or unpinned Honcho install surface: {relative_path}"

    lazy_deps = (ROOT / "tools/lazy_deps.py").read_text(encoding="utf-8")
    cli = (ROOT / "plugins/memory/honcho/cli.py").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert f'"memory.honcho": ("{HONCHO_SPEC}",)' in lazy_deps
    assert HONCHO_SPEC in cli
    assert f'name = "honcho-ai"\nversion = "{HONCHO_VERSION}"' in lock
    assert re.search(
        rf'name = "honcho-ai", marker = "extra == \'honcho\'", specifier = "=={re.escape(HONCHO_VERSION)}"',
        lock,
    )


def test_installed_honcho_sdk_version_and_api_are_compatible():
    from honcho import Honcho
    from honcho.peer import Peer
    from honcho.session import Session

    assert version("honcho-ai") == HONCHO_VERSION
    expected_parameters = {
        (Honcho, "search"): {"query", "filters", "limit", "scope"},
        (Session, "upload_file"): {
            "file",
            "peer",
            "metadata",
            "configuration",
            "created_at",
        },
        (Session, "add_messages"): {"messages"},
        (Peer, "message"): {"content", "metadata", "configuration", "created_at"},
        (Peer, "search"): {"query", "filters", "limit"},
        (Peer, "conclusions_of"): {"target"},
        (Peer, "set_card"): {"peer_card", "target"},
    }
    for (owner, method), required in expected_parameters.items():
        actual = set(inspect.signature(getattr(owner, method)).parameters) - {"self"}
        assert required <= actual, f"{owner.__name__}.{method} API changed: {actual}"