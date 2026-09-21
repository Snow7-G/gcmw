"""Deployment bundle checks (``botscreen-public/deploy``).

The bundle is *configuration*, so almost everything here is a static assertion
about the tracked files — but two of them are real behaviour checks:

- the Agent API entry refuses to bind anything but loopback **even when asked**;
- ``render-assets.sh`` is executed and its output inspected (no placeholder may
  survive, shell files must stay syntactically valid).

What these tests are for: the previous deployment drifted from the repository
because the units only existed on the host. A unit that silently reverted to
``Restart=always`` or lost ``DEBUG=1`` would resurrect the kiosk lock and the
"window pops back up after close" behaviour, so those two properties are pinned
here rather than left to a README reminder.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
SYSTEMD_DIR = DEPLOY_DIR / "systemd"
BIN_DIR = DEPLOY_DIR / "bin"
CONFIG_DIR = DEPLOY_DIR / "config"

_HEX40 = re.compile(r"\b[0-9a-f]{40}\b")
_SECRETISH_NAMES = ("API_KEY", "TOKEN", "SECRET", "CREDENTIAL", "PASSWORD")
_PLACEHOLDERISH = ("YOUR_", "<", "CHANGEME", "set-me")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _unit(name: str) -> str:
    return _read(SYSTEMD_DIR / name)


def _directive(unit_text: str, key: str) -> list[str]:
    return [
        line.split("=", 1)[1].strip()
        for line in unit_text.splitlines()
        if line.startswith(f"{key}=")
    ]


def _load_agent_entry():
    # Import WITHOUT writing bytecode: a stray __pycache__ next to the tracked
    # entry point is exactly what used to break the renderer on a platform whose
    # Python caches file-loaded modules (Linux/3.11; macOS/3.14 does not).
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return _load_agent_entry_uncached()
    finally:
        sys.dont_write_bytecode = previous


def _load_agent_entry_uncached():
    spec = importlib.util.spec_from_file_location(
        "deploy_run_demo_agent_api", BIN_DIR / "run-demo-agent-api.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBundleHasNoHostSpecificOrSecretValues:
    @pytest.mark.parametrize(
        "path", sorted(p for p in DEPLOY_DIR.rglob("*") if p.is_file())
    )
    def test_no_release_sha_anywhere(self, path):
        # A pinned SHA in a template is what forced a full re-deploy every time
        # the release moved; paths must go through <root>/current instead.
        assert _HEX40.search(_read(path)) is None, (
            f"{path.name} hardcodes a release SHA"
        )

    def test_no_private_keys_logs_or_env_files(self):
        names = {p.name for p in DEPLOY_DIR.rglob("*") if p.is_file()}
        assert "id_ed25519" not in names
        assert not any(name.endswith(".env") for name in names), "only *.env.example"
        assert not any(name.endswith((".log", ".crash")) for name in names)

    def test_no_lan_addresses_or_ssh_material(self):
        for path in DEPLOY_DIR.rglob("*"):
            if not path.is_file():
                continue
            text = _read(path)
            assert "BEGIN OPENSSH" not in text
            assert "ssh-ed25519" not in text
            assert "172.20.10." not in text, f"{path.name} carries a LAN address"

    def test_units_never_embed_a_secret(self):
        for name in (
            "qa-server.service",
            "botscreen.service",
            "gcmw-agent-demo.service",
        ):
            text = _unit(name)
            assert "Bearer " not in text
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or "=" not in stripped:
                    continue
                key, _, value = stripped.partition("=")
                if any(word in key.upper() for word in _SECRETISH_NAMES):
                    assert not value or any(ph in value for ph in _PLACEHOLDERISH), (
                        f"{name} sets {key} to a real-looking value"
                    )


class TestVoiceBridgeUnit:
    def test_pins_loopback_explicitly(self):
        # Double gate: qa_server.py already defaults to loopback; the unit repeats
        # it so a change to the code default cannot expose the LAN silently.
        exec_start = _directive(_unit("qa-server.service"), "ExecStart")
        assert len(exec_start) == 1
        assert exec_start[0].endswith(
            "-m uvicorn qa_server:app --host 127.0.0.1 --port 8000"
        )

    def test_never_binds_all_interfaces(self):
        assert "0.0.0.0" not in _unit("qa-server.service")

    def test_uses_the_voice_bridge_environment_file(self):
        assert _directive(_unit("qa-server.service"), "EnvironmentFile") == [
            "@@GCMW_CONFIG_DIR@@/voice-bridge.env"
        ]

    def test_clean_exit_is_not_resurrected(self):
        unit = _unit("qa-server.service")
        assert _directive(unit, "Restart") == ["on-failure"]
        assert "Restart=always" not in unit


class TestUiUnitIsAClosableWindow:
    def test_debug_is_pinned(self):
        assert (
            _directive(_unit("botscreen.service"), "Environment").count("DEBUG=1") == 1
        )

    def test_restart_is_on_failure_not_always(self):
        unit = _unit("botscreen.service")
        assert _directive(unit, "Restart") == ["on-failure"]
        assert "Restart=always" not in unit, (
            "`always` respawns the window after a user closes it"
        )

    def test_display_is_not_hardcoded(self):
        unit = _unit("botscreen.service")
        assert "Environment=DISPLAY=" not in unit, (
            "the kiosk display does not decorate windows — DISPLAY must be resolved"
        )
        assert "DISPLAY=:0" not in unit

    def test_exec_goes_through_the_display_resolver(self):
        exec_start = _directive(_unit("botscreen.service"), "ExecStart")
        assert exec_start == ["@@GCMW_DEPLOY_DIR@@/bin/start-botscreen-ui"]

    def test_waits_for_an_x_socket_before_starting(self):
        pre = _directive(_unit("botscreen.service"), "ExecStartPre")
        assert pre and all("/tmp/.X11-unix/X" in entry for entry in pre)


class TestAgentApiUnit:
    def test_is_a_user_unit(self):
        assert _directive(_unit("gcmw-agent-demo.service"), "WantedBy") == [
            "default.target"
        ]

    def test_is_hardened(self):
        unit = _unit("gcmw-agent-demo.service")
        for flag in (
            "NoNewPrivileges=true",
            "RestrictSUIDSGID=true",
            "LockPersonality=true",
            "UMask=0077",
        ):
            assert flag in unit

    def test_environment_file_is_optional_so_mock_mode_needs_no_secrets(self):
        assert _directive(_unit("gcmw-agent-demo.service"), "EnvironmentFile") == [
            "-@@GCMW_CONFIG_DIR@@/demo-agent-api.env"
        ]

    def test_runs_the_tracked_entry_point(self):
        exec_start = _directive(_unit("gcmw-agent-demo.service"), "ExecStart")
        assert exec_start == [
            "@@GCMW_VENV_PYTHON@@ @@GCMW_DEPLOY_DIR@@/bin/run-demo-agent-api.py"
        ]


class TestWindowLauncher:
    @pytest.mark.parametrize("forbidden", ["nohup", "setsid", "&>", " & "])
    def test_never_detaches(self, forbidden):
        assert forbidden not in _read(BIN_DIR / "start-botscreen-ui")

    def test_execs_so_the_exit_code_reaches_systemd(self):
        assert "\nexec " in _read(BIN_DIR / "start-botscreen-ui")

    def test_defaults_debug_on_for_manual_launches(self):
        assert 'export DEBUG="${DEBUG:-1}"' in _read(BIN_DIR / "start-botscreen-ui")

    def test_resolves_the_display_by_owner(self):
        text = _read(BIN_DIR / "start-botscreen-ui")
        assert "stat -c %u" in text and "id -u" in text

    def test_fails_loudly_instead_of_guessing(self):
        text = _read(BIN_DIR / "start-botscreen-ui")
        assert "exit 2" in text
        assert 'DISPLAY=":0"' not in text


class TestAgentApiEntry:
    def test_origin_allowlist_is_exact(self):
        module = _load_agent_entry()
        assert module.ALLOWED_ORIGINS == (
            "null",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        )
        assert "*" not in module.ALLOWED_ORIGINS

    def test_methods_and_headers_are_the_expected_minimum(self):
        module = _load_agent_entry()
        assert module.ALLOWED_METHODS == ("GET", "POST", "DELETE", "OPTIONS")
        assert module.ALLOWED_HEADERS == (
            "Authorization",
            "Content-Type",
            "Last-Event-ID",
        )

    def test_defaults_are_loopback(self):
        module = _load_agent_entry()
        assert module.DEFAULT_HOST == "127.0.0.1"
        assert module.DEFAULT_PORT == 8001

    def test_refuses_to_bind_anything_but_loopback(self, capsys):
        """Real behaviour: the guard fires before any server is started."""
        module = _load_agent_entry()
        assert module.main(["--host", "0.0.0.0"]) == 2
        assert "loopback only" in capsys.readouterr().err

    def test_reports_a_missing_assembly_instead_of_starting_halfway(
        self, tmp_path, capsys
    ):
        module = _load_agent_entry()
        assert module.main(["--server-dir", str(tmp_path)]) == 2
        assert "demo assembly not found" in capsys.readouterr().err


class TestEnvironmentExamples:
    @pytest.mark.parametrize(
        "name", ["demo-agent-api.env.example", "voice-bridge.env.example"]
    )
    def test_secret_named_keys_hold_placeholders_only(self, name):
        for line in _read(CONFIG_DIR / name).splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if any(word in key.upper() for word in _SECRETISH_NAMES) and value:
                assert any(ph in value for ph in _PLACEHOLDERISH), (
                    f"{name} gives {key} a real-looking value"
                )

    def test_voice_bridge_documents_the_exact_base_url_contract(self):
        text = _read(CONFIG_DIR / "voice-bridge.env.example")
        assert "GCMW_VOICE_AGENT_API_BASE=http://127.0.0.1:8001/api/v1" in text
        assert "MUST be exactly" in text

    def test_agent_api_defaults_to_the_offline_provider(self):
        text = _read(CONFIG_DIR / "demo-agent-api.env.example")
        assert "GCMW_ACTIVE_PROVIDER=mock" in text


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None,
    reason="render-assets.sh needs bash + python3",
)
class TestRenderScript:
    def _render(self, tmp_path: Path, bundle: Path | None = None) -> Path:
        out = tmp_path / "rendered"
        completed = subprocess.run(
            [  # check=False on purpose: the helper's output IS the diagnosis
                "bash",
                str((bundle or DEPLOY_DIR) / "bin" / "render-assets.sh"),
                "--out",
                str(out),
                "--root",
                "/opt/gcmw",
                "--user",
                "demo-user",
                "--home",
                "/home/demo-user",
                "--deploy-dir",
                "/opt/gcmw/deploy/current",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            # Surface the helper's own output: a bare CalledProcessError hides
            # WHICH step failed, which is exactly what differs between platforms.
            pytest.fail(
                f"render-assets.sh exited {completed.returncode}\n"
                f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
            )
        self.helper_stdout = completed.stdout
        return out

    def test_leaves_no_placeholder_behind(self, tmp_path):
        out = self._render(tmp_path)
        complete = re.compile(r"@@GCMW_[A-Z_]+@@")
        leftovers = [
            f"{path.relative_to(out)}: {line}"
            for path in out.rglob("*")
            if path.is_file()
            for line in _read(path).splitlines()
            if complete.search(line)
        ]
        assert leftovers == []

    def test_renders_the_values_it_was_given(self, tmp_path):
        out = self._render(tmp_path)
        unit = _read(out / "systemd" / "botscreen.service")
        assert "ExecStart=/opt/gcmw/deploy/current/bin/start-botscreen-ui" in unit
        assert "User=demo-user" in unit
        assert "Environment=HOME=/home/demo-user" in unit
        launcher = _read(out / "bin" / "start-botscreen-ui")
        assert 'exec "/opt/gcmw/current/botscreen"' in launcher

    def test_rendered_shell_files_stay_valid(self, tmp_path):
        out = self._render(tmp_path)
        for name in ("start-botscreen-ui", "render-assets.sh"):
            subprocess.run(
                ["bash", "-n", str(out / "bin" / name)], check=True, capture_output=True
            )

    def test_rendered_units_keep_the_two_pinned_properties(self, tmp_path):
        out = self._render(tmp_path)
        unit = _read(out / "systemd" / "botscreen.service")
        assert "Environment=DEBUG=1" in unit
        assert "Restart=on-failure" in unit
        bridge = _read(out / "systemd" / "qa-server.service")
        assert "--host 127.0.0.1 --port 8000" in bridge

    def test_renders_exactly_the_three_files_of_bin(self, tmp_path):
        out = self._render(tmp_path)
        assert sorted(path.name for path in (out / "bin").iterdir()) == [
            "render-assets.sh",
            "run-demo-agent-api.py",
            "start-botscreen-ui",
        ]

    def test_stray_bytecode_cache_does_not_break_rendering(self, tmp_path):
        """Regression for a real CI-only failure.

        Importing the entry point makes CPython cache it beside its source
        (`bin/__pycache__/*.pyc`, observed on 3.11) and that blob CONTAINS the
        placeholder strings, so a renderer that walks the directory and decodes
        everything as UTF-8 dies with UnicodeDecodeError. Non-templates must be
        skipped, and never copied into the output.
        """
        bundle = tmp_path / "bundle"
        shutil.copytree(DEPLOY_DIR, bundle)
        cache = bundle / "bin" / "__pycache__"
        cache.mkdir()
        (cache / "run-demo-agent-api.cpython-311.pyc").write_bytes(
            b"\xa7\x0d\x00@@GCMW_SERVER_DIR@@\x00binary"
        )

        out = self._render(tmp_path, bundle=bundle)

        assert not (out / "bin" / "__pycache__").exists()
        assert "skipped" in self.helper_stdout
        assert "none left" in self.helper_stdout


class TestReadmeCoversTheOperationalContract:
    def _readme(self) -> str:
        return _read(DEPLOY_DIR / "README.md")

    @pytest.mark.parametrize(
        "heading",
        ["## Startup order", "## Health checks", "## Rollback"],
    )
    def test_documents_the_three_operational_sections(self, heading):
        assert heading in self._readme()

    def test_states_that_hardware_acceptance_is_outstanding(self):
        text = self._readme()
        assert "Not yet verified on hardware" in text
        assert "physical microphone" in text

    def test_explains_why_the_display_is_resolved(self):
        text = self._readme()
        assert "mutter-x11-frames" in text
        assert "does not reparent" in text

    def test_explains_why_the_corpus_is_seeded_by_code(self):
        assert "demo_seed.py" in self._readme()


class TestBundleClaimsMatchTheShippedCode:
    """Cross-checks: the unit templates must agree with the code they launch."""

    def test_qa_server_defaults_to_loopback_in_code_too(self):
        server_dir = Path(__file__).resolve().parents[1]
        text = (server_dir / "qa_server.py").read_text(encoding="utf-8")
        assert 'HOST = "127.0.0.1"' in text
        assert "0.0.0.0" not in text

    def test_voice_bridge_contract_url_matches_the_adapter(self):
        from voice_agent_adapter import DEMO_CONTRACT_BASE_URL

        example = _read(CONFIG_DIR / "voice-bridge.env.example")
        assert f"GCMW_VOICE_AGENT_API_BASE={DEMO_CONTRACT_BASE_URL}" in example

    def test_agent_entry_points_at_the_tracked_demo_assembly(self):
        module = _load_agent_entry()
        assert str(module.DEMO_SCRIPT_RELATIVE) == "scripts/demo_showcase.py"
        server_dir = Path(__file__).resolve().parents[1]
        assert (server_dir / module.DEMO_SCRIPT_RELATIVE).is_file()
