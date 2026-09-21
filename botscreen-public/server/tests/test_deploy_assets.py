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
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
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


def _write_stub_assembly(
    directory: Path, *, thread_alive: bool, cleanup_log: Path
) -> Path:
    """Write a fake ``scripts/demo_showcase.py`` for the Agent entry point.

    The entry only needs ``create_app``, a ``start_server`` returning an object
    with a ``.thread``, and ``stop_server``. Nothing here binds a port, touches
    the network or starts Electron — ``thread_alive`` decides, deterministically,
    whether the service thread is still running or has already exited.
    """
    scripts = directory / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    stub = scripts / "demo_showcase.py"
    stub.write_text(
        "class _Thread:\n"
        "    def is_alive(self):\n"
        f"        return {thread_alive!r}\n"
        "\n"
        "\n"
        "class _Handle:\n"
        "    def __init__(self):\n"
        "        self.thread = _Thread()\n"
        "\n"
        "\n"
        "def create_app(*args, **kwargs):\n"
        "    return object()\n"
        "\n"
        "\n"
        "def start_server(host, port, cors_for_dev_frontends):\n"
        "    return _Handle()\n"
        "\n"
        "\n"
        "def stop_server(handle):\n"
        f"    with open({str(cleanup_log)!r}, 'a', encoding='utf-8') as stream:\n"
        "        stream.write('stopped\\n')\n",
        encoding="utf-8",
    )
    return stub


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

    def test_stop_signal_is_a_clean_exit_and_cleanup_still_runs(
        self, tmp_path, monkeypatch
    ):
        """P2 regression (review probe): an operator stop must exit 0.

        The unit is ``Restart=on-failure``, so a signal must NOT be reported as a
        failure — otherwise stopping the demo Agent API would bounce it straight
        back up. Cleanup has to run on this path too.
        """
        module = _load_agent_entry()
        cleanup = tmp_path / "cleanup.log"
        _write_stub_assembly(tmp_path, thread_alive=True, cleanup_log=cleanup)

        handlers: dict[int, object] = {}

        class _SignalRecorder:
            SIGTERM = signal.SIGTERM
            SIGINT = signal.SIGINT

            @staticmethod
            def signal(signum, handler):
                handlers[signum] = handler

        # Replace the module's `signal` handle: calling the real one from a worker
        # thread is illegal, and a real handler would also outlive the test.
        monkeypatch.setattr(module, "signal", _SignalRecorder())

        result: dict[str, int] = {}
        worker = threading.Thread(
            target=lambda: result.__setitem__(
                "code", module.main(["--server-dir", str(tmp_path)])
            ),
            daemon=True,
        )
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while signal.SIGTERM not in handlers and time.monotonic() < deadline:
                time.sleep(0.02)
            assert signal.SIGTERM in handlers, "the entry never installed a handler"
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        finally:
            worker.join(timeout=10)

        assert not worker.is_alive(), "the entry did not return after the stop signal"
        assert result["code"] == 0
        assert cleanup.read_text(encoding="utf-8") == "stopped\n"

    def test_a_service_thread_that_dies_alone_is_reported_as_failure(
        self, tmp_path, capsys
    ):
        """P2 regression (review probe): the thread exiting on its own used to
        return 0 as well.

        With ``Restart=on-failure`` that reads as a deliberate stop, so an
        unexpected 8001 outage would never be restarted. Cleanup still runs.
        """
        module = _load_agent_entry()
        cleanup = tmp_path / "cleanup.log"
        _write_stub_assembly(tmp_path, thread_alive=False, cleanup_log=cleanup)

        assert module.main(["--server-dir", str(tmp_path)]) != 0
        assert "without a stop signal" in capsys.readouterr().err
        assert cleanup.read_text(encoding="utf-8") == "stopped\n"


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
    def _render(
        self,
        tmp_path: Path,
        bundle: Path | None = None,
        deploy_dir: str | None = "/opt/gcmw/deploy/current",
        root: str = "/opt/gcmw",
    ) -> Path:
        out = tmp_path / "rendered"
        arguments = [  # check=False on purpose: the helper's output IS the diagnosis
            "bash",
            str((bundle or DEPLOY_DIR) / "bin" / "render-assets.sh"),
            "--out",
            str(out),
            "--root",
            root,
            "--user",
            "demo-user",
            "--home",
            "/home/demo-user",
        ]
        if deploy_dir is not None:
            arguments += ["--deploy-dir", deploy_dir]
        completed = subprocess.run(
            arguments,
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

    @staticmethod
    def _install(rendered: Path, root: Path) -> None:
        """Simulate the README's install step, so unit targets can be dereferenced.

        Everything the units exec is created as a stand-in: a fake ``botscreen``
        for the release and a fake interpreter under ``<root>/venv``. Nothing
        here launches Electron or binds a port.

        The README is installed straight from the bundle (the renderer does not
        copy it: its placeholder table documents the token NAMES, so substituting
        them would destroy it).
        """
        bundle = root / "deploy" / "current"
        shutil.copytree(rendered / "bin", bundle / "bin")
        shutil.copy2(DEPLOY_DIR / "README.md", bundle / "README.md")

        release = root / "current"
        release.mkdir(parents=True, exist_ok=True)
        stub_app = release / "botscreen"
        stub_app.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub_app.chmod(0o755)

        interpreter = root / "venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True, exist_ok=True)
        interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        interpreter.chmod(0o755)

    @staticmethod
    def _exec_targets_under(root: Path, out: Path) -> list[str]:
        complete = re.compile(r"@@GCMW_[A-Z_]+@@")
        targets: list[str] = []
        for unit_path in sorted((out / "systemd").glob("*.service")):
            for entry in _directive(_read(unit_path), "ExecStart"):
                assert not complete.search(entry), (
                    f"{unit_path.name} keeps an unrendered token: {entry}"
                )
                for token in entry.split():
                    candidate = token.strip('"')
                    if candidate.startswith(f"{root}/"):
                        assert Path(candidate).is_file(), (
                            f"{unit_path.name} execs {candidate}, which the"
                            " documented install step must create"
                        )
                        targets.append(candidate)
        return targets

    def test_default_deploy_dir_targets_the_install_dir_not_the_checkout(
        self, tmp_path
    ):
        """P1 regression (review probe: ``unit_targets_source=True``).

        Omitting ``--deploy-dir`` used to default to the bundle's own source
        directory, so the rendered units exec'd unrendered templates sitting in
        the repository — and the render still exited 0, which is why the existing
        cases (all of which pass --deploy-dir explicitly) never caught it.
        """
        root = tmp_path / "opt" / "gcmw"
        rendered = self._render(tmp_path, deploy_dir=None, root=str(root))
        self._install(rendered, root)

        unit = _read(rendered / "systemd" / "botscreen.service")
        assert _directive(unit, "ExecStart") == [
            f"{root}/deploy/current/bin/start-botscreen-ui"
        ]
        assert str(DEPLOY_DIR) not in unit, (
            "a default render must never exec the checkout"
        )

    def test_every_exec_target_of_a_default_render_is_rendered_and_installed(
        self, tmp_path
    ):
        """P1 regression (review probe: ``target_has_unrendered_tokens=True``).

        Dereference what all three units exec after the documented install: each
        target must exist and must be the RENDERED copy, not the template.
        """
        root = tmp_path / "opt" / "gcmw"
        rendered = self._render(tmp_path, deploy_dir=None, root=str(root))
        self._install(rendered, root)

        targets = self._exec_targets_under(root, rendered)
        bundle_bin = f"{root}/deploy/current/bin"
        assert f"{bundle_bin}/start-botscreen-ui" in targets
        assert f"{bundle_bin}/run-demo-agent-api.py" in targets, (
            "the units must exec the installed bundle's entry points"
        )
        launcher = _read(root / "deploy" / "current" / "bin" / "start-botscreen-ui")
        assert "@@GCMW_" not in launcher
        assert f'exec "{root}/current/botscreen"' in launcher

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
        for name in (
            "start-botscreen-ui",
            "render-assets.sh",
            "backup-assets.sh",
            "rollback-assets.sh",
        ):
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

    def test_renders_exactly_the_files_of_bin(self, tmp_path):
        out = self._render(tmp_path)
        assert sorted(path.name for path in (out / "bin").iterdir()) == [
            "backup-assets.sh",
            "render-assets.sh",
            "rollback-assets.sh",
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


@pytest.mark.skipif(shutil.which("bash") is None, reason="the bundle scripts are bash")
class TestBackupAndRollbackRestoreTheFiles:
    """The install -> backup -> overwrite -> rollback loop.

    File operations only: every directory is an argument to the scripts, which is
    what makes this runnable in a temp dir — no service is touched and nothing
    needs sudo. The scripts inherit the ambient environment, exactly as an
    operator's shell would give it to them.

    Why this exists: the previous procedure timestamped per file, backed up only
    qa-server.service, and then overwrote botscreen.service and the user unit with
    no copy at all. A rollback that references a file nobody saved is not a
    rollback, and prose in a README could not catch that — this can.

    Two contracts are pinned beyond the happy path, because both were real:

    - a snapshot that cannot be *proven* complete changes **nothing** (the old code
      restored some files, kept the rest new, and once even exited 0);
    - a write that fails anyway is reported as a partial restore — naming the
      failing target as possibly modified and counting only the targets whose copy
      AND mode both landed — rather than exiting non-zero in silence.
    """

    STAMP = "20260921-120000"

    @staticmethod
    def _layout(tmp_path: Path) -> dict[str, Path]:
        return {
            "backup_dir": tmp_path / "var" / "backups" / "gcmw",
            "system_unit_dir": tmp_path / "etc" / "systemd" / "system",
            "user_unit_dir": tmp_path
            / "home"
            / "operator"
            / ".config"
            / "systemd"
            / "user",
            "deploy_dir": tmp_path / "opt" / "gcmw" / "deploy" / "current",
        }

    def _script(
        self, layout: dict[str, Path], name: str, *extra: str
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "bash",
                str(BIN_DIR / name),
                "--backup-dir",
                str(layout["backup_dir"]),
                "--system-unit-dir",
                str(layout["system_unit_dir"]),
                "--user-unit-dir",
                str(layout["user_unit_dir"]),
                "--deploy-dir",
                str(layout["deploy_dir"]),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _write(path: Path, content: str, mode: int = 0o644) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(mode)

    def _old_state(self, layout: dict[str, Path]) -> None:
        """Before the install: three files, and no user unit / Agent entry point."""
        self._write(layout["system_unit_dir"] / "qa-server.service", "OLD qa\n", 0o600)
        self._write(layout["system_unit_dir"] / "botscreen.service", "OLD ui\n")
        self._write(
            layout["deploy_dir"] / "bin" / "start-botscreen-ui", "OLD launcher\n", 0o700
        )

    def _full_old_state(self, layout: dict[str, Path]) -> None:
        """All five managed files present, so every manifest row is COPIED.

        The ABSENT path is covered by _old_state; this is the shape that makes a
        snapshot droppable in the middle, which is what the review injected.
        """
        self._old_state(layout)
        self._write(
            layout["user_unit_dir"] / "gcmw-agent-demo.service", "OLD user unit\n"
        )
        self._write(
            layout["deploy_dir"] / "bin" / "run-demo-agent-api.py", "OLD entry\n", 0o755
        )

    def _new_state(self, layout: dict[str, Path]) -> None:
        self._write(layout["system_unit_dir"] / "qa-server.service", "NEW qa\n", 0o600)
        self._write(layout["system_unit_dir"] / "botscreen.service", "NEW ui\n", 0o640)
        self._write(
            layout["user_unit_dir"] / "gcmw-agent-demo.service", "NEW user unit\n"
        )
        self._write(
            layout["deploy_dir"] / "bin" / "start-botscreen-ui", "NEW launcher\n", 0o755
        )
        self._write(
            layout["deploy_dir"] / "bin" / "run-demo-agent-api.py", "NEW entry\n", 0o755
        )

    @staticmethod
    def _managed(layout: dict[str, Path]) -> dict[str, Path]:
        """The five files an install overwrites. Order matches the scripts."""
        return {
            "qa-server.service": layout["system_unit_dir"] / "qa-server.service",
            "botscreen.service": layout["system_unit_dir"] / "botscreen.service",
            "gcmw-agent-demo.service": (
                layout["user_unit_dir"] / "gcmw-agent-demo.service"
            ),
            "start-botscreen-ui": (layout["deploy_dir"] / "bin" / "start-botscreen-ui"),
            "run-demo-agent-api.py": (
                layout["deploy_dir"] / "bin" / "run-demo-agent-api.py"
            ),
        }

    @classmethod
    def _fingerprint(cls, layout: dict[str, Path]) -> dict[str, tuple[str, int] | None]:
        """Content AND mode of every managed file, or None where it is absent."""
        return {
            name: (
                (_read(path), path.stat().st_mode & 0o777) if path.is_file() else None
            )
            for name, path in cls._managed(layout).items()
        }

    @classmethod
    def _manifest_lines(cls, layout: dict[str, Path], stamp: str) -> list[str]:
        return _read(layout["backup_dir"] / stamp / "MANIFEST.tsv").splitlines()

    @classmethod
    def _write_manifest(
        cls, layout: dict[str, Path], stamp: str, lines: list[str]
    ) -> None:
        path = layout["backup_dir"] / stamp / "MANIFEST.tsv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _rollback_elsewhere(
        self, layout: dict[str, Path], elsewhere: dict[str, Path], *extra: str
    ) -> subprocess.CompletedProcess:
        """Roll back the real snapshot while pointing the directories at a
        different root — the operator-supplied directories must win, so this has
        to be refused rather than silently writing to the manifest's paths."""
        return subprocess.run(
            [
                "bash",
                str(BIN_DIR / "rollback-assets.sh"),
                "--backup-dir",
                str(layout["backup_dir"]),
                "--system-unit-dir",
                str(elsewhere["system_unit_dir"]),
                "--user-unit-dir",
                str(elsewhere["user_unit_dir"]),
                "--deploy-dir",
                str(elsewhere["deploy_dir"]),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_backup_then_rollback_restores_content_and_mode(self, tmp_path):
        layout = self._layout(tmp_path)
        self._old_state(layout)

        backup = self._script(layout, "backup-assets.sh", "--stamp", self.STAMP)
        assert backup.returncode == 0, backup.stderr

        # the overwrite really happened ...
        self._new_state(layout)
        launcher = layout["deploy_dir"] / "bin" / "start-botscreen-ui"
        assert _read(launcher) == "NEW launcher\n"

        # ... and the rollback puts the OLD bytes back, at the OLD mode
        rollback = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert rollback.returncode == 0, rollback.stderr

        assert _read(layout["system_unit_dir"] / "qa-server.service") == "OLD qa\n"
        assert _read(layout["system_unit_dir"] / "botscreen.service") == "OLD ui\n"
        assert _read(launcher) == "OLD launcher\n"
        assert launcher.stat().st_mode & 0o777 == 0o700, (
            "the recorded mode must be restored, not the install's 0755"
        )

    def test_entries_absent_before_the_install_are_removed(self, tmp_path):
        layout = self._layout(tmp_path)
        self._old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        user_unit = layout["user_unit_dir"] / "gcmw-agent-demo.service"
        agent_entry = layout["deploy_dir"] / "bin" / "run-demo-agent-api.py"
        assert user_unit.is_file() and agent_entry.is_file()

        rollback = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert rollback.returncode == 0, rollback.stderr

        assert not user_unit.exists(), "restoring a first-time install must remove it"
        assert not agent_entry.exists()
        assert "removed" in rollback.stdout

    def test_dry_run_reports_without_touching_anything(self, tmp_path):
        layout = self._layout(tmp_path)
        self._old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )
        self._new_state(layout)

        plan = self._script(
            layout, "rollback-assets.sh", "--stamp", self.STAMP, "--dry-run"
        )
        assert plan.returncode == 0, plan.stderr
        assert "would:" in plan.stdout

        assert _read(layout["system_unit_dir"] / "qa-server.service") == "NEW qa\n"
        assert (layout["user_unit_dir"] / "gcmw-agent-demo.service").is_file()

    def test_one_stamp_lists_every_managed_file(self, tmp_path):
        """All five files must be recorded — present or not — under one stamp."""
        layout = self._layout(tmp_path)
        self._old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        snapshot = layout["backup_dir"] / self.STAMP
        manifest = _read(snapshot / "MANIFEST.tsv").splitlines()
        assert manifest[0] == "status\tmode\toriginal\tstored"
        assert [line.split("\t")[0] for line in manifest[1:]] == [
            "COPIED",
            "COPIED",
            "ABSENT",
            "COPIED",
            "ABSENT",
        ]

        version = _read(snapshot / "VERSION")
        assert f"stamp={self.STAMP}" in version
        assert "current=" in version
        assert "server_head=" in version

    # ------------------------------------------------------------------
    # A snapshot that cannot be proven complete must change NOTHING.
    #
    # The failure these pin is nastier than a plain crash: the old restore
    # checked and wrote in the same pass, so a snapshot missing its fifth file
    # put the first four back, kept the fifth new, and exited 1 — a tree where
    # some files are old, some new, and nothing records which is which. A
    # truncated manifest was worse still: it restored one file and exited 0.
    #
    # An interrupted backup is not a contrived input. It is what a killed or
    # disk-full run leaves behind, and "restore the newest snapshot" picked it
    # by name ordering.
    # ------------------------------------------------------------------

    def test_a_snapshot_missing_its_last_file_is_refused_before_any_write(
        self, tmp_path
    ):
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        stored = layout["backup_dir"] / self.STAMP / "files" / "5_run-demo-agent-api.py"
        assert stored.is_file(), "precondition: the fifth file was snapshotted"
        stored.unlink()

        refused = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert refused.returncode != 0
        assert "incomplete" in refused.stderr
        assert self._fingerprint(layout) == before, (
            "the old code restored the first four files and left the fifth new"
        )
        assert "Nothing was modified" in refused.stderr

    def test_a_truncated_manifest_is_refused_before_any_write(self, tmp_path):
        layout = self._layout(tmp_path)
        self._old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        lines = self._manifest_lines(layout, self.STAMP)
        assert len(lines) == 6, "precondition: header + five rows"
        self._write_manifest(layout, self.STAMP, lines[:2])  # header + one row

        refused = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert refused.returncode != 0
        assert "5" in refused.stderr and "expected" in refused.stderr
        assert self._fingerprint(layout) == before, (
            "the old code restored the single surviving row and exited 0"
        )

    def test_a_duplicated_target_is_refused_before_any_write(self, tmp_path):
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        lines = self._manifest_lines(layout, self.STAMP)
        launcher = lines[4].split("\t")[2]
        row = lines[5].split("\t")
        # Row 5 now names the launcher as well: one target twice, and the Agent
        # entry point never named at all.
        lines[5] = "\t".join([row[0], row[1], launcher, row[3]])
        self._write_manifest(layout, self.STAMP, lines)

        refused = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert refused.returncode != 0
        assert "exactly once" in refused.stderr
        assert self._fingerprint(layout) == before

    def test_an_extra_target_is_refused_before_any_write(self, tmp_path):
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        outsider = tmp_path / "elsewhere" / "something-else.service"
        lines = self._manifest_lines(layout, self.STAMP)
        # A sixth row, whose stored path DOES exist: only the count rejects it.
        lines.append(f"COPIED\t644\t{outsider}\tfiles/1_qa-server.service")
        self._write_manifest(layout, self.STAMP, lines)

        refused = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert refused.returncode != 0
        assert "6 target" in refused.stderr
        assert not outsider.exists()
        assert self._fingerprint(layout) == before

    def test_a_manifest_cannot_redirect_a_write_outside_the_named_directories(
        self, tmp_path
    ):
        """A complete-looking manifest is still refused if its targets are not
        the five files under the directories this caller named."""
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        outside = tmp_path / "elsewhere" / "bin"
        lines = self._manifest_lines(layout, self.STAMP)
        for index in range(1, 6):
            fields = lines[index].split("\t")
            fields[2] = str(outside / f"file{index}")
            lines[index] = "\t".join(fields)
        self._write_manifest(layout, self.STAMP, lines)

        refused = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert refused.returncode != 0
        assert "do not cover" in refused.stderr
        assert not outside.exists(), "the manifest must not steer cp or rm"
        assert self._fingerprint(layout) == before

    def test_a_manifest_path_that_escapes_the_snapshot_is_refused(self, tmp_path):
        """A stored path is a claim about a file INSIDE the snapshot."""
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        lines = self._manifest_lines(layout, self.STAMP)
        fields = lines[5].split("\t")
        lines[5] = "\t".join([*fields[:3], "../../../../etc/hosts"])
        self._write_manifest(layout, self.STAMP, lines)

        refused = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert refused.returncode != 0
        assert ".." in refused.stderr
        assert self._fingerprint(layout) == before

    def test_directories_that_disagree_with_the_manifest_are_refused(self, tmp_path):
        """The caller's directories decide, not the manifest.

        Without this the operator can hand the script a temp root and watch it
        write to the live install path the manifest recorded instead.
        """
        layout = self._layout(tmp_path)
        self._old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        elsewhere = self._layout(tmp_path / "elsewhere")
        refused = self._rollback_elsewhere(layout, elsewhere, "--stamp", self.STAMP)
        assert refused.returncode != 0
        assert "do not cover" in refused.stderr or "exactly once" in refused.stderr
        assert self._fingerprint(layout) == before, "the live tree must be untouched"
        assert not elsewhere["system_unit_dir"].exists(), (
            "nothing may be created under the other root either"
        )
        assert not elsewhere["deploy_dir"].exists()

    def test_a_leftover_staging_directory_cannot_be_named(self, tmp_path):
        """An unfinished backup is not a snapshot, whatever its name says."""
        layout = self._layout(tmp_path)
        self._old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        partial = layout["backup_dir"] / f".{self.STAMP}.partial.AbC123"
        (partial / "files").mkdir(parents=True)
        (partial / "MANIFEST.tsv").write_text(
            "status\tmode\toriginal\tstored\n", encoding="utf-8"
        )

        refused = self._script(
            layout, "rollback-assets.sh", "--stamp", f".{self.STAMP}.partial.AbC123"
        )
        assert refused.returncode == 2
        assert "staging" in refused.stderr
        assert self._fingerprint(layout) == before

        listed = self._script(layout, "rollback-assets.sh", "--list")
        assert listed.returncode == 0
        assert f"  {self.STAMP}\n" in listed.stdout, "the published stamp is listed"
        assert f".{self.STAMP}.partial" not in listed.stdout, (
            "an unfinished backup must not be offered as a choice"
        )

    def test_rollback_requires_an_explicit_stamp(self, tmp_path):
        """No newest-wins: name ordering is what picks the interrupted snapshot."""
        layout = self._layout(tmp_path)
        self._old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        refused = self._script(layout, "rollback-assets.sh")
        assert refused.returncode == 2
        assert "--stamp is required" in refused.stderr
        assert self._fingerprint(layout) == before

    def test_dry_run_validates_before_it_prints_a_plan(self, tmp_path):
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        before = self._fingerprint(layout)

        plan = self._script(
            layout, "rollback-assets.sh", "--stamp", self.STAMP, "--dry-run"
        )
        assert plan.returncode == 0, plan.stderr
        assert "would:" in plan.stdout
        assert "NOTHING was modified" in plan.stdout
        assert self._fingerprint(layout) == before

        stored = layout["backup_dir"] / self.STAMP / "files" / "5_run-demo-agent-api.py"
        stored.unlink()
        refused = self._script(
            layout, "rollback-assets.sh", "--stamp", self.STAMP, "--dry-run"
        )
        assert refused.returncode != 0, "a dry run must validate the same way"
        assert "incomplete" in refused.stderr
        assert self._fingerprint(layout) == before

    def test_an_interrupted_backup_publishes_nothing(self, tmp_path):
        """A run that dies halfway must leave no stamp to roll back to."""
        layout = self._layout(tmp_path)
        self._old_state(layout)
        # Fail on the SECOND file, so the first is already staged when it dies:
        # the staging directory has content at the moment of the failure.
        unreadable = layout["system_unit_dir"] / "botscreen.service"
        unreadable.chmod(0o000)
        try:
            unreadable.read_bytes()
        except PermissionError:
            pass
        else:
            unreadable.chmod(0o644)
            pytest.skip("this user can read a mode-000 file, so the copy cannot fail")

        died = self._script(layout, "backup-assets.sh", "--stamp", self.STAMP)
        assert died.returncode != 0
        assert not (layout["backup_dir"] / self.STAMP).exists(), (
            "a partial snapshot must never be published under the stamp"
        )
        leftovers = [
            entry.name
            for entry in layout["backup_dir"].iterdir()
            if entry.name.startswith(".")
        ]
        assert leftovers == [], f"staging was not cleaned up: {leftovers}"

        unreadable.chmod(0o644)
        refused = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert refused.returncode != 0, "there is nothing valid to roll back to"

    def test_a_write_that_fails_midway_is_reported_as_a_partial_restore(self, tmp_path):
        """The restore cannot be atomic, so it must not pretend it was.

        A real failure between two writes leaves some files old and some new. What
        is not acceptable is exiting non-zero WITHOUT saying which files already
        changed — from the outside that is indistinguishable from a clean restore,
        and it is the one outcome the validation pass cannot rule out in advance.
        """
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        # Make the fourth write impossible without depending on permissions (which
        # differ under root): deploy/current/bin becomes a FILE, so writing
        # bin/<name> fails with ENOTDIR on any POSIX cp.
        bin_dir = layout["deploy_dir"] / "bin"
        for child in bin_dir.iterdir():
            child.unlink()
        bin_dir.rmdir()
        bin_dir.write_text("not a directory\n", encoding="utf-8")

        partial = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert partial.returncode != 0
        assert "PARTIALLY RESTORED" in partial.stderr, partial.stderr
        assert "3 of 5" in partial.stderr, (
            "the report has to say how far it got, not just that it failed"
        )
        # the three writes that did happen really are the old bytes
        assert _read(layout["system_unit_dir"] / "qa-server.service") == "OLD qa\n"
        assert _read(layout["system_unit_dir"] / "botscreen.service") == "OLD ui\n"
        assert (
            _read(layout["user_unit_dir"] / "gcmw-agent-demo.service")
            == "OLD user unit\n"
        )

    def test_a_target_that_copied_but_could_not_be_chmodded_is_called_out(
        self, tmp_path
    ):
        """A landed copy is not a finished restore.

        An entry counts as done only when BOTH steps ran. If the copy succeeds and
        the mode change is refused, that target's CONTENT has already changed while
        the entry is still a failure — so it has to be reported as possibly
        modified and excluded from the completed count. The first version of this
        report did neither: it listed only completed entries and said nothing about
        the one it stopped on.
        """
        layout = self._layout(tmp_path)
        self._full_old_state(layout)
        assert (
            self._script(layout, "backup-assets.sh", "--stamp", self.STAMP).returncode
            == 0
        )

        self._new_state(layout)
        # A target we can WRITE but do not OWN: /dev/null is 0666 and owned by root
        # everywhere, so `cp -f` into it succeeds and `chmod` is refused. That is the
        # shape of a real "copy landed, mode refused" failure, without inventing a
        # fault the script cannot meet in the field.
        launcher = layout["deploy_dir"] / "bin" / "start-botscreen-ui"
        launcher.unlink()
        launcher.symlink_to("/dev/null")
        # chmod is refused only for a file we are neither root nor the owner of.
        if os.geteuid() == 0 or os.stat("/dev/null").st_uid == os.geteuid():
            pytest.skip("root, or /dev/null is ours: the mode change cannot fail")

        partial = self._script(layout, "rollback-assets.sh", "--stamp", self.STAMP)
        assert partial.returncode != 0
        assert "MAY HAVE BEEN PARTIALLY MODIFIED" in partial.stderr, partial.stderr
        assert str(launcher) in partial.stderr, "the failing target must be named"
        assert "3 of 5" in partial.stderr, (
            "and the count must stay at 3: a copy alone does not make it restored"
        )
        assert str(layout["system_unit_dir"] / "qa-server.service") in partial.stderr


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
