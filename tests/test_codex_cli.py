"""Tests for limbic.cerebellum.codex_cli — command shaping and subprocess handling.

Retry/quota behaviour lives in tests/test_cerebellum.py::TestCodexCLIRetry.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

from limbic.cerebellum import codex_cli as cc


# ---------------------------------------------------------------------------
# Command shaping
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_cmd(monkeypatch):
    """Run codex_json/codex_research against a stub _run and return the argv."""
    seen: list[list[str]] = []

    def _run(cmd, timeout):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(cc, "_run", _run)
    return seen


class TestIsolationFlags:
    """Both entry points get --ephemeral/--ignore-user-config by default.

    codex_research is the call that reads untrusted web pages with network
    egress, so omitting them there was the worse of the two defaults. hvaskjer
    was rebinding codex_cli._run at runtime to inject them.
    """

    def test_codex_research_is_isolated_by_default(self, captured_cmd):
        cc.codex_research("mission")
        cmd = captured_cmd[0]
        assert "--ephemeral" in cmd
        assert "--ignore-user-config" in cmd

    def test_codex_json_is_isolated(self, captured_cmd):
        cc.codex_json("prompt")
        cmd = captured_cmd[0]
        assert "--ephemeral" in cmd
        assert "--ignore-user-config" in cmd

    def test_isolated_false_restores_host_profile(self, captured_cmd):
        cc.codex_research("mission", isolated=False)
        cmd = captured_cmd[0]
        assert "--ephemeral" not in cmd
        assert "--ignore-user-config" not in cmd

    def test_isolation_does_not_disturb_agentic_flags(self, captured_cmd):
        """The flags that make the run agentic must survive the insertion."""
        cc.codex_research("mission")
        cmd = captured_cmd[0]
        assert "tools.web_search=true" in cmd
        assert "sandbox_workspace_write.network_access=true" in cmd
        assert cmd[:2] == ["codex", "exec"]
        assert cmd[-1] == "mission"


# ---------------------------------------------------------------------------
# Bounded output
# ---------------------------------------------------------------------------


class TestTail:
    def test_keeps_the_end_not_the_start(self):
        tail = cc._Tail(10)
        tail.add("abcdefghijklmnop")
        assert tail.get().endswith("ghijklmnop")

    def test_reports_how_much_it_dropped(self):
        tail = cc._Tail(10)
        tail.add("x" * 25)
        assert "15 earlier characters discarded" in tail.get()

    def test_under_limit_is_verbatim(self):
        tail = cc._Tail(100)
        tail.add("short")
        assert tail.get() == "short"


# ---------------------------------------------------------------------------
# Subprocess handling (real child processes)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    """Put an executable named `codex` on PATH that runs a given python snippet."""

    def install(body: str) -> None:
        script = tmp_path / "codex"
        script.write_text(
            f"#!{sys.executable}\n" + textwrap.dedent(body), encoding="utf-8"
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(cc, "_DISABLED_UNTIL", 0.0)

    return install


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
class TestRunSubprocess:
    def test_output_is_capped(self, fake_codex, monkeypatch):
        monkeypatch.setattr(cc, "OUTPUT_LIMIT", 1000)
        fake_codex("""
            import sys
            sys.stdout.write("y" * 50000)
        """)
        proc = cc._run(["codex"], timeout=30)
        assert proc.returncode == 0
        # The cap plus the one-line "discarded" preamble.
        assert len(proc.stdout) < 1100
        assert "earlier characters discarded" in proc.stdout

    def test_large_output_does_not_deadlock(self, fake_codex):
        """A PIPE without concurrent draining blocks the child at ~64KB."""
        fake_codex("""
            import sys
            sys.stdout.write("a" * 300000)
            sys.stderr.write("b" * 300000)
        """)
        proc = cc._run(["codex"], timeout=30)
        assert proc.returncode == 0
        assert proc.stdout.endswith("a")
        assert proc.stderr.endswith("b")

    def test_timeout_raises(self, fake_codex):
        fake_codex("""
            import time
            time.sleep(30)
        """)
        with pytest.raises(cc.CodexCLIError, match="timed out"):
            cc._run(["codex"], timeout=1)

    def test_timeout_kills_the_whole_process_tree(self, fake_codex, tmp_path):
        """subprocess.run(timeout=) kills only the direct child; codex spawns helpers.

        The child forks a grandchild that would outlive it and keep writing.
        """
        marker = tmp_path / "grandchild.txt"
        fake_codex(f"""
            import os, sys, time
            if os.fork() == 0:
                for _ in range(200):
                    with open({str(marker)!r}, "a") as fh:
                        fh.write("tick\\n")
                        fh.flush()
                    time.sleep(0.1)
                os._exit(0)
            time.sleep(30)
        """)
        with pytest.raises(cc.CodexCLIError, match="timed out"):
            cc._run(["codex"], timeout=1)
        settled = marker.read_text() if marker.exists() else ""
        time.sleep(0.5)
        after = marker.read_text() if marker.exists() else ""
        assert after == settled, "grandchild survived the timeout and kept writing"

    def test_nonzero_exit_is_returned_not_raised(self, fake_codex):
        """_run reports; _finish is what turns a bad exit into CodexCLIError."""
        fake_codex("""
            import sys
            sys.stderr.write("boom")
            sys.exit(3)
        """)
        proc = cc._run(["codex"], timeout=30)
        assert proc.returncode == 3
        assert "boom" in proc.stderr

    def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        with pytest.raises(cc.CodexCLIError, match="not available"):
            cc._run(["codex"], timeout=5)
