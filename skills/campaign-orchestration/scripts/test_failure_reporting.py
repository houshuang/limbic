"""Real captured 1–2 Nov driver state, plus offline process boundary tests.

No tested function is mocked. Captured commands are inspected, never replayed
against book worktrees; executions below use local, free deterministic commands.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

S = Path(__file__).resolve().parent
F = S / 'fixtures/driver-failures'


class FailureReporting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def call(self, *args, **kwargs):
        return subprocess.run(list(args), stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=15, **kwargs)

    def capture(self, name):
        target = self.root / name
        shutil.copytree(F / name, target)
        return target

    def test_captured_empty_tail_reports_exact_failed_check_and_twenty_lines(self):
        lane = self.capture('final-validation')
        provenance = json.loads((F / 'PROVENANCE.json').read_text())['final-validation']
        for name, sha in provenance['sha256'].items():
            self.assertEqual(hashlib.sha256((lane / name).read_bytes()).hexdigest(), sha)
        self.assertIn('```\n```', (lane / 'DRIVER-FAILED.md').read_text())
        results = json.loads((lane / 'RESULTS.json').read_text())
        # Only relocate log paths; failing argv and statuses remain captured bytes.
        for check in results['checks']:
            check['log'] = str(lane / Path(check['log']).name)
        (lane / 'RESULTS.json').write_text(json.dumps(results))
        before = {p.name: p.read_bytes() for p in lane.iterdir()}
        r = self.call(sys.executable, str(S / 'failure_report.py'), 'inspect', str(lane), '1')
        self.assertEqual(r.returncode, 0, r.stderr)
        failed = next(c for c in results['checks'] if c['returncode'])
        self.assertIn(json.dumps(failed['command']), r.stdout)
        self.assertIn('Exit code: 1', r.stdout)
        self.assertIn((lane / 'driver.pid').read_text().strip(), r.stdout)
        self.assertIn('\n'.join((lane / 'check-2.log').read_text().splitlines()[-20:]), r.stdout)
        self.assertIn('FAIL lost', r.stdout)
        self.assertEqual(before, {p.name: p.read_bytes() for p in lane.iterdir()})

    def test_captured_alif_auth_cannot_exit_success(self):
        lane = self.capture('alif-auth')
        command = ['/bin/cat', str(lane / 'auth.log')]
        r = self.call(sys.executable, str(S / 'deliverables.py'), 'run', str(lane), '--', *command)
        self.assertEqual(r.returncode, 77, r.stderr)
        report = (lane / 'DRIVER-FAILED.md').read_text()
        self.assertIn(json.dumps(command), report)
        self.assertIn('AUTHENTICATION-FAILED', report)
        self.assertIn('token_revoked', report)
        self.assertEqual((lane / 'SESSION-EXIT').read_text().strip(), '77')
        self.assertEqual((lane / 'auth.log').read_bytes(), (F / 'alif-auth/auth.log').read_bytes())
        self.assertEqual(len(list((lane / 'driver-attempts').glob('*.log'))), 1)

    def test_actual_drive_loop_captures_redirected_stderr_and_preserves_original(self):
        lane = self.root / 'lane with spaces'; lane.mkdir()
        # Run the real installed sleepsafe, never a substitute for the function under test.
        safe = Path.home() / '.agents/bin/sleepsafe.py'
        self.assertTrue(safe.is_file())
        old = b'original failure evidence\n'
        (lane / 'DRIVER-FAILED.md').write_bytes(old)
        argv = [sys.executable, '-c', "import sys; print('specific prerequisite failed', file=sys.stderr); sys.exit(19)"]
        command = ['/bin/bash', str(S / 'drive-loop.sh'), '--once', str(lane), 'failure-report-test', '30', '--', *argv]
        # Real launch.log redirection recreates the original missing-driver.log shape.
        with (lane / 'launch.log').open('w') as log:
            r = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, timeout=15, env={**os.environ, 'SLEEPSAFE': str(safe)})
        self.assertEqual(r.returncode, 19)
        self.assertEqual((lane / 'DRIVER-EXIT').read_text().strip(), '19')
        self.assertGreater(int((lane / 'driver.pid').read_text()), 0)
        self.assertEqual(json.loads((lane / 'driver.cmd').read_text())['args'][-len(argv):], argv)
        report = (lane / 'DRIVER-FAILED.md').read_text()
        self.assertIn(json.dumps(argv), report)
        self.assertIn('specific prerequisite failed', report)
        self.assertIn('Exit code: 19', report)
        self.assertIn(old, [p.read_bytes() for p in (lane / 'failure-history').iterdir()])

    def restricted_env(self, lane):
        binary = self.root / 'bin'; binary.mkdir(exist_ok=True)
        # Enough real executables to reach launch preflight, deliberately no codex.
        for name in ('python3', 'dirname', 'basename', 'stat', 'date'):
            target = binary / name
            if not target.exists(): target.symlink_to(shutil.which(name))
        return {**os.environ, 'PATH': str(binary), 'CAMPAIGN_DIR': str(lane.parent),
                'REPO': str(self.root), 'ALLOW_UNWATCHED': '1'}

    def test_launch_and_resume_missing_codex_consume_no_state(self):
        lane = self.capture('alif-auth')
        for name in ('HANDOFF.md', 'DONE.md', 'DRIVER-FAILED.md', 'SESSION-EXIT'):
            (lane / name).write_text('original ' + name)
        message = self.root / 'message'; message.write_text('continue')
        env = self.restricted_env(lane)
        for script, args in [('launch.sh', [lane.name]), ('resume.sh', [str(lane.parent), lane.name, str(message)])]:
            before = {p.name: p.read_bytes() for p in lane.iterdir()}
            r = self.call('/bin/bash', str(S / script), *args, env=env)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn('codex is not executable on PATH', r.stderr)
            self.assertNotIn('launched', r.stdout); self.assertNotIn('resumed', r.stdout)
            self.assertEqual(before, {p.name: p.read_bytes() for p in lane.iterdir()})
            self.assertFalse((lane / 'session-at').exists())

    def test_missing_node_refuses_absolute_npm_codex_before_launch_state(self):
        lane = self.capture('alif-auth')
        env = self.restricted_env(lane)
        codex = self.root / 'bin/codex'
        codex.write_text('#!/usr/bin/env node\n'); codex.chmod(0o755)
        r = self.call('/bin/bash', str(S / 'launch.sh'), lane.name, env=env)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('node is not executable on PATH', r.stderr)
        self.assertFalse((lane / 'session-at').exists())
        self.assertFalse((lane / 'PROMPT.txt').exists())

    def test_alif_canonical_and_lexical_paths_export_documented_home(self):
        row = json.loads((F / 'alif-auth/driver.cmd').read_text())
        for lane in [row['args'][0], json.loads((F / 'PROVENANCE.json').read_text())['alif-auth']['source']]:
            r = self.call(sys.executable, str(S / 'launch_environment.py'), lane,
                          env={**os.environ, 'CODEX_HOME': '/root/.codex'})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('export CODEX_HOME=/opt/alif/.codex', r.stdout)

    def test_session_retains_stdout_semantics_and_detects_current_auth_only(self):
        lane = self.capture('alif-auth')
        log = lane / 'codex.log'
        log.write_bytes((lane / 'auth.log').read_bytes())
        command = [sys.executable, str(S / 'deliverables.py'), 'session', str(lane), '--', '/bin/echo', 'healthy']
        with log.open('a') as stream:
            r = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=5)
        self.assertEqual(r.returncode, 0)  # historical authentication failure is not current
        command[-2:] = ['/bin/cat', str(lane / 'auth.log')]
        with log.open('a') as stream:
            r = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=5)
        self.assertEqual(r.returncode, 77)
        self.assertIn('AUTHENTICATION-FAILED', (lane / 'DRIVER-FAILED.md').read_text())

    def test_native_77_is_reported_without_inventing_authentication_failure(self):
        lane = self.root / 'native77'; lane.mkdir()
        r = self.call(sys.executable, str(S / 'deliverables.py'), 'run', str(lane), '--',
                      '/bin/sh', '-c', 'echo native permission failure; exit 77')
        self.assertEqual(r.returncode, 77)
        report = (lane / 'DRIVER-FAILED.md').read_text()
        self.assertIn('Process exit code: 77', report)
        self.assertNotIn('AUTHENTICATION-FAILED', report)

    def test_real_keeper_refuses_missing_cli_before_fresh_accounting(self):
        import test_lane_keeper
        keeper_module = test_lane_keeper.lk
        lane = self.capture('final-validation')
        row = json.loads((lane / 'driver.cmd').read_text())
        row['cwd'] = str(lane); row['args'][1] = str(lane)
        (lane / 'driver.cmd').write_text(json.dumps(row))
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps(dict(repo=str(self.root), state=str(self.root / 'keeper-state'),
                                       budget=dict(enabled=False))))
        keeper = keeper_module.Keeper(cfg)
        old_path = keeper_module.PATH
        try:
            keeper_module.PATH = self.restricted_env(lane)['PATH']
            keeper.launch(lane.parent, lane, 'captured failed-launch regression')
        finally:
            keeper_module.PATH = old_path
        self.assertFalse((lane / 'fresh').exists())
        self.assertFalse((lane / 'fresh-at').exists())
        self.assertFalse((lane / 'session-at').exists())
        self.assertIn('codex is not executable on PATH', (keeper.k / 'EVENTS').read_text())
        self.assertEqual([e['kind'] for e in keeper.st['events'].values()], ['LAUNCH-REFUSED'])
        self.assertFalse((lane.parent / 'autoresume.log').exists())

    def test_normal_incident_discussion_is_not_auth_failure(self):
        lane = self.root / 'lane'; lane.mkdir()
        r = self.call(sys.executable, str(S / 'deliverables.py'), 'run', str(lane), '--',
                      sys.executable, '-c', "print('The old incident was token_revoked; fixed in shared tooling.')")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((lane / 'DRIVER-FAILED.md').exists())


if __name__ == '__main__':
    unittest.main()
