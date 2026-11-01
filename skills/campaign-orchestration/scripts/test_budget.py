"""Run: python3 skills/campaign-orchestration/scripts/test_budget.py"""
import importlib.util, json, os, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import budget  # noqa: E402
import report  # noqa: E402

spec = importlib.util.spec_from_file_location('lane_keeper', HERE / 'lane-keeper.py')
lk = importlib.util.module_from_spec(spec)
sys.modules['lane_keeper'] = lk
spec.loader.exec_module(lk)

WEEK = 7 * 24 * 3600


def snap(used, resets_in, **kw):
    t = time.time()
    return {'engine': 'codex', 'used': used, 'window_min': 10080, 'resets_at': t + resets_in, 't': t, **kw}


class PacingTest(unittest.TestCase):
    def setUp(self):
        self.P = budget.load('/nonexistent')

    def test_line_rises_with_elapsed_time_and_never_exceeds_cap(self):
        t = time.time()
        early = budget.pacing_line(snap(0, WEEK - 3600), 40, 5, 'normal', t)
        mid = budget.pacing_line(snap(0, WEEK / 2), 40, 5, 'normal', t)
        late = budget.pacing_line(snap(0, 60), 40, 5, 'normal', t)
        self.assertLess(early, 6)
        self.assertAlmostEqual(mid, 35, delta=0.5)
        self.assertEqual(late, 60)

    def test_overnight_burn_is_held(self):
        # 1 Nov 2026: 26% of the Codex week used nine hours into it
        ok, reason, info = budget.gate('codex', self.P, snap=snap(26, WEEK - 9 * 3600))
        self.assertFalse(ok)
        self.assertIn('pacing line', reason)

    def test_modes(self):
        s = snap(30, WEEK / 2)
        self.assertTrue(budget.gate('codex', self.P, snap=s)[0])
        self.P['mode']['codex'] = 'low'
        self.assertFalse(budget.gate('codex', self.P, snap=s)[0])
        self.P['mode']['codex'] = 'off'
        self.assertFalse(budget.gate('codex', self.P, snap=snap(0, WEEK / 2))[0])
        self.P['mode']['codex'] = 'burn'
        self.assertTrue(budget.gate('codex', self.P, snap=snap(55, WEEK - 3600))[0])
        self.assertFalse(budget.gate('codex', self.P, snap=snap(61, WEEK - 3600))[0])   # burn never eats the reserve

    def test_burn_expires_at_its_reset(self):
        t = time.time()
        self.P['mode']['codex'] = 'burn'
        self.P['mode_until'] = {'codex': t - 1}
        self.assertEqual(budget.mode_of(self.P, 'codex', t), 'normal')

    def test_after_reset_usage_counts_as_zero(self):
        ok, reason, info = budget.gate('codex', self.P, snap=snap(100, -60, blocked=True))
        self.assertTrue(ok, reason)
        self.assertEqual(info['used'], 0)

    def test_limit_reached_and_five_hour_window_hold(self):
        self.assertFalse(budget.gate('codex', self.P, snap=snap(10, WEEK / 2, blocked=True))[0])
        s = snap(10, WEEK / 2, five_used=97, five_resets_at=time.time() + 3600)
        ok, reason, _ = budget.gate('claude', self.P, snap=s)
        self.assertFalse(ok)
        self.assertIn('five-hour', reason)

    def test_no_data_allows_and_says_so(self):
        with mock.patch.object(budget, 'snapshot', return_value=None):
            ok, reason, _ = budget.gate('claude', self.P)
        self.assertTrue(ok)
        self.assertIn('no usage data', reason)


class SnapshotTest(unittest.TestCase):
    def test_codex_snapshot_reads_newest_rate_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'sessions/2026/11/01'; d.mkdir(parents=True)
            rl = {'primary': {'used_percent': 26.0, 'window_minutes': 10080, 'resets_at': 1794089124}}
            (d / 'a.jsonl').write_text(json.dumps({'timestamp': '2026-11-01T07:00:00Z', 'type': 'event_msg',
                                                   'payload': {'type': 'token_count', 'rate_limits': rl}}) + '\n')
            S = budget.codex_snapshot(tmp)
        self.assertEqual(S['used'], 26.0)
        self.assertEqual(S['resets_at'], 1794089124)

    def test_claude_snapshot_skips_codex_logs_sharing_the_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'coordinator-1.log'
            ev = {'type': 'rate_limit_event', 'rate_limit_info': {'unifiedWindows': {
                'five_hour': {'utilization': 0.52, 'resetsAt': 1793486400, 'status': 'allowed'},
                'seven_day': {'utilization': 1, 'resetsAt': 1793502000, 'status': 'rejected'}}}}
            old.write_text(json.dumps(ev) + '\n')
            newer = Path(tmp) / 'coordinator-2.log'
            newer.write_text('{"type":"item.completed"}\n')
            os.utime(old, (1, 1))
            S = budget.claude_snapshot([str(Path(tmp) / 'coordinator-*.log')])
        self.assertEqual(S['used'], 100)
        self.assertTrue(S['blocked'])


class CliTest(unittest.TestCase):
    def test_allow_stop_reserve_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, 'AGENTS_BUDGET': f'{tmp}/b.json', 'CODEX_HOME': tmp}
            run = lambda *a: subprocess.run([sys.executable, str(HERE / 'budget.py'), *a], env=env, capture_output=True, text=True)
            self.assertEqual(run('allow', '2h').returncode, 0)
            P = json.loads(Path(f'{tmp}/b.json').read_text())
            self.assertAlmostEqual(P['autonomy_until'], time.time() + 7200, delta=30)
            run('reserve', 'codex', '55'); run('mode', 'claude', 'off'); run('stop')
            P = json.loads(Path(f'{tmp}/b.json').read_text())
            self.assertEqual((P['reserve']['codex'], P['mode']['claude'], P['autonomy_until']), (55, 'off', 0))
            self.assertEqual(run('mode', 'claude', 'everything').returncode, 2)

    def test_hold_files_follow_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget.write_holds({'codex': (False, 'over line'), 'claude': (True, 'ok')}, tmp)
            self.assertEqual((Path(tmp) / 'HOLD-codex').read_text().strip(), 'over line')
            self.assertFalse((Path(tmp) / 'HOLD-claude').exists())
            budget.write_holds({'codex': (True, 'ok')}, tmp)
            self.assertFalse((Path(tmp) / 'HOLD-codex').exists())


class KeeperGateTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        (self.tmp / 'repo').mkdir()
        cfg = self.tmp / 'c.json'
        cfg.write_text(json.dumps({'repo': str(self.tmp / 'repo'), 'campaigns': [], 'state': str(self.tmp / 'state'),
                                   'budget': {'enabled': True}, 'coordinator': {'engine': 'codex'}}))
        for name, val in (('POLICY', self.tmp / 'budget.json'), ('HOLD_DIR', self.tmp / 'holds')):
            p = mock.patch.object(budget, name, val); p.start(); self.addCleanup(p.stop)
        self.K = lk.Keeper(cfg); self.K.dry = True
        self.K.st['events'] = {'r': {'lane': '/r', 'name': 'r', 'kind': 'READY', 'detail': 'd', 't': time.time()}}
        self.K.st['coordinator'] = {'last': 0, 'day': time.strftime('%F'), 'starts': 0}

    def gates(self, codex_used, allow_hours):
        budget.save({**budget.load(), 'autonomy_until': time.time() + allow_hours * 3600})
        s = snap(codex_used, WEEK / 2)
        with mock.patch.object(budget, 'snapshot', side_effect=lambda e, P: s if e == 'codex' else None):
            self.K.update_budget()

    def test_coordinator_needs_autonomy_and_budget(self):
        self.gates(10, -1)   # autonomy expired
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 0)
        self.assertIn('AUTONOMY-EXPIRED', [e['kind'] for e in self.K.st['events'].values()])
        self.gates(50, 4)    # over the line
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 0)
        self.assertIn('BUDGET-HOLD', [e['kind'] for e in self.K.st['events'].values()])
        self.gates(10, 4)
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 1)

    def test_budget_events_never_start_a_coordinator(self):
        self.K.st['events'] = {}
        self.gates(50, -1)
        self.assertTrue(self.K.st['events'])
        self.K.st['coordinator']['last'] = 0
        self.gates(10, 4)
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 0)

    def test_resume_needs_budget_but_not_autonomy_and_launch_needs_both(self):
        d = self.tmp / 'lane'; d.mkdir()
        self.gates(10, -1)
        self.assertTrue(self.K.may_spend('codex', d))
        self.assertFalse(self.K.may_spend('codex', d, new=True))
        self.gates(50, 4)
        self.assertFalse(self.K.may_spend('codex', d))
        self.assertIn('pacing line', self.K.lane(d)['deferred'])

    def test_disabled_budget_allows_everything(self):
        self.K.cfg['budget'] = {}
        self.assertTrue(self.K.may_spend('codex', None, new=True))

    def test_holds_source_obeys_mirrored_files(self):
        hd = self.tmp / 'mirrored'; hd.mkdir()
        self.K.cfg['budget'] = {'enabled': True, 'source': 'holds', 'hold_dir': str(hd)}
        (hd / 'HOLD-codex').write_text('codex 26% used, pacing line 8%')
        (hd / 'HOLD-autonomy').write_text('autonomy expired')
        self.K.update_budget()
        self.assertFalse(self.K.may_spend('codex'))
        self.assertTrue(self.K.may_spend('claude'))
        self.assertFalse(self.K.may_spend('claude', new=True))
        (hd / 'HOLD-autonomy').unlink(); (hd / 'HOLD-codex').unlink()
        self.K.update_budget()
        self.assertTrue(self.K.may_spend('codex', new=True))

    def test_mirror_pushes_on_change_only(self):
        self.K.dry = False
        self.K.cfg['budget'] = {'enabled': True, 'mirror': ['hostx'], 'hold_dir': str(self.tmp / 'h')}
        with mock.patch.object(lk.subprocess, 'run') as run:
            run.return_value.returncode = 0
            self.gates(10, 4); self.gates(10, 4)
            self.assertEqual(run.call_count, 1)
            self.assertIn('rm -f ~/.agents/budget/HOLD-codex', run.call_args.args[0][-1])
            self.gates(50, 4)
            self.assertEqual(run.call_count, 2)
            self.assertIn('HOLD-codex', run.call_args.args[0][-1])
            self.assertIn('printf', run.call_args.args[0][-1])

    def test_lane_engine(self):
        d = self.tmp / 'l2'; d.mkdir()
        self.assertEqual(lk.Keeper.lane_engine(d), 'codex')
        (d / 'engine').write_text('none\n')
        self.assertEqual(lk.Keeper.lane_engine(d), 'none')


class DriveLoopHoldTest(unittest.TestCase):
    def test_repeating_loop_waits_on_hold_and_once_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); holds = tmp / 'holds'; holds.mkdir()
            (holds / 'HOLD-codex').write_text('over line\n')
            fake = tmp / 'sleepsafe.py'
            fake.write_text('#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n'); fake.chmod(0o755)
            env = {**os.environ, 'SLEEPSAFE': str(fake), 'AGENTS_BUDGET_DIR': str(holds), 'BUDGET_POLL': '1'}
            for once, expect_ran in ((False, False), (True, True)):
                s = tmp / f'lane{int(once)}'; s.mkdir()
                args = ['bash', str(HERE / 'drive-loop.sh'), *(['--once'] if once else []), str(s), 'job', '60', '--',
                        'sh', '-c', f'touch {s}/ran']
                p = subprocess.Popen(args, env=env, stdout=open(s / 'driver.log', 'w'), stderr=subprocess.STDOUT)
                deadline = time.time() + 4
                while time.time() < deadline and p.poll() is None and not (s / 'ran').exists():
                    time.sleep(.1)
                self.assertEqual((s / 'ran').exists(), expect_ran, (s / 'driver.log').read_text())
                if not once:
                    self.assertIn('budget hold', (s / 'driver.log').read_text())
                    (holds / 'HOLD-codex').unlink()
                    p.wait(timeout=10)
                    self.assertTrue((s / 'ran').exists())
                    (holds / 'HOLD-codex').write_text('over line\n')
                p.wait(timeout=10)


class ReportTest(unittest.TestCase):
    def test_status_written_and_digest_once_per_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'repo').mkdir()
            cfg = tmp / 'c.json'
            cfg.write_text(json.dumps({'repo': str(tmp / 'repo'), 'campaigns': [], 'state': str(tmp / 'state'),
                                       'digest': {'times': ['00:00']}}))
            K = lk.Keeper(cfg)
            K.st['events'] = {'r': {'lane': '/r', 'name': 'r', 'kind': 'READY', 'detail': 'checkpoint', 't': 0}}
            self.assertEqual(report.due_digest(K), '00:00')
            report.tick(K)
            md = (K.k / 'STATUS.md').read_text()
            self.assertIn('READY r: checkpoint', md)
            self.assertIn('<title>Keeper status</title>', (K.k / 'STATUS.html').read_text())
            self.assertIsNone(report.due_digest(K))


if __name__ == '__main__':
    unittest.main()
