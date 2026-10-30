"""Run: python3 skills/campaign-orchestration/scripts/test_lane_keeper.py"""
import importlib.util, json, os, tempfile, time, unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('lane_keeper', Path(__file__).with_name('lane-keeper.py'))
lk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lk)


def touch(p, t):
    p.touch(); os.utime(p, (t, t))


class RetireTest(unittest.TestCase):
    def setUp(self):
        for var in ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_SESSION_ID'):
            os.environ.pop(var, None)
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / 'repo').mkdir()
        cfg = self.tmp / 'coordinator.json'
        cfg.write_text(json.dumps({'repo': str(self.tmp / 'repo'), 'campaigns': [], 'state': str(self.tmp / 'state')}))
        lk.REG = self.tmp / 'reg'
        lk.REG.mkdir()
        self.K = lk.Keeper(cfg)
        self.t = time.time()

    def register(self, sid, reg_t, transcript_t):
        tr = self.tmp / f'{sid}.jsonl'
        touch(tr, transcript_t)
        f = lk.REG / f'{sid}.json'
        f.write_text(json.dumps({'repo': str(self.K.repo), 'transcript': str(tr)}))
        os.utime(f, (reg_t, reg_t))
        return f

    def retire_at(self, t, sid=None):
        self.K.retire(sid)
        os.utime(self.K.k / 'retired', (t, t))

    def test_retire_ignores_older_heartbeat_and_registry(self):
        touch(self.K.k / 'watch.heartbeat', self.t - 20)
        self.register('old', self.t - 3000, self.t - 10)
        self.assertIsNotNone(self.K.coordinator_live())
        self.retire_at(self.t - 15)
        self.assertIsNone(self.K.coordinator_live())

    def test_retire_with_session_removes_registry(self):
        f = self.register('abc-1', self.t - 3000, self.t)
        self.K.retire('abc-1')
        self.assertFalse(f.exists())
        self.assertIsNone(self.K.coordinator_live())

    def test_newer_signals_count_after_retire(self):
        self.retire_at(self.t - 100)
        touch(self.K.k / 'watch.heartbeat', self.t - 5)
        self.assertEqual(self.K.coordinator_live(), 'watch heartbeat')
        os.unlink(self.K.k / 'watch.heartbeat')
        self.register('new', self.t - 50, self.t - 5)
        self.assertEqual(self.K.coordinator_live(), 'session new active')

    def test_retire_bypasses_min_gap_once(self):
        self.K.dry = True
        self.K.st['events'] = {'k': {'lane': '/x', 'name': 'x', 'kind': 'READY', 'detail': 'd', 't': self.t}}
        self.K.st['coordinator'] = {'last': self.t - 60, 'day': time.strftime('%F'), 'starts': 1}
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 1)   # within min gap
        self.retire_at(self.t - 30)
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 2)
        self.K.st['events']['k2'] = dict(self.K.st['events']['k'])
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 2)   # bypass used up

    def test_start_command_streams_with_session_id(self):
        cmd = lk.Keeper.coordinator_cmd('p', 'sid-1')
        joined = ' '.join(cmd)
        self.assertIn('--output-format stream-json', joined)
        self.assertIn('--verbose', cmd)
        self.assertIn('--session-id sid-1', joined)


if __name__ == '__main__':
    unittest.main()
