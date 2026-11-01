"""Run: python3 skills/campaign-orchestration/scripts/test_dashboard.py (also under /usr/bin/python3 3.9)."""
import importlib.util, json, os, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dashboard as db

spec = importlib.util.spec_from_file_location('lane_keeper', HERE / 'lane-keeper.py')
lk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lk)

QUEUE = """# Work queue

Intro line that is not an item.

## P0 · Now

- [x] **Done first** → lane a (abc123).
- [ ] **Open one** → next.
  continuation of open one
  second continuation
- [~] **Running two** → lane b
- [x] **Done middle** (def456)
- [!] **Waiting three**: waits on b

- [ ] **Open four**
## P1 · Later

- [ ] **Later one**
- [ ] **Later two**
  with detail
trailing paragraph
"""


def ids(text, section):
    return [b['id'] for b in db.queue_blocks(text) if b['kind'] == 'item' and b['section'] == section and b['mark'] != 'x']


class QueueTest(unittest.TestCase):
    def test_blocks_round_trip(self):
        self.assertEqual(QUEUE, ''.join(l for b in db.queue_blocks(QUEUE) for l in b['lines']))
        no_newline = QUEUE.rstrip('\n')
        self.assertEqual(no_newline, ''.join(l for b in db.queue_blocks(no_newline) for l in b['lines']))

    def test_every_move_preserves_every_line(self):
        for section in ('P0 · Now', 'P1 · Later'):
            for item in ids(QUEUE, section):
                for d in ('up', 'down', 'top'):
                    out = db.queue_move(QUEUE, item, d)
                    self.assertEqual(sorted(QUEUE.splitlines(keepends=True)), sorted(out.splitlines(keepends=True)))
                    self.assertEqual(len(QUEUE), len(out))

    def test_move_down_then_up_is_identity(self):
        item = ids(QUEUE, 'P0 · Now')[0]
        down = db.queue_move(QUEUE, item, 'down')
        self.assertNotEqual(QUEUE, down)
        self.assertEqual(QUEUE, db.queue_move(down, item, 'up'))

    def test_continuations_travel_and_done_items_stay(self):
        open_one = ids(QUEUE, 'P0 · Now')[0]
        out = db.queue_move(QUEUE, open_one, 'down')
        lines = out.splitlines()
        i = lines.index('- [ ] **Open one** → next.')
        self.assertEqual(lines[i + 1:i + 3], ['  continuation of open one', '  second continuation'])
        self.assertLess(lines.index('- [~] **Running two** → lane b'), i)
        for done in ('- [x] **Done first** → lane a (abc123).', '- [x] **Done middle** (def456)'):
            self.assertEqual(QUEUE.splitlines().index(done), lines.index(done))

    def test_top_and_section_boundary(self):
        last = ids(QUEUE, 'P0 · Now')[-1]
        out = db.queue_move(QUEUE, last, 'top')
        self.assertEqual(ids(out, 'P0 · Now')[0], last)
        self.assertEqual(ids(out, 'P1 · Later'), ids(QUEUE, 'P1 · Later'))
        first_later = ids(QUEUE, 'P1 · Later')[0]
        self.assertEqual(QUEUE, db.queue_move(QUEUE, first_later, 'up'))   # never crosses into P0

    def test_refusals(self):
        with self.assertRaises(KeyError):
            db.queue_move(QUEUE, 'nope-1', 'up')
        done = next(b['id'] for b in db.queue_blocks(QUEUE) if b.get('mark') == 'x')
        with self.assertRaises(ValueError):
            db.queue_move(QUEUE, done, 'up')
        with self.assertRaises(ValueError):
            db.queue_move(QUEUE, ids(QUEUE, 'P0 · Now')[0], 'sideways')

    def test_view(self):
        v = db.queue_view(QUEUE)
        self.assertEqual(['P0 · Now', 'P1 · Later'], [s['title'] for s in v])
        self.assertEqual(['done', 'open', 'running', 'done', 'waiting', 'open'], [i['state'] for i in v[0]['items']])
        self.assertEqual('Open one', v[0]['items'][1]['title'])


class Env(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        self.tmp = Path(t.name)
        (self.tmp / 'repo').mkdir()
        self.state = self.tmp / 'state'
        (self.state / 'keeper').mkdir(parents=True)
        self.camp = self.tmp / 'camp'
        self.camp.mkdir()
        self.queue = self.state / 'QUEUE.md'
        self.queue.write_text(QUEUE)
        self.cfg = {'repo': str(self.tmp / 'repo'), 'state': str(self.state), 'campaigns': [str(self.camp)],
                    'capacity': {'queue': str(self.queue)},
                    'dashboard': {'actions': {'queue-move': {'builtin': 'queue-move'}, 'owner-priority': {'builtin': 'owner-priority'},
                                              'lane-pause': {'builtin': 'lane-pause'}, 'lane-resume': {'builtin': 'lane-resume'},
                                              'echo': {'argv': ['echo', 'hello']}}}}
        self.cfg_path = self.tmp / 'coordinator.json'
        self.cfg_path.write_text(json.dumps(self.cfg))


class PriorityTest(Env):
    def test_set_clear_and_refuse(self):
        p = db.priorities_path(self.cfg)
        db.set_priority(p, 'FB-12', 'P0')
        db.set_priority(p, 'FB-7', 'P2')
        P = json.loads(p.read_text())
        self.assertEqual({'FB-12': 'P0', 'FB-7': 'P2'}, {k: v['priority'] for k, v in P['items'].items()})
        db.set_priority(p, 'FB-12', '')
        self.assertEqual(['FB-7'], list(json.loads(p.read_text())['items']))
        for item, prio in (('FB-1', 'P9'), ('../x', 'P1'), ('', 'P1'), ('a b', 'P1')):
            with self.assertRaises(ValueError):
                db.set_priority(p, item, prio)
        self.assertEqual(['FB-7'], list(json.loads(p.read_text())['items']))
        self.assertEqual([], [f for f in os.listdir(self.state) if f.endswith('.tmp')])

    def test_corrupt_file_starts_empty(self):
        p = db.priorities_path(self.cfg)
        p.write_text('{not json')
        db.set_priority(p, 'FB-3', 'P3')
        self.assertEqual({'FB-3'}, set(json.loads(p.read_text())['items']))


class RequestTest(unittest.TestCase):
    hosts = {'127.0.0.1:4170', 'localhost:4170'}
    origins = {'http://127.0.0.1:4170', 'http://localhost:4170'}
    good = {'Host': '127.0.0.1:4170', 'Origin': 'http://127.0.0.1:4170', 'X-Dashboard-Token': 'tok',
            'Content-Type': 'application/json'}

    def check(self, **change):
        h = {**self.good, **change}
        h = {k: v for k, v in h.items() if v is not None}
        db.check_request(h, self.hosts, self.origins, 'tok')

    def test_good_request_passes(self):
        self.check()
        self.check(Host='localhost:4170', Origin='http://localhost:4170', **{'Sec-Fetch-Site': 'same-origin'})

    def test_token_refusals(self):
        for tok in (None, '', 'tok2', 'to'):
            with self.assertRaises(db.Refused) as e:
                self.check(**{'X-Dashboard-Token': tok})
            self.assertEqual(403, e.exception.status)
            self.assertTrue(e.exception.reload)

    def test_origin_host_and_type_refusals(self):
        for change, status in (({'Origin': None}, 403), ({'Origin': 'http://evil.example'}, 403),
                               ({'Origin': 'http://127.0.0.1:4171'}, 403), ({'Host': 'evil.example:4170'}, 421),
                               ({'Sec-Fetch-Site': 'cross-site'}, 403), ({'Content-Type': 'text/plain'}, 415)):
            with self.assertRaises(db.Refused) as e:
                self.check(**change)
            self.assertEqual(status, e.exception.status, change)


class ActionTest(Env):
    def actions(self):
        C = db.Collector(self.cfg_path)
        C.snap = {'t': time.time(), 'lanes': db.scan(self.cfg['campaigns'], 'Mac')}
        return db.Actions(C)

    def test_whitelist_refusal(self):
        A = self.actions()
        for name in ('rm', 'autonomy-renew', 'preview-start', '', 'queue-move '):
            with self.assertRaises(db.Refused) as e:
                A.do(name, {})
            self.assertEqual(403, e.exception.status)
        self.assertEqual('hello', A.do('echo', {'argv': ['rm', '-rf', '/']}))   # args never reach argv

    def test_queue_move_checks_revision(self):
        A = self.actions()
        item = ids(QUEUE, 'P0 · Now')[0]
        with self.assertRaises(db.Refused) as e:
            A.do('queue-move', {'item': item, 'dir': 'down', 'rev': 'stale'})
        self.assertEqual(409, e.exception.status)
        self.assertEqual(QUEUE, self.queue.read_text())
        A.do('queue-move', {'item': item, 'dir': 'down', 'rev': db.queue_rev(QUEUE)})
        moved = self.queue.read_text()
        A.do('queue-move', {'item': item, 'dir': 'up', 'rev': db.queue_rev(moved)})
        self.assertEqual(QUEUE, self.queue.read_text())

    def test_priority_action(self):
        A = self.actions()
        A.do('owner-priority', {'id': 'FB-9', 'priority': 'P1'})
        self.assertEqual('P1', db.load_priorities(db.priorities_path(self.cfg))['items']['FB-9']['priority'])
        with self.assertRaises(db.Refused):
            A.do('owner-priority', {'id': 'FB-9', 'priority': 'urgent'})

    def test_lane_must_be_scanned(self):
        A = self.actions()
        with self.assertRaises(db.Refused) as e:
            A.do('lane-pause', {'lane': '../../etc'})
        self.assertEqual(404, e.exception.status)


class LaneTest(Env):
    def lane(self, name, args=('st', 'n', '60', '--', 'true')):
        d = self.camp / name
        d.mkdir()
        (d / 'driver.pid').write_text('999999\n')
        (d / 'driver.cmd').write_text(json.dumps({'cwd': str(self.tmp), 'args': [*args[:1], str(d), *args[1:]]}))
        return d

    def test_scan_states(self):
        d = self.lane('a')
        (d / 'driver.log').write_text('one\n\x1b[32mtwo\x1b[0m\n\n')
        (d / 'engine').write_text('codex\n')
        r = db.scan([str(self.camp)], 'Mac')[0]
        self.assertEqual(('idle', 'two', 'codex', False), (r['state'], r['progress'], r['engine'], r['driver']))
        (d / 'READY-1.md').write_text('x')
        r = db.scan([str(self.camp)], 'Mac')[0]
        self.assertEqual(('READY', 'READY-1.md'), (r['state'], r['ready']))
        later = time.time() + 5
        (d / 'HANDLED').write_text('integrated abc\n'); os.utime(d / 'HANDLED', (later, later))
        r = db.scan([str(self.camp)], 'Mac')[0]
        self.assertEqual(('', 'integrated abc'), (r['ready'], r['handled']))

    def test_pause_and_resume(self):
        d = self.lane('b')
        K = lk.Keeper(self.cfg_path)
        K.pause(K.lane_dir(d))
        self.assertTrue((d / 'STOP').exists() and (d / 'PAUSED').exists())
        self.assertEqual('paused', db.scan([str(self.camp)], 'Mac')[0]['state'])
        with mock.patch.object(K, 'run') as run:
            K.resume_driver(d)
        self.assertFalse((d / 'STOP').exists() or (d / 'PAUSED').exists())
        argv = run.call_args[0][0]
        self.assertEqual(['bash', str(lk.SCRIPTS / 'drive-loop.sh'), 'st', str(d), 'n', '60', '--', 'true'], argv)

    def test_resume_leaves_a_coordinator_stop(self):
        d = self.lane('c')
        (d / 'STOP').write_text('retired by coordinator\n')
        K = lk.Keeper(self.cfg_path)
        with self.assertRaises(ValueError):
            K.resume_driver(d)
        with self.assertRaises(ValueError):
            K.pause(d)
        self.assertEqual('retired by coordinator\n', (d / 'STOP').read_text())

    def test_once_jobs_and_foreign_dirs_refused(self):
        d = self.lane('d', args=('--once', 'n', '60', '--', 'true'))
        K = lk.Keeper(self.cfg_path)
        with self.assertRaises(ValueError):
            K.pause(d)
        with self.assertRaises(ValueError):
            K.lane_dir(self.tmp)

    def test_prompt_mentions_priorities_only_when_present(self):
        K = lk.Keeper(self.cfg_path)
        self.assertEqual('', K.priorities_line())
        db.set_priority(db.priorities_path(self.cfg), 'FB-1', 'P0')
        self.assertIn('OWNER-PRIORITIES.json', K.priorities_line())


class PageTest(Env):
    def test_fragment_renders_and_escapes(self):
        C = db.Collector(self.cfg_path)
        A = db.Actions(C)
        S = {'t': time.time(), 'lanes': [], 'events': [{'kind': 'READY', 'name': '<b>x</b>', 'detail': '"q"', 't': time.time()}],
             'queue': {'rev': 'r', 'sections': db.queue_view(QUEUE)}, 'disk': {'free': 50, 'floor': 30, 'samples': [[0, 1], [1, 2]]},
             'feeds': [{'panels': [{'id': 'fb', 'title': 'Feedback', 'prioritise': True, 'pills': [['open', 2]],
                                    'groups': [{'title': 'Open', 'open': True, 'columns': ['Item'],
                                                'rows': [{'id': 'FB-2', 'cells': ['FB-2']}, {'id': 'FB-1', 'cells': ['FB-1']}]}]}]}],
             'priorities': {'FB-1': {'priority': 'P0'}}}
        h = db.Page(C, A).fragment(S)
        self.assertIn('&lt;b&gt;x&lt;/b&gt;', h)
        self.assertNotIn('<b>x</b>', h)
        self.assertLess(h.index('>FB-1<'), h.index('>FB-2<'))   # owner P0 sorts first
        self.assertIn('data-act="queue-move"', h)


if __name__ == '__main__':
    unittest.main()
