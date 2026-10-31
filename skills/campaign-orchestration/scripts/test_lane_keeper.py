"""Run: python3 skills/campaign-orchestration/scripts/test_lane_keeper.py"""
import datetime as dt
import importlib.util, io, json, os, subprocess, tempfile, time, unittest
from unittest import mock
from pathlib import Path

spec = importlib.util.spec_from_file_location('lane_keeper', Path(__file__).with_name('lane-keeper.py'))
lk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lk)
hook_spec = importlib.util.spec_from_file_location('coordinator_hook', Path(__file__).with_name('coordinator-hook.py'))
hook = importlib.util.module_from_spec(hook_spec)
hook_spec.loader.exec_module(hook)


def touch(p, t):
    p.touch(); os.utime(p, (t, t))


class RetireTest(unittest.TestCase):
    def setUp(self):
        env_patch = mock.patch.dict(os.environ)
        env_patch.start(); self.addCleanup(env_patch.stop)
        for var in ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_SESSION_ID'):
            os.environ.pop(var, None)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)
        (self.tmp / 'repo').mkdir()
        cfg = self.tmp / 'coordinator.json'
        cfg.write_text(json.dumps({'repo': str(self.tmp / 'repo'), 'campaigns': [], 'state': str(self.tmp / 'state')}))
        saved = lk.REG
        self.addCleanup(setattr, lk, 'REG', saved)
        lk.REG = self.tmp / 'reg'
        lk.REG.mkdir()
        self.K = lk.Keeper(cfg)
        self.t = time.time()

    def register(self, sid, reg_t, transcript_t):
        tr = self.tmp / f'{sid}.jsonl'
        tr.write_text(json.dumps({'type': 'assistant', 'timestamp': self.iso(transcript_t),
                                  'message': {'content': []}}) + '\n')
        os.utime(tr, (transcript_t, transcript_t))
        f = lk.REG / f'{sid}.json'
        f.write_text(json.dumps({'repo': str(self.K.repo), 'transcript': str(tr)}))
        os.utime(f, (reg_t, reg_t))
        return f

    @staticmethod
    def iso(t):
        return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()

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

    def append(self, sid, kind, t, **fields):
        with (self.tmp / f'{sid}.jsonl').open('a') as fh:
            fh.write(json.dumps({'type': kind, 'timestamp': self.iso(t), **fields}) + '\n')

    def test_metadata_mtime_never_extends_assistant_liveness(self):
        self.register('quiet', self.t - 4000, self.t - lk.COORD_LIVE - 1)
        for kind in ('attachment', 'ai-title', 'last-prompt', 'progress', 'system'):
            self.append('quiet', kind, self.t)
        self.assertIsNone(self.K.coordinator_live())

    def test_user_tool_result_counts_but_plain_user_prompt_does_not(self):
        self.register('working', self.t - 4000, self.t - lk.COORD_LIVE - 10)
        self.append('working', 'user', self.t, message={'content': 'hello'})
        self.assertIsNone(self.K.coordinator_live())
        self.append('working', 'user', self.t, message={'content': [
            {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'done'}]})
        self.assertEqual(self.K.coordinator_live(), 'session working active')

    def test_invalid_tail_entries_do_not_hide_recent_assistant(self):
        self.register('working', self.t - 4000, self.t - 10)
        tr = self.tmp / 'working.jsonl'
        with tr.open('a') as fh:
            fh.write('not json\n')
            fh.write(json.dumps({'type': 'assistant', 'timestamp': 'invalid'}) + '\n')
            fh.write('{"type":"assistant",')  # in-flight append
        self.assertEqual(self.K.coordinator_live(), 'session working active')

    def test_dead_watch_pid_does_not_extend_a_quiet_coordinators_liveness(self):
        self.register('quiet', self.t - 4000, self.t - lk.COORD_LIVE - 1)
        child = subprocess.Popen(['true']); child.wait()
        (self.K.k / 'watch.heartbeat').write_text(f'quiet {child.pid}\n')
        self.assertIsNone(self.K.coordinator_live())

    def test_9bfba72f_monitor_stops_then_metadata_keeps_arriving(self):
        sid = '9bfba72f-13d0-493c-9525-e754bdfd80c4'
        self.register(sid, self.t - 9000, self.t - 80 * 60)  # finished 01:30
        touch(self.K.k / 'watch.heartbeat', self.t - 80 * 60)
        for minutes in (60, 40, 20, 0):
            self.append(sid, 'attachment', self.t - minutes * 60)
            self.append(sid, 'ai-title', self.t - minutes * 60)
            self.append(sid, 'last-prompt', self.t - minutes * 60)
        self.K.dry = True
        self.K.st['events'] = {'ready': {'lane': '/reader-readings', 'kind': 'READY',
                                       'detail': 'READY-1.md', 't': self.t - 17 * 60}}
        self.assertIsNone(self.K.coordinator_live())
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 1)

    def test_transcript_tail_is_bounded_and_does_not_use_mtime_when_no_work_is_found(self):
        tr = self.tmp / 'large.jsonl'
        old = json.dumps({'type': 'assistant', 'timestamp': self.iso(self.t - 4000)}) + '\n'
        recent = json.dumps({'type': 'assistant', 'timestamp': self.iso(self.t)}) + '\n'
        tr.write_text(old + ('{"type":"attachment","data":"padding"}\n' * 100) + recent + '{partial')
        with mock.patch.object(lk, 'TRANSCRIPT_TAIL', 256):
            self.assertAlmostEqual(lk.transcript_activity(tr), self.t, places=4)
            tr.write_text(old + ('{"type":"attachment","data":"padding"}\n' * 100))
            self.assertEqual(lk.transcript_activity(tr), 0)

    def test_assistant_timestamp_with_z_counts_even_when_file_mtime_is_old(self):
        self.register('recent', self.t - 4000, self.t - 4000)
        tr = self.tmp / 'recent.jsonl'
        tr.write_text(json.dumps({'type': 'assistant', 'timestamp': self.iso(self.t).replace('+00:00', 'Z')}) + '\n')
        os.utime(tr, (self.t - 4000, self.t - 4000))
        self.assertEqual(self.K.coordinator_live(), 'session recent active')

    def test_retired_id_cannot_revive_by_registry_or_transcript_writes(self):
        for sid in ('f57f51d7', '1f992bfb'):
            with self.subTest(sid=sid):
                self.retire_at(self.t - lk.RETIRE_GRACE - 1, sid)
                self.register(sid, self.t, self.t)
                self.assertIsNone(self.K.coordinator_live())
                (lk.REG / f'{sid}.json').unlink()

    def test_older_retired_ids_stay_retired_after_successor_retires(self):
        self.retire_at(self.t - 200, 'old')
        self.retire_at(self.t - 100, 'successor')
        self.register('old', self.t, self.t)
        self.assertIsNone(self.K.coordinator_live())

    def test_retired_id_cannot_revive_via_headless_pid_or_watch(self):
        self.retire_at(self.t - 100, 'old')
        (self.K.k / 'coordinator.session').write_text('old\n')
        touch(self.K.k / 'coordinator.pid', self.t)
        (self.K.k / 'watch.heartbeat').write_text('old 123\n')
        with mock.patch.object(lk, 'alive', return_value=True):
            self.assertIsNone(self.K.coordinator_live())

    def test_start_command_streams_with_session_id(self):
        cmd = lk.Keeper.coordinator_cmd('p', 'sid-1')
        joined = ' '.join(cmd)
        self.assertIn('--output-format stream-json', joined)
        self.assertIn('--verbose', cmd)
        self.assertIn('--session-id sid-1', joined)


class RegistrationTest(unittest.TestCase):
    register = RetireTest.register
    iso = staticmethod(RetireTest.iso)

    def setUp(self):
        RetireTest.setUp(self)
        self.cfg = {**self.K.cfg}
        (self.K.repo / '.coordinator.json').write_text(json.dumps(self.cfg))
        self.sid = '1f10ba84-4b9b-418b-8680-8056e4917781'
        self.payload = {'session_id': self.sid, 'cwd': str(self.K.repo),
                        'transcript_path': str(self.tmp / 'watchdog.jsonl')}
        patcher = mock.patch.object(hook, 'REG', lk.REG)
        patcher.start(); self.addCleanup(patcher.stop)

    def call_hook(self, command, **updates):
        payload = {**self.payload, **updates, 'tool_input': {'command': command}}
        with mock.patch.object(hook.sys, 'argv', ['coordinator-hook.py', 'register']), \
             mock.patch.object(hook.sys, 'stdin', io.StringIO(json.dumps(payload))), \
             mock.patch.dict(os.environ, {'CLAUDE_CODE_SESSION_ID': self.sid}):
            hook.main()

    def test_non_coordinator_commands_and_quoted_watch_do_not_register(self):
        for command in ('lane-keeper.py tick', 'lane-keeper.py retire --session other',
                        'lane-keeper.py status', 'lane-keeper.py ack /lane handled',
                        'bash launch.sh campaign lane', 'bash drive-loop.sh /lane job 10 -- true',
                        "cat > HANDOVER.md <<'EOF'\nRun lane-keeper.py watch next.\nEOF"):
            with self.subTest(command=command):
                self.call_hook(command)
                self.assertFalse((lk.REG / f'{self.sid}.json').exists())

    def test_keeper_cli_never_registers_the_callers_environment_session(self):
        for args in (['status'], ['tick'], ['retire', '--session', 'other']):
            with self.subTest(args=args), \
                 mock.patch.object(lk.sys, 'argv', ['lane-keeper.py', *args, '--config', str(self.K.cfg_path)]), \
                 mock.patch.dict(os.environ, {'CLAUDE_CODE_SESSION_ID': self.sid}):
                lk.main()
                self.assertFalse((lk.REG / f'{self.sid}.json').exists())

    def test_actual_watch_session_can_register(self):
        (self.K.k / 'watch.heartbeat').write_text(f'{self.sid} {os.getpid()}\n')
        self.call_hook('python3 lane-keeper.py watch')
        self.assertEqual(json.loads((lk.REG / f'{self.sid}.json').read_text())['transcript'],
                         self.payload['transcript_path'])

    def test_only_started_headless_id_registers_not_callers_env_id(self):
        (self.K.k / 'coordinator.session').write_text('headless\n')
        (self.K.k / 'coordinator.pid').write_text(str(os.getpid()))
        self.call_hook('lane-keeper.py tick')
        self.assertFalse((lk.REG / f'{self.sid}.json').exists())
        self.call_hook('git status', session_id='headless')
        self.assertTrue((lk.REG / 'headless.json').exists())

    def test_retired_hook_cannot_reregister_and_removes_a_late_write(self):
        (self.K.k / 'watch.heartbeat').write_text(f'{self.sid} {os.getpid()}\n')
        self.K.retire(self.sid)
        self.call_hook('lane-keeper.py watch')
        self.assertFalse((lk.REG / f'{self.sid}.json').exists())
        self.register(self.sid, self.t, self.t)
        self.call_hook('git status')
        self.assertFalse((lk.REG / f'{self.sid}.json').exists())

    def test_watch_records_owner_and_removes_heartbeat_when_it_exits(self):
        def stop_watch(_):
            self.assertEqual((self.K.k / 'watch.heartbeat').read_text().split()[0], self.sid)
            self.call_hook('git status')
            self.assertTrue((lk.REG / f'{self.sid}.json').exists())
            raise KeyboardInterrupt
        with mock.patch.object(lk.sys, 'argv', ['lane-keeper.py', 'watch', '--config', str(self.K.cfg_path)]), \
             mock.patch.dict(os.environ, {'CLAUDE_CODE_SESSION_ID': self.sid}), \
             mock.patch.object(lk.time, 'sleep', side_effect=stop_watch):
            with self.assertRaises(KeyboardInterrupt):
                lk.main()
        self.assertFalse((self.K.k / 'watch.heartbeat').exists())


if __name__ == '__main__':
    unittest.main()
