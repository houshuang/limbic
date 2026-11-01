"""Run: python3 skills/campaign-orchestration/scripts/test_lane_keeper.py"""
import sys
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

    def test_retire_does_not_bypass_min_gap(self):
        self.K.dry = True
        self.K.st['events'] = {'k': {'lane': '/x', 'name': 'x', 'kind': 'READY', 'detail': 'd', 't': self.t}}
        self.K.st['coordinator'] = {'last': self.t - 60, 'day': time.strftime('%F'), 'starts': 1}
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 1)   # within min gap
        self.retire_at(self.t - 30)
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 1)   # a two-minute coordinator does not earn an immediate successor
        self.K.st['coordinator']['last'] = self.t - 601
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 2)

    def test_informational_events_never_start_a_coordinator(self):
        # 1 Nov 2026 00:01–02:32: a successor every ~2 min, each only acknowledging HEAVY-WAIT, used all 48 daily starts
        self.K.dry = True
        self.K.st['coordinator'] = {'last': self.t - 3600, 'day': time.strftime('%F'), 'starts': 0}
        self.K.st['events'] = {f'h{i}': {'lane': f'/l{i}', 'name': f'l{i}', 'kind': k, 'detail': 'd', 't': self.t}
                               for i, k in enumerate(['HEAVY-WAIT', 'QUIET', 'STALE-PREVIEW', 'ORPHAN', 'STOPPED-BUT-RUNNING'])}
        for _ in range(5):
            self.K.maybe_start_coordinator()
            self.K.st['coordinator']['last'] = self.t - 3600
        self.assertEqual(self.K.st['coordinator']['starts'], 0)
        self.K.st['events']['r'] = {'lane': '/r', 'name': 'r', 'kind': 'READY', 'detail': 'd', 't': self.t}
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 1)
        self.assertEqual(self.K.st['coordinator']['last_keys'], ['r'])   # stuck detection follows actionable events only

    def test_info_kinds_configurable(self):
        self.K.dry = True
        self.K.cfg['coordinator'] = {'info_kinds': []}
        self.K.st['coordinator'] = {'last': self.t - 3600, 'day': time.strftime('%F'), 'starts': 0}
        self.K.st['events'] = {'h': {'lane': '/l', 'name': 'l', 'kind': 'HEAVY-WAIT', 'detail': 'd', 't': self.t}}
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 1)

    def test_lease_counts_as_live_until_it_expires_or_is_released(self):
        self.assertIsNone(self.K.coordinator_live())
        self.K.set_lease('overnight-executor', 30, 'release 15')
        self.assertEqual(self.K.coordinator_live(), 'lease overnight-executor')
        self.K.dry = True
        self.K.st['coordinator'] = {'last': self.t - 3600, 'day': time.strftime('%F'), 'starts': 0}
        self.K.st['events'] = {'r': {'lane': '/r', 'name': 'r', 'kind': 'READY', 'detail': 'd', 't': self.t}}
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'], 0)   # the executor holds integration
        self.K.release_lease()
        self.assertIsNone(self.K.coordinator_live())
        (self.K.k / 'LEASE').write_text(json.dumps({'owner': 'old', 'until': self.t - 1}))
        self.assertIsNone(self.K.coordinator_live())

    def test_lane_executor_lease_locks_integration_but_not_coordination(self):
        camp = Path(self.K.cfg['campaigns'][0]) if self.K.cfg.get('campaigns') else None
        if camp is None:
            camp = Path(tempfile.mkdtemp()); self.K.cfg['campaigns'] = [str(camp)]
        (camp / 'serial-executor').mkdir(parents=True, exist_ok=True)
        self.K.set_lease('serial-executor', 480, 'queue')
        self.assertIsNone(self.K.coordinator_live())
        self.assertIn('locked by lane serial-executor', self.K.lease_line())
        self.K.set_lease('interactive-night', 60, 'owner session')
        self.assertEqual(self.K.coordinator_live(), 'lease interactive-night')
        self.assertEqual(self.K.lease_line(), '')
        (self.K.k / 'LEASE').write_text('not json')
        self.assertIsNone(self.K.coordinator_live())

    def test_schedule_fires_once_per_applicable_day(self):
        today = dt.datetime.now()
        self.K.cfg['schedule'] = [{'at': '00:00', 'kind': 'CUTOFF', 'note': 'freeze scope'},
                                  {'at': '00:00', 'days': [(today + dt.timedelta(days=1)).strftime('%a')], 'kind': 'RELEASE-DAY'},
                                  {'at': '00:00', 'date': '2000-01-01', 'kind': 'DEADLINE'},
                                  {'at': '23:59' if today.strftime('%H:%M') < '23:59' else '00:00', 'kind': 'LATER'},
                                  {'kind': 'BROKEN'}]
        self.K.schedule_events()
        self.K.schedule_events()
        kinds = [e['kind'] for e in self.K.st['events'].values()]
        self.assertEqual(kinds.count('CUTOFF'), 1)
        self.assertNotIn('RELEASE-DAY', kinds)
        self.assertNotIn('DEADLINE', kinds)
        if today.strftime('%H:%M') < '23:59':
            self.assertNotIn('LATER', kinds)
        self.K.st['events'].clear()
        self.K.schedule_events()
        self.assertEqual(self.K.st['events'], {})   # acknowledged today's cutoff: it does not fire again today

    def test_handover_check_refuses_long_or_run_together_text(self):
        self.K.state_dir.mkdir(parents=True, exist_ok=True)
        good = self.K.state_dir / 'a-HANDOVER-20261101-0100.md'
        good.write_text('Release 15 is live. The songbook lane waits on /Users/stian/src/skard-state/x/READY-2.md.\n' * 10)
        self.assertEqual(self.K.handover_problems(good), [])
        bad = self.K.state_dir / 'b-HANDOVER-20261101-0200.md'
        bad.write_text('Owner Codexchat01a19402,appointed byStian. First release15code/subjectbatch integrated14e11b726,'
                       'feedbackregeneratedeff9c230e;mainclean. Itsthinworktreeautomaticallyremoved,preservedhandoff.')
        self.assertTrue(any('per word' in p or 'run-together' in p for p in self.K.handover_problems(bad)))
        long = self.K.state_dir / 'c-HANDOVER-20261101-0300.md'
        long.write_text('word ' * 700)
        self.assertTrue(any('characters (limit' in p for p in self.K.handover_problems(long)))
        self.assertEqual(self.K.handover(), long)

    def test_retire_cli_refuses_bad_handover_unless_forced(self):
        self.K.state_dir.mkdir(parents=True, exist_ok=True)
        (self.K.state_dir / 'x-HANDOVER-20261101-0300.md').write_text('a' * 3000)
        cfg = str(self.tmp / 'coordinator.json')
        run = lambda *a: subprocess.run([sys.executable, str(Path(__file__).with_name('lane-keeper.py')), 'retire',
                                         '--config', cfg, '--session', 'abc', *a], capture_output=True, text=True,
                                        env={**os.environ, 'HOME': str(self.tmp)})
        r = run()
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('retire refused', r.stdout)
        self.assertFalse((self.K.k / 'retired').exists())
        r = run('--force')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('abc', (self.K.k / 'retired').read_text())

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

    def test_codex_successor_uses_adapter_and_run_specific_retirement(self):
        self.K.cfg['coordinator']={'engine':'codex','model':'fixture-model','effort':'medium'}
        self.K.st['events']={'k':{'lane':'/lane','kind':'READY','detail':'checkpoint'}}
        with mock.patch.object(self.K,'coordinator_live',return_value=None), mock.patch.object(lk.subprocess,'Popen') as popen:
            popen.return_value.pid=99999999
            self.K.maybe_start_coordinator()
        command=popen.call_args.args[0]
        self.assertIn(str(lk.SCRIPTS/'codex-coordinator.py'),command)
        self.assertIn('--reservation',command)
        self.assertIn('fixture-model',command)
        self.assertEqual(popen.call_args.kwargs['stdin'],subprocess.DEVNULL)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        prompt=(self.K.k/'coordinator-prompt.txt').read_text()
        self.assertIn('$(cat ',prompt)
        self.assertIn('.session)',prompt)
        self.assertTrue((self.K.k/'coordinator.session').read_text().startswith('pending-'))

    def test_codex_engine_keeps_start_gap_daily_and_stuck_limits(self):
        self.K.dry=True
        self.K.cfg['coordinator']={'engine':'codex','max_starts_per_day':2,'min_gap_minutes':10}
        self.K.st['events']={'k':{'lane':'/lane','kind':'READY','detail':'checkpoint'}}
        self.K.st['coordinator']={'day':time.strftime('%F'),'starts':1,'last':self.t-60}
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'],1)
        self.K.st['coordinator']['last']=self.t-1000
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'],2)
        self.K.st['coordinator']['last']=self.t-1000
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'],2)
        self.K.st['coordinator'].update(starts=0,same=2,last_keys=['k'],last=self.t-1000)
        self.K.maybe_start_coordinator()
        self.assertEqual(self.K.st['coordinator']['starts'],0)
        self.assertTrue((self.K.k/'COORDINATOR-STUCK.md').exists())


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


class FrictionTest(unittest.TestCase):
    setUp = RetireTest.setUp

    def lane(self):
        C=self.tmp/'campaign'; C.mkdir(exist_ok=True)
        d=C/'lane'; d.mkdir(exist_ok=True)
        (C/'costguard.py').write_text('fixture')
        (d/'pid').write_text('99999999')
        return C,d

    def test_keeper_launch_uses_versioned_guarded_launcher_for_old_campaign_copies(self):
        C,d=self.lane()
        (C/'launch.sh').write_text('unguarded old campaign launcher')
        with mock.patch.object(self.K,'run') as run:
            self.K.launch(C,d,'fixture')
        command=run.call_args.args[0]
        self.assertIn(str(lk.SCRIPTS/'launch.sh'),command)
        self.assertIn(f'CAMPAIGN_DIR={C}',command)

    def test_append_only_log_uses_latest_session_start_for_short_session_detection(self):
        C,d=self.lane()
        (d/'codex.log').write_text('session id: old\nnew session exited\n')
        (d/'fresh-at').write_text(str(self.t-15))
        (d/'session-at').write_text(str(self.t-10))
        with mock.patch.object(lk,'birth',return_value=self.t-5000), mock.patch.object(self.K,'run') as run:
            self.K.launch(C,d,'fixture')
        run.assert_not_called()
        self.assertIn('SHORT-SESSION',[e['kind'] for e in self.K.st['events'].values()])

    def test_append_only_log_does_not_revive_a_previous_sessions_archived_handoff(self):
        C,d=self.lane()
        (d/'codex.log').write_text('session id: old\nsession id: new\n')
        (d/'session-at').write_text(str(self.t-10))
        archived=d/'HANDOFF.0.md'; archived.write_text('previous queue')
        os.utime(archived,(self.t-100,self.t-100))
        with mock.patch.object(lk,'birth',return_value=self.t-5000):
            self.assertFalse(self.K.handoff(d))
        self.assertFalse((d/'HANDOFF.md').exists())

    def test_ready_scope_uses_head_and_does_not_treat_base_or_measurements_as_heads(self):
        repo=self.K.repo
        subprocess.run(['git','init','-q','-b','main',str(repo)],check=True)
        for key,val in [('user.email','t@example.invalid'),('user.name','Test')]:
            subprocess.run(['git','-C',str(repo),'config',key,val],check=True)
        subprocess.run(['git','-C',str(repo),'commit','--allow-empty','-qm','integrated'],check=True)
        sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        ready=self.tmp/'lane/READY-1.md';ready.parent.mkdir()
        ready.write_text(f'HEAD `{sha}`, base `abcdef123456`.\nMeasurement: 1234567 records.\n')
        scope=self.tmp/'RELEASE-SCOPE';scope.write_text('lane/READY-1\n')
        self.K.cfg['release_scope']=str(scope)
        self.K.check_release_scope()
        self.assertEqual([e['kind'] for e in self.K.st['events'].values()],['RELEASE-READY'])

    def test_cost_cap_starts_approved_next_driver_without_a_resume(self):
        C,d=self.lane()
        (d/'NEXT-COMMAND').write_text(f'bash {lk.SCRIPTS / "drive-loop.sh"} --once {d} job 30 -- true\n')
        started=mock.Mock(pid=12345)
        def run(argv, **kwargs):
            return subprocess.CompletedProcess(argv,0,stdout='lane\n' if 'costguard.py' in str(argv) else '')
        with mock.patch.object(lk.subprocess,'run',side_effect=run), \
             mock.patch.object(lk.os,'kill'), mock.patch.object(lk.time,'sleep'), \
             mock.patch.object(self.K,'run',return_value=started) as spawn:
            self.K.costguard(C,['lane'])
        self.assertEqual(spawn.call_count,1)
        argv=spawn.call_args.args[0]
        self.assertIn('drive-loop.sh',' '.join(argv)); self.assertNotIn('resume.sh',' '.join(argv))
        self.assertEqual((d/'driver.pid').read_text().strip(),'12345')
        self.assertFalse((d/'NEXT-COMMAND').exists())

    def test_cost_cap_without_next_keeps_handoff_resume(self):
        C,d=self.lane()
        self.K.session_id=mock.Mock(return_value='11111111-1111-1111-1111-111111111111')
        with mock.patch.object(lk.subprocess,'run',side_effect=lambda argv, **kwargs: subprocess.CompletedProcess(argv,0,stdout='lane' if 'costguard.py' in str(argv) else '')), \
             mock.patch.object(lk.os,'kill',side_effect=ProcessLookupError), mock.patch.object(lk.time,'sleep'), \
             mock.patch.object(self.K,'run') as spawn:
            self.K.costguard(C,['lane'])
        self.assertIn('resume.sh',' '.join(spawn.call_args.args[0]))

    def test_cost_cap_without_verified_session_refuses_handoff_resume(self):
        C,d=self.lane()
        (d/'codex.log').write_text('documentation says `session id: <uuid>`\n')
        with mock.patch.object(lk.subprocess,'run',side_effect=lambda argv, **kwargs: subprocess.CompletedProcess(argv,0,stdout='lane' if 'costguard.py' in str(argv) else '')), \
             mock.patch.object(lk.os,'kill',side_effect=ProcessLookupError), mock.patch.object(lk.time,'sleep'), \
             mock.patch.object(self.K,'run') as spawn:
            self.K.costguard(C,['lane'])
        spawn.assert_not_called()
        self.assertIn('ENDED',[e['kind'] for e in self.K.st['events'].values()])

    @mock.patch.object(lk, "owns_process", return_value=True)
    def test_next_command_rejects_shell_and_other_lane_and_is_not_restarted_twice(self, owner):
        C,d=self.lane()
        for command in ('echo unsafe',f'bash {lk.SCRIPTS / "drive-loop.sh"} {C} job 30 -- true',
                        f'bash {lk.SCRIPTS / "drive-loop.sh"} {d} job 30 -- true\necho unsafe'):
            (d/'NEXT-COMMAND').write_text(command)
            with mock.patch.object(self.K,'run') as run:
                self.assertFalse(self.K.start_next_command(d)); run.assert_not_called()
        (d/'driver.pid').write_text(str(os.getpid()))
        (d/'NEXT-COMMAND').write_text(f'bash {lk.SCRIPTS / "drive-loop.sh"} {d} job 30 -- true')
        with mock.patch.object(self.K,'run') as run:
            self.assertTrue(self.K.start_next_command(d)); run.assert_not_called()

    def test_integrated_scope_emits_once_per_hash_and_waits_for_cycle(self):
        repo=self.K.repo
        subprocess.run(['git','init','-q','-b','main',str(repo)],check=True)
        for key,val in [('user.email','t@example.invalid'),('user.name','Test')]:
            subprocess.run(['git','-C',str(repo),'config',key,val],check=True)
        p=repo/'file';p.write_text('first')
        subprocess.run(['git','-C',str(repo),'add','file'],check=True)
        subprocess.run(['git','-C',str(repo),'commit','-qm','first'],check=True)
        sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        scope=self.tmp/'RELEASE-SCOPE';scope.write_text(sha+'\n')
        self.K.cfg['release_scope']=str(scope)
        hold=self.K.state_dir/'RELEASE-CYCLE-RUNNING';hold.write_text(f'pid={os.getpid()}\n')
        self.K.check_release_scope();self.assertFalse(self.K.st['events'])
        hold.unlink();self.K.check_release_scope()
        self.assertEqual([e['kind'] for e in self.K.st['events'].values()],['RELEASE-READY'])
        self.K.st['events'].clear();self.K.check_release_scope();self.assertFalse(self.K.st['events'])
        scope.write_text(sha+'\n# revised scope\n');self.K.check_release_scope()
        self.assertEqual(len(self.K.st['events']),1)

    def test_next_command_really_runs_detached_in_a_temp_lane(self):
        C,d=self.lane()
        fake=self.tmp/'sleepsafe.py'
        fake.write_text('#!/usr/bin/env python3\nimport subprocess,sys\nsys.exit(subprocess.run(sys.argv[sys.argv.index("--")+1:]).returncode)\n')
        fake.chmod(0o755)
        output=self.tmp/'ran'
        import shlex
        command=['bash',str(lk.SCRIPTS/'drive-loop.sh'),'--once',str(d),'fixture','10','--',
                 sys.executable,'-c',f'from pathlib import Path; Path({str(output)!r}).write_text("yes")']
        (d/'NEXT-COMMAND').write_text(shlex.join(command)+'\n')
        processes=[]
        original=self.K.run
        def run(*args):
            process=original(*args);processes.append(process);return process
        with mock.patch.dict(os.environ,{'SLEEPSAFE':str(fake)}), mock.patch.object(self.K,'run',side_effect=run):
            self.assertTrue(self.K.start_next_command(d))
        try:
            self.assertEqual(processes[0].wait(timeout=15),0,(d/'driver.log').read_text())
        finally:
            if processes[0].poll() is None:
                os.killpg(processes[0].pid,15)
                processes[0].wait(timeout=5)
        self.assertEqual(output.read_text(),'yes')
        self.assertFalse((d/'NEXT-COMMAND').exists())

    @mock.patch.object(lk, "owns_process", return_value=True)
    def test_waiting_for_a_silent_heavy_slot_is_reported_from_process_tree(self, owner):
        C,d=self.lane();(d/'driver.pid').write_text(str(os.getpid()))
        tree=f'{os.getpid()} 1 bash drive-loop.sh\n123456 {os.getpid()} bash scripts/heavy.sh python3 tests/run_parallel.py\n123457 123456 sleep 30\n'
        with mock.patch.object(lk.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout=tree)), \
             mock.patch.object(lk,'newest_in',return_value=self.t-lk.QUIET-1):
            self.K.tend(C,d)
        self.assertIn('HEAVY-WAIT',[e['kind'] for e in self.K.st['events'].values()])
        self.assertFalse((d/'STOP').exists())

    def test_scope_missing_commit_never_ready(self):
        scope=self.tmp/'RELEASE-SCOPE';scope.write_text('f'*40+'\n')
        self.K.cfg['release_scope']=str(scope)
        self.K.check_release_scope();self.assertFalse(self.K.st['events'])

    @mock.patch.object(lk, "owns_process", return_value=True)
    def test_waiting_on_heavy_lock_is_reported_without_stop_or_quiet(self, owner):
        C,d=self.lane()
        (d/'driver.pid').write_text(str(os.getpid()))
        (d/'driver.log').write_text('heavy.sh: memory free 9%, waiting\n')
        with mock.patch.object(lk,'newest_in',return_value=self.t-lk.QUIET-1):
            self.K.tend(C,d)
        self.assertIn('HEAVY-WAIT',[e['kind'] for e in self.K.st['events'].values()])
        self.assertNotIn('QUIET',[e['kind'] for e in self.K.st['events'].values()])
        self.assertFalse((d/'STOP').exists())


    @mock.patch.object(lk, "owns_process", return_value=True)
    def test_acknowledged_heavy_wait_stays_quiet_but_new_driver_reports(self, owner):
        C,d=self.lane()
        (d/'driver.pid').write_text(str(os.getpid()))
        touch(d/'driver.pid',self.t)
        (d/'driver.log').write_text('heavy.sh: memory free 9%, waiting\n')
        self.K.tend(C,d)
        self.assertEqual([e['kind'] for e in self.K.st['events'].values()],['HEAVY-WAIT'])
        (d/'HANDLED').write_text('verified healthy wait')
        touch(d/'HANDLED',self.t+1)
        self.K.clear_events({})
        # Simulate several ticks, including a keeper reload after persisted ack.
        for _ in range(3):
            self.K.tend(C,d)
            self.assertFalse(self.K.st['events'])
        self.K.save()
        self.K=lk.Keeper(self.K.cfg_path)
        self.K.tend(C,d)
        self.assertFalse(self.K.st['events'])
        touch(d/'driver.pid',self.t+2)
        self.K.tend(C,d)
        self.assertEqual([e['kind'] for e in self.K.st['events'].values()],['HEAVY-WAIT'])

    def test_acknowledged_heavy_wait_does_not_hide_ready_or_terminal(self):
        C,d=self.lane()
        (d/'driver.pid').write_text(str(os.getpid()))
        touch(d/'driver.pid',self.t)
        (d/'driver.log').write_text('heavy.sh: waiting for slots\n')
        (d/'HANDLED').write_text('verified healthy wait')
        touch(d/'HANDLED',self.t+1)
        (d/'READY-1.md').write_text('partial checkpoint')
        touch(d/'READY-1.md',self.t+2)
        self.K.tend(C,d)
        self.assertEqual([e['kind'] for e in self.K.st['events'].values()],['READY'])
        (d/'driver.pid').write_text('99999999')
        (d/'DONE.md').write_text('queue outcomes')
        touch(d/'DONE.md',self.t+3)
        self.K.tend(C,d)
        self.assertIn('DONE',[e['kind'] for e in self.K.st['events'].values()])


class SessionGuardTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.C=self.root/'campaign';self.d=self.C/'lane';self.d.mkdir(parents=True)
        (self.C/'briefs').mkdir();(self.C/'briefs/lane.md').write_text('fixture')
        (self.d/'codex.log').write_text('session id: 11111111-1111-1111-1111-111111111111\n')
        self.codex_home=self.root/'codex-home'
        session=self.codex_home/'sessions/2026/10/31/rollout-11111111-1111-1111-1111-111111111111.jsonl'
        session.parent.mkdir(parents=True)
        session.write_text(json.dumps({'type':'session_meta','payload':{'id':'11111111-1111-1111-1111-111111111111'}})+'\n')
        self.msg=self.root/'message';self.msg.write_text('continue')
        self.bin=self.root/'bin';self.bin.mkdir()
        codex=self.bin/'codex';codex.write_text('#!/bin/sh\necho "session id: 22222222-2222-2222-2222-222222222222"\n');codex.chmod(0o755)
        self.env={**os.environ,'PATH':str(self.bin)+':'+os.environ['PATH'],'REPO':str(self.root),
                  'CAMPAIGN_DIR':str(self.C),'ALLOW_UNWATCHED':'1','CODEX_HOME':str(self.codex_home)}

    def call(self,script):
        args=[str(self.C),'lane',str(self.msg)] if script=='resume.sh' else ['lane']
        return subprocess.run(['bash',str(lk.SCRIPTS/script),*args],env=self.env,capture_output=True,text=True,timeout=5)

    def test_both_entrypoints_refuse_live_or_recent_launch_without_touching_log(self):
        for script in ('launch.sh','resume.sh'):
            for live in (True,False):
                with self.subTest(script=script,live=live):
                    (self.d/'pid').write_text(str(os.getpid()) if live else '99999999')
                    if not live:(self.d/'session-at').write_text(str(int(time.time())))
                    before=(self.d/'codex.log').read_bytes()
                    self.assertNotEqual(self.call(script).returncode,0)
                    self.assertEqual((self.d/'codex.log').read_bytes(),before)

    def test_concurrent_resume_has_one_winner_and_preserves_session_id(self):
        argv=['bash',str(lk.SCRIPTS/'resume.sh'),str(self.C),'lane',str(self.msg)]
        procs=[subprocess.Popen(argv,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
        results=[p.communicate(timeout=5) for p in procs]
        self.assertEqual(sorted(p.returncode for p in procs),[0,3],results)
        self.assertIn('session id: 11111111-1111-1111-1111-111111111111',(self.d/'codex.log').read_text())

    def test_concurrent_launch_and_resume_have_one_winner(self):
        commands=[['bash',str(lk.SCRIPTS/'launch.sh'),'lane'],
                  ['bash',str(lk.SCRIPTS/'resume.sh'),str(self.C),'lane',str(self.msg)]]
        procs=[subprocess.Popen(argv,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for argv in commands]
        results=[p.communicate(timeout=5) for p in procs]
        self.assertEqual(sorted(p.returncode for p in procs),[0,3],results)
        self.assertIn('session id: 11111111-1111-1111-1111-111111111111',(self.d/'codex.log').read_text())

    def test_echoed_placeholder_does_not_replace_actual_header(self):
        log=self.d/'codex.log'
        log.write_text(log.read_text()+'launch.sh writes `session id: <uuid>` in logs\nsession id: <uuid>\n')
        self.assertEqual(lk.last_session_id(log),'11111111-1111-1111-1111-111111111111')
        with mock.patch.object(lk,'CODEX_SESSIONS',self.codex_home/'sessions'):
            self.assertEqual(lk.Keeper.session_id(None,self.d),'11111111-1111-1111-1111-111111111111')

    def test_latest_unknown_uuid_never_falls_back_to_previous_session(self):
        log=self.d/'codex.log'
        log.write_text(log.read_text()+'session id: 99999999-9999-9999-9999-999999999999\n')
        self.assertEqual(lk.last_session_id(log),'99999999-9999-9999-9999-999999999999')
        with mock.patch.object(lk,'CODEX_SESSIONS',self.codex_home/'sessions'):
            self.assertIsNone(lk.Keeper.session_id(None,self.d))
        result=self.call('resume.sh')
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertNotIn('resumed lane',result.stdout)
        self.assertFalse((self.d/'pid').exists())

    def test_placeholder_only_log_never_starts_a_session(self):
        (self.d/'codex.log').write_text('session id: <uuid>\n')
        result=self.call('resume.sh')
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertFalse((self.d/'pid').exists())

    def test_mismatched_local_metadata_refuses_resume(self):
        session=next((self.codex_home/'sessions').glob('*/*/*/*.jsonl'))
        session.write_text(json.dumps({'type':'session_meta','payload':{'id':'other-session'}})+'\n')
        self.assertEqual(self.call('resume.sh').returncode,2)

    def test_append_only_headers_and_json_thread_started_choose_latest(self):
        log=self.d/'codex.log'
        # Exercise a header crossing a backwards-read block boundary.
        log.write_text('session id: 11111111-1111-1111-1111-111111111111\n'+'x'*65510+'\n'+'session id: 22222222-2222-2222-2222-222222222222\n'+'y'*65510)
        self.assertEqual(lk.last_session_id(log),'22222222-2222-2222-2222-222222222222')
        with log.open('a') as stream:
            stream.write('\n'+json.dumps({'type':'thread.started','thread_id':'33333333-3333-3333-3333-333333333333'})+'\n')
        self.assertEqual(lk.last_session_id(log),'33333333-3333-3333-3333-333333333333')


class ClaudeEngineTest(unittest.TestCase):
    """Lanes whose model file names a Claude model run `claude -p` instead of `codex exec`."""
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.C=self.root/'campaign';self.d=self.C/'lane';self.d.mkdir(parents=True)
        (self.C/'briefs').mkdir();(self.C/'briefs/lane.md').write_text('BRIEF-FIXTURE')
        self.msg=self.root/'message';self.msg.write_text('MESSAGE-FIXTURE')
        self.bin=self.root/'bin';self.bin.mkdir()
        self.calls=self.root/'calls'
        for name in ('claude','codex'):
            f=self.bin/name
            f.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                         f'open({str(self.calls)!r},"a").write(json.dumps({{"bin":{name!r},"argv":sys.argv[1:],'
                         '"key":os.environ.get("ANTHROPIC_API_KEY")})+"\\n")\n')
            f.chmod(0o755)
        self.home=self.root/'claude-home'
        self.env={**os.environ,'PATH':str(self.bin)+':'+os.environ['PATH'],'REPO':str(self.root),'CAMPAIGN_DIR':str(self.C),
                  'ALLOW_UNWATCHED':'1','ANTHROPIC_API_KEY':'sk-fixture','CLAUDE_CONFIG_DIR':str(self.home)}

    def run_script(self,script,*args):
        r=subprocess.run(['bash',str(lk.SCRIPTS/script),*args],env=self.env,capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        for _ in range(50):   # the worker is started with nohup in the background
            if self.calls.exists() and self.calls.read_text().strip(): break
            time.sleep(0.1)
        return [json.loads(l) for l in self.calls.read_text().splitlines()]

    def claude_session(self,sid):
        p=self.home/'projects'/'-fixture-repo'; p.mkdir(parents=True,exist_ok=True); (p/f'{sid}.jsonl').write_text('{}\n')

    def test_launch_with_claude_model_starts_claude_with_session_id_in_log(self):
        calls=self.run_script('launch.sh','lane','sonnet','high')
        self.assertEqual([c['bin'] for c in calls],['claude'])
        argv=calls[0]['argv']
        for flag in ('-p','--verbose'): self.assertIn(flag,argv)
        self.assertEqual(argv[argv.index('--model')+1],'sonnet')
        self.assertEqual(argv[argv.index('--permission-mode')+1],'bypassPermissions')
        self.assertEqual(argv[argv.index('--output-format')+1],'stream-json')
        sid=argv[argv.index('--session-id')+1]
        self.assertIn('BRIEF-FIXTURE',argv[argv.index('-p')+1])
        self.assertIsNone(calls[0]['key'])
        self.assertEqual(lk.last_session_id(self.d/'codex.log'),sid)
        self.assertEqual((self.d/'model').read_text().split()[0],'sonnet')
        self.assertTrue((self.d/'pid').exists() and (self.d/'session-at').exists())

    def test_launch_with_codex_model_keeps_codex(self):
        calls=self.run_script('launch.sh','lane','gpt-6.1-sol','medium')
        self.assertEqual([c['bin'] for c in calls],['codex'])
        self.assertEqual(calls[0]['argv'][:3],['exec','-m','gpt-6.1-sol'])

    def test_resume_claude_lane_resumes_its_claude_session(self):
        (self.d/'codex.log').write_text('session id: 11111111-2222-3333-4444-555555555555\n')
        (self.d/'model').write_text('opus high\n')
        self.claude_session('11111111-2222-3333-4444-555555555555')
        calls=self.run_script('resume.sh',str(self.C),'lane',str(self.msg))
        self.assertEqual([c['bin'] for c in calls],['claude'])
        argv=calls[0]['argv']
        self.assertEqual(argv[argv.index('--resume')+1],'11111111-2222-3333-4444-555555555555')
        self.assertIn('MESSAGE-FIXTURE',argv); self.assertIn('-p',argv); self.assertNotIn('--session-id',argv)
        self.assertEqual(argv[argv.index('--permission-mode')+1],'bypassPermissions')
        self.assertIsNone(calls[0]['key'])

    def test_switch_from_codex_to_claude_starts_fresh_claude_session_with_handoff(self):
        (self.d/'codex.log').write_text('session id: 33333333-3333-3333-3333-333333333333\n')
        (self.d/'model').write_text('sonnet medium\n')   # rewritten by the coordinator from a Codex model
        (self.d/'HANDOFF.md').write_text('HANDOFF-FIXTURE')
        calls=self.run_script('resume.sh',str(self.C),'lane',str(self.msg))
        self.assertEqual([c['bin'] for c in calls],['claude'])
        argv=calls[0]['argv']
        self.assertNotIn('--resume',argv)
        sid=argv[argv.index('--session-id')+1]
        prompt=argv[argv.index('-p')+1]
        self.assertIn('BRIEF-FIXTURE',prompt); self.assertIn('HANDOFF-FIXTURE',prompt)
        self.assertEqual(argv[argv.index('--model')+1],'sonnet')
        self.assertEqual(lk.last_session_id(self.d/'codex.log'),sid)

    def test_keeper_session_id_accepts_claude_session_file_for_claude_lane(self):
        (self.d/'codex.log').write_text('session id: 44444444-4444-4444-4444-444444444444\n'); (self.d/'model').write_text('sonnet medium\n')
        with mock.patch.object(lk,'CLAUDE_PROJECTS',self.home/'projects'):
            self.assertIsNone(lk.Keeper.session_id(None,self.d))
            self.claude_session('44444444-4444-4444-4444-444444444444')
            self.assertEqual(lk.Keeper.session_id(None,self.d),'44444444-4444-4444-4444-444444444444')

    def test_keeper_resume_of_switched_lane_launches_fresh(self):
        K=mock.Mock(); K.dry=False
        (self.d/'codex.log').write_text('session id: 33333333-3333-3333-3333-333333333333\n'); (self.d/'model').write_text('opus high\n')
        with mock.patch.object(lk,'CLAUDE_PROJECTS',self.home/'projects'):
            K.session_id=lambda d: lk.Keeper.session_id(K,d)
            lk.Keeper.resume(K,self.C,self.d,self.msg,'continue #1')
        K.launch.assert_called_once()
        K.run.assert_not_called(); K.event.assert_not_called()

    def test_codex_lane_without_session_still_ends(self):
        K=mock.Mock()
        (self.d/'codex.log').write_text('session id: gone\n'); (self.d/'model').write_text('gpt-6.1-sol medium\n')
        K.session_id=lambda d: lk.Keeper.session_id(K,d)
        lk.Keeper.resume(K,self.C,self.d,self.msg,'continue #1')
        K.launch.assert_not_called(); K.event.assert_called_once()

    def test_claude_limit_messages_count_as_capacity(self):
        for line in ('{"type":"result","is_error":true,"result":"Claude AI usage limit reached|1793432400"}',
                     '{"type":"result","is_error":true,"result":"API Error: 529 {\\"type\\":\\"error\\",\\"error\\":{\\"type\\":\\"overloaded_error\\"}}"}',
                     '{"type":"result","is_error":true,"result":"API Error: 429 rate_limit_error"}',
                     '{"type":"rate_limit_event","rate_limit_info":{"status":"rejected","resetsAt":1793432400}}',
                     "You've hit your session limit \u00b7 resets 4am"):
            with self.subTest(line=line): self.assertTrue(lk.TRANSIENT.search(line))
        self.assertFalse(lk.TRANSIENT.search('{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","resetsAt":1}}'))

    def test_costguard_skips_claude_lanes_and_reads_stream_json_without_crashing(self):
        (self.d/'model').write_text('sonnet medium\n')
        (self.d/'codex.log').write_text('session id: abc\n{"type":"system","subtype":"init"}\n{"type":"result","usage":{"input_tokens":5}}\n')
        r=subprocess.run([sys.executable,str(lk.SCRIPTS/'costguard.py'),'lane'],env={**self.env,'CAMPAIGN_DIR':str(self.C)},
                         capture_output=True,text=True,timeout=10)
        self.assertEqual((r.returncode,r.stdout.strip()),(0,''),r.stderr)
        self.assertIn('claude',(self.C/'COST.md').read_text())


class RunawayTest(unittest.TestCase):
    """ORPHAN / STOPPED-BUT-RUNNING / STALE-PREVIEW / RESOURCE-FLOOR checks in the keeper tick."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        self.repo = self.tmp / 'repo'; self.wt = self.repo / '.claude/worktrees/lane-a'; self.wt.mkdir(parents=True)
        self.C = self.tmp / 'campaign'; self.d = self.C / 'lane'; self.d.mkdir(parents=True)
        self.cfg_extra = {}
        self.disk, self.swap = 100.0, 1.0
        self.killed = []
        self.table = {}
        self.cwd = {}

    def keeper(self, **extra):
        cfg = self.tmp / 'coordinator.json'
        cfg.write_text(json.dumps({'repo': str(self.repo), 'campaigns': [str(self.C)], 'state': str(self.tmp / 'state'),
                                   'disk_watch': [str(self.repo)], **extra}))
        return lk.Keeper(cfg)

    def proc(self, pid, cmd, ppid=1, age=3600, rss=6_000_000, pgid=None):
        self.table[pid] = {'pid': pid, 'ppid': ppid, 'pgid': pgid or pid, 'age': age, 'rss': rss, 'cmd': cmd}

    def scan(self, K):
        with mock.patch.object(lk, 'ps_table', return_value=self.table), \
             mock.patch.object(lk, 'proc_cwds', side_effect=lambda pids: {p: self.cwd[p] for p in pids if p in self.cwd}), \
             mock.patch('driver_state.process_cwd', side_effect=lambda pid: self.cwd.get(int(pid), '')),  \
             mock.patch.object(lk, 'disk_free_gb', side_effect=lambda p: self.disk), \
             mock.patch.object(lk, 'swap_used_gb', side_effect=lambda: self.swap), \
             mock.patch.object(lk, 'top_processes', return_value='1 7366M python3 run_unit.py'), \
             mock.patch.object(lk, 'largest_dirs', return_value='13.0G /repo/data'), \
             mock.patch.object(lk.Keeper, 'kill_pids', side_effect=lambda pids, why: self.killed.append(sorted(pids))):
            K.runaways()

    def kinds(self, K):
        return [e['kind'] for e in K.st['events'].values()]

    def test_orphan_under_worktree_is_reported_not_reaped_by_default(self):
        self.proc(500, f'python3 tests/run_parallel.py'); self.cwd[500] = str(self.wt)
        self.proc(501, 'python3 tests/run_unit.py test_ui301', ppid=500, pgid=500)
        K = self.keeper(); self.scan(K)
        self.assertEqual(self.kinds(K), ['ORPHAN'])
        self.assertIn('500', next(iter(K.st['events'].values()))['detail'])
        self.assertEqual(self.killed, [])

    def test_orphan_reaped_with_its_tree_when_enabled(self):
        self.proc(500, f'python3 {self.wt}/tests/run_parallel.py')
        self.proc(501, 'python3 tests/run_unit.py', ppid=500, pgid=500)
        K = self.keeper(reap_orphans=True); self.scan(K)
        self.assertEqual(self.killed, [[500, 501]])

    def test_seal_runs_are_never_reaped(self):
        # 1 Nov 2026: a 5.3 GB seal-to-box of campaign evidence ran detached from the session that started it
        self.proc(700, f'python3 /Users/x/.agents/bin/sleepsafe.py --name seal-skard-lane-evidence -- seal-to-box ev {self.C}/lane/http')
        self.proc(701, f'rsync -a {self.C}/lane/http box:', ppid=700)
        self.proc(702, f'/Users/x/src/storage/tools/seal-to-box ev {self.C}/lane/images')
        self.proc(500, f'python3 {self.wt}/tests/run_parallel.py')
        K = self.keeper(reap_orphans=True); self.scan(K)
        self.assertEqual(self.killed, [[500]])

    def test_release_and_heavy_holders_protect_wrapper_and_descendants(self):
        for heavy in (False, True):
            with self.subTest(heavy=heavy):
                self.killed.clear()
                self.proc(500, f'bash {self.wt}/release-wrapper.sh')
                self.proc(501, 'bash release_cycle.sh --full', ppid=500)
                self.proc(502, 'node export-structured.mjs', ppid=501)
                self.proc(600, f'python3 {self.wt}/unrelated.py')
                K = self.keeper(reap_orphans=True)
                K.state_dir.mkdir(parents=True, exist_ok=True)
                hold = K.state_dir / 'RELEASE-CYCLE-RUNNING'
                hold.unlink(missing_ok=True)
                if heavy:
                    lock = K.state_dir / '.heavy.exclusive'
                    lock.mkdir(exist_ok=True); (lock/'pid').write_text('501')
                else:
                    hold.write_text('pid=501\nstage=export\n')
                self.scan(K)
                self.assertEqual(self.killed, [[600]])

    def test_stopped_lane_cannot_override_active_release_protection(self):
        self.proc(700, f'bash drive-loop.sh {self.d} release 172800')
        self.proc(701, 'bash release_cycle.sh --full', ppid=700)
        (self.d/'driver.pid').write_text('700')
        stop=self.d/'STOP';stop.write_text('batch freeze');touch(stop,time.time()-700)
        K=self.keeper(reap_orphans=True)
        K.state_dir.mkdir(parents=True,exist_ok=True)
        (K.state_dir/'RELEASE-CYCLE-RUNNING').write_text('pid=701\nstage=gate\n')
        with mock.patch.object(lk,'alive',return_value=True): self.scan(K)
        self.assertEqual(self.killed, [])

    def test_not_orphans_young_outside_lanes_tracked_or_sessions(self):
        self.proc(600, f'python3 {self.wt}/x.py', age=60)                       # younger than 30 min
        self.proc(601, 'python3 /elsewhere/daemon.py')                          # not under a lane root
        self.proc(602, f'bash drive-loop.sh {self.C}/lane job 3000')            # the lane's tracked driver
        (self.d / 'driver.pid').write_text('602')
        self.proc(603, f'/Users/x/.local/bin/claude -p "work in {self.wt}"')    # a session
        self.proc(604, f'node {self.wt}/thing.js', ppid=603)                    # inside a session's tree, not ppid 1
        self.proc(605, f'python3 {self.wt}/coordinator-child.py')
        (self.tmp / 'state/keeper').mkdir(parents=True); (self.tmp / 'state/keeper/coordinator.pid').write_text('605')
        K = self.keeper(reap_orphans=True); self.scan(K)
        self.assertEqual(self.kinds(K), []); self.assertEqual(self.killed, [])

    def test_keeper_never_reaps_its_own_tree(self):
        self.proc(os.getpid(), f'python3 {self.wt}/lane-keeper.py tick')
        K = self.keeper(reap_orphans=True); self.scan(K)
        self.assertEqual(self.killed, [])

    def test_orphan_reported_once_even_after_ack(self):
        self.proc(500, f'python3 {self.wt}/run_parallel.py')
        K = self.keeper(); self.scan(K)
        (K.k / 'HANDLED').write_text('acked'); K.clear_events({})
        self.scan(K)
        self.assertEqual(self.kinds(K), [])

    def test_stopped_lane_driver_alive_after_ten_minutes(self):
        self.proc(700, f'bash drive-loop.sh {self.d} job 172800 -- true')
        (self.d / 'driver.cmd').write_text(json.dumps({'cwd':str(self.repo),'args':[str(self.d),'job','172800','--','true']}))
        self.cwd[700] = str(self.repo)
        self.proc(701, 'python3 sleepsafe.py --name job --limit 172800 -- python3 driver.py', ppid=700, pgid=700)
        (self.d / 'driver.pid').write_text('700')
        stop = self.d / 'STOP'; stop.write_text('stop'); touch(stop, time.time() - 300)
        with mock.patch.object(lk, 'alive', side_effect=lambda f, pid=None: f.name == 'driver.pid'):
            K = self.keeper(reap_orphans=True); self.scan(K)
            self.assertEqual((self.kinds(K), self.killed), ([], []))
            touch(stop, time.time() - 700); self.scan(K)
        self.assertEqual(self.kinds(K), ['STOPPED-BUT-RUNNING'])
        self.assertEqual(self.killed, [[700, 701]])

    def test_stale_preview_from_ended_session(self):
        self.proc(800, f'node {self.repo}/web/delivery/local.mjs --export x', age=7 * 3600)
        self.proc(801, f'node {self.wt}/web/delivery/local.mjs', age=7 * 3600, ppid=803)
        self.proc(803, '/Users/x/.local/bin/claude --dangerously-skip-permissions', ppid=5)   # live session
        self.proc(802, f'node {self.wt}/web/delivery/local.mjs', age=2 * 3600)
        K = self.keeper(reap_orphans=True); self.scan(K)
        self.assertEqual(self.kinds(K), ['STALE-PREVIEW'])
        self.assertEqual(self.killed, [[800]])

    def test_resource_floor_once_per_hour_with_top_and_dirs(self):
        K = self.keeper(); self.scan(K)
        self.assertEqual(self.kinds(K), [])
        self.disk, self.swap = 18.0, 45.0
        self.scan(K); self.scan(K)
        self.assertEqual(self.kinds(K), ['RESOURCE-FLOOR'])
        detail = next(iter(K.st['events'].values()))['detail']
        for part in ('18.0 GB', '45.0 GB', 'run_unit.py', '/repo/data'): self.assertIn(part, detail)

    def test_resource_floor_config(self):
        self.disk = 18.0
        K = self.keeper(disk_floor_gb=10, swap_ceiling_gb=50); self.scan(K)
        self.assertEqual(self.kinds(K), [])

    def test_real_orphan_is_killed(self):
        """A real double-forked sleep under a temp worktree: reparented to pid 1, then reaped."""
        marker = f'reaper-fixture-{os.getpid()}'
        subprocess.run(['sh', '-c', f'cd {self.wt} && (python3 -c "import time; time.sleep(300)" {marker} '
                        '</dev/null >/dev/null 2>&1 &)'], check=True, timeout=10)
        time.sleep(0.5)
        table = lk.ps_table()
        pids = [p for p, r in table.items() if marker in r['cmd'] and r['ppid'] == 1]
        if not pids: self.skipTest('orphans are not reparented to pid 1 here')
        K = self.keeper(reap_orphans=True, orphan_age_minutes=0, reap_grace_seconds=2)
        with mock.patch.object(lk, 'disk_free_gb', return_value=100.0), mock.patch.object(lk, 'swap_used_gb', return_value=0.0):
            K.runaways()
        time.sleep(0.3)
        self.assertFalse([p for p, r in lk.ps_table().items() if marker in r['cmd']])
        self.assertIn('ORPHAN', self.kinds(K))

    def test_etime_parsing(self):
        self.assertEqual([lk.etime_seconds(s) for s in ('05', '01:02', '03:00:01', '02-01:00:00')], [5, 62, 10801, 176400])


if __name__ == '__main__':
    unittest.main()
