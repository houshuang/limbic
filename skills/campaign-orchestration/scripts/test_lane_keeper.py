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
        with mock.patch.object(lk.subprocess,'run',side_effect=lambda argv, **kwargs: subprocess.CompletedProcess(argv,0,stdout='lane' if 'costguard.py' in str(argv) else '')), \
             mock.patch.object(lk.os,'kill',side_effect=ProcessLookupError), mock.patch.object(lk.time,'sleep'), \
             mock.patch.object(self.K,'run') as spawn:
            self.K.costguard(C,['lane'])
        self.assertIn('resume.sh',' '.join(spawn.call_args.args[0]))

    def test_next_command_rejects_shell_and_other_lane_and_is_not_restarted_twice(self):
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

    def test_waiting_for_a_silent_heavy_slot_is_reported_from_process_tree(self):
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

    def test_waiting_on_heavy_lock_is_reported_without_stop_or_quiet(self):
        C,d=self.lane()
        (d/'driver.pid').write_text(str(os.getpid()))
        (d/'driver.log').write_text('heavy.sh: memory free 9%, waiting\n')
        with mock.patch.object(lk,'newest_in',return_value=self.t-lk.QUIET-1):
            self.K.tend(C,d)
        self.assertIn('HEAVY-WAIT',[e['kind'] for e in self.K.st['events'].values()])
        self.assertNotIn('QUIET',[e['kind'] for e in self.K.st['events'].values()])
        self.assertFalse((d/'STOP').exists())


class SessionGuardTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.C=self.root/'campaign';self.d=self.C/'lane';self.d.mkdir(parents=True)
        (self.C/'briefs').mkdir();(self.C/'briefs/lane.md').write_text('fixture')
        (self.d/'codex.log').write_text('session id: old-fixture\n')
        self.msg=self.root/'message';self.msg.write_text('continue')
        self.bin=self.root/'bin';self.bin.mkdir()
        codex=self.bin/'codex';codex.write_text('#!/bin/sh\necho "session id: new-fixture"\n');codex.chmod(0o755)
        self.env={**os.environ,'PATH':str(self.bin)+':'+os.environ['PATH'],'REPO':str(self.root),
                  'CAMPAIGN_DIR':str(self.C),'ALLOW_UNWATCHED':'1'}

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
        self.assertIn('session id: old-fixture',(self.d/'codex.log').read_text())

    def test_concurrent_launch_and_resume_have_one_winner(self):
        commands=[['bash',str(lk.SCRIPTS/'launch.sh'),'lane'],
                  ['bash',str(lk.SCRIPTS/'resume.sh'),str(self.C),'lane',str(self.msg)]]
        procs=[subprocess.Popen(argv,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for argv in commands]
        results=[p.communicate(timeout=5) for p in procs]
        self.assertEqual(sorted(p.returncode for p in procs),[0,3],results)
        self.assertIn('session id: old-fixture',(self.d/'codex.log').read_text())


if __name__ == '__main__':
    unittest.main()
