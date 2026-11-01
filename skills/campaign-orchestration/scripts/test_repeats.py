"""Offline regressions for the repeated 1 November 2026 lane failures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import test_lane_keeper as fixtures
import deliverables
lk = fixtures.lk
SCRIPTS = Path(__file__).resolve().parent


class ContractsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.lane = self.root / 'lane'; self.lane.mkdir()
        self.safe = self.root / 'sleepsafe.py'
        self.safe.write_text('#!/bin/bash\nshift 4; shift; exec "$@"\n'); self.safe.chmod(0o755)

    def drive(self):
        return subprocess.run(['bash', str(SCRIPTS/'drive-loop.sh'), '--once', str(self.lane), 'fixture', '30', '--', 'true'],
                              env={**os.environ, 'SLEEPSAFE': str(self.safe)}, capture_output=True, text=True)

    def test_zero_exit_missing_deliverable_is_failure_65(self):
        (self.lane/'EXPECTS').write_text('{"files":["DONE.md"]}')
        result = self.drive()
        self.assertEqual(result.returncode, 65, result.stdout + result.stderr)
        self.assertEqual((self.lane/'DRIVER-EXIT').read_text().strip(), '65')
        self.assertIn('missing file: DONE.md', (self.lane/'DRIVER-FAILED.md').read_text())
        (self.lane/'DONE.md').write_text('complete')
        self.assertEqual(self.drive().returncode, 0)
        self.assertFalse((self.lane/'DRIVER-FAILED.md').exists())

    def test_jsonl_must_be_nonempty_and_parseable(self):
        (self.lane/'EXPECTS').write_text('{"jsonl":["intake.jsonl"]}')
        for content, reason in [('', 'empty JSONL'), ('{bad}\n', 'invalid JSONL')]:
            (self.lane/'intake.jsonl').write_text(content)
            self.assertEqual(self.drive().returncode, 65)
            self.assertIn(reason + ': intake.jsonl', (self.lane/'DRIVER-FAILED.md').read_text())
        (self.lane/'intake.jsonl').write_text('{}\n')
        self.assertEqual(self.drive().returncode, 0)

    def test_commit_contract_refuses_failed_done_without_acceptance_commit(self):
        def git(*args):
            return subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True, text=True).stdout.strip()
        git('init', '-b', 'accepted'); git('config','user.name','Fixture'); git('config','user.email','f@example.org')
        git('commit','--allow-empty','-m','base'); base = git('rev-parse','HEAD')
        (self.lane/'DONE.md').write_text('failed staging; next: integrate')
        (self.lane/'EXPECTS').write_text(json.dumps({'files':['DONE.md'], 'commit':{'repo':'..','base':base,'branch':'accepted'}}))
        self.assertEqual(self.drive().returncode, 65)
        self.assertIn('missing commit: accepted', (self.lane/'DRIVER-FAILED.md').read_text())
        git('commit','--allow-empty','-m','acceptance')
        self.assertEqual(self.drive().returncode, 0)

    def test_actual_launcher_wraps_fake_zero_exit_without_model_calls(self):
        campaign = self.root/'campaign'; (campaign/'briefs').mkdir(parents=True)
        lane=campaign/'worker'; lane.mkdir()
        (campaign/'briefs/worker.md').write_text('fixture'); (lane/'EXPECTS').write_text('{"files":["DONE.md"]}')
        binary=self.root/'bin'; binary.mkdir()
        for name in ('codex','caffeinate'):
            path=binary/name
            path.write_text('#!/bin/bash\n' + ('exit 0\n' if name=='codex' else 'shift; exec "$@"\n')); path.chmod(0o755)
        result=subprocess.run(['bash',str(SCRIPTS/'launch.sh'),'worker'],capture_output=True,text=True,
                              env={**os.environ,'PATH':str(binary)+':'+os.environ['PATH'], 'CAMPAIGN_DIR':str(campaign),
                                   'REPO':str(self.root), 'ALLOW_UNWATCHED':'1'})
        self.assertEqual(result.returncode,0,result.stderr)
        deadline=time.time()+5
        while not (lane/'SESSION-EXIT').exists() and time.time()<deadline: time.sleep(.02)
        self.assertEqual((lane/'SESSION-EXIT').read_text().strip(),'65')
        self.assertIn('DONE.md',(lane/'DRIVER-FAILED.md').read_text())

    def test_removed_checkout_requires_proven_completed_integration(self):
        env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        def git(*args):
            return subprocess.check_output(['git','-C',str(self.root),*args],env=env,text=True).strip()
        git('init','-b','main');git('config','user.name','Fixture');git('config','user.email','f@example.org')
        git('commit','--allow-empty','-m','base');base=git('rev-parse','HEAD')
        git('checkout','-b','accepted');git('commit','--allow-empty','-m','result')
        accepted=git('rev-parse','HEAD');git('checkout','main');git('merge','--ff-only','accepted')
        released = self.root/'removed-worktree'
        registry=self.root/'registry';registry.mkdir()
        row=dict(path=str(released),branch='accepted',shared_root=str(self.root),released=True,
                 removed_utc='2026-11-01',integrated_commit=accepted)
        marker=registry/'lane.json';marker.write_text(json.dumps(row))
        (self.lane/'DONE.md').write_text('complete')
        (self.lane/'EXPECTS').write_text(json.dumps(dict(files=['DONE.md'],commit=dict(repo=str(released),base=base,branch='accepted'))))
        with mock.patch.object(deliverables,'LANE_REGISTRY',registry):
            self.assertEqual(deliverables.missing(self.lane),[])
            # Production registry contains an empty unrelated file and legacy
            # list-shaped records. Neither may poison a valid released owner.
            (registry/'000-empty.json').write_text('')
            (registry/'001-legacy.json').write_text('[]')
            self.assertEqual(deliverables.missing(self.lane),[])
            for field,value in [('released',False),('removed_utc',None),('integrated_commit',base),('branch','other')]:
                marker.write_text(json.dumps({**row,field:value}))
                self.assertTrue(deliverables.missing(self.lane),field)
                self.assertNotIn('invalid EXPECTS', str(deliverables.missing(self.lane)))
            marker.write_text(json.dumps(row))
            duplicate=registry/'duplicate.json';duplicate.write_text(json.dumps(row))
            self.assertIn('found 2', str(deliverables.missing(self.lane)))
            duplicate.unlink()
            # A malformed matching registry supplies no proof, even though
            # another branch exists in the shared repository.
            marker.write_text('')
            self.assertIn(str(marker), str(deliverables.missing(self.lane)))
            marker.unlink()
            self.assertTrue(deliverables.missing(self.lane))

    def test_session_handoff_defers_contract_to_live_nested_driver(self):
        child=self.lane/'processing-driver'; child.mkdir()
        command=f'bash drive-loop.sh --once {child} fixture 30 -- true'
        process=subprocess.Popen(['bash','-c','exec -a "$1" sleep 30','fixture',command],cwd=self.root)
        self.addCleanup(lambda: (process.terminate(), process.wait(timeout=5)) if process.poll() is None else None)
        (child/'driver.pid').write_text(str(process.pid))
        (child/'driver.cmd').write_text(json.dumps({'cwd':str(self.root),'args':['--once',str(child),'fixture','30','--','true']}))
        (self.lane/'EXPECTS').write_text('{"files":["final.json"]}')
        result=subprocess.run([sys.executable,str(SCRIPTS/'deliverables.py'),'run',str(self.lane),'--','true'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((self.lane/'DRIVER-FAILED.md').exists())
        # The outer drive-loop must honor the same actual nested ownership,
        # without treating its own wrapper as proof of continuation.
        self.assertEqual(self.drive().returncode, 0)
        (self.lane/'DONE.md').write_text('premature milestone')
        self.assertEqual(self.drive().returncode, 65)


    def test_session_wrapper_forwards_stop_to_its_worker(self):
        child_pid=self.root/'child.pid'
        code="import os,pathlib,time; pathlib.Path("+repr(str(child_pid))+").write_text(str(os.getpid())); time.sleep(30)"
        process=subprocess.Popen([sys.executable,str(SCRIPTS/'deliverables.py'),'run',str(self.lane),'--',sys.executable,'-c',code])
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline=time.time()+5
        while not child_pid.exists() and time.time()<deadline: time.sleep(.02)
        self.assertTrue(child_pid.exists())
        pid=int(child_pid.read_text())
        self.addCleanup(lambda: self.kill_if_live(pid))
        process.terminate(); self.assertEqual(process.wait(timeout=5),143)
        with self.assertRaises(ProcessLookupError): os.kill(pid,0)

    @staticmethod
    def kill_if_live(pid):
        try: os.kill(pid,15)
        except ProcessLookupError: pass


    def test_audit_under_bash_reexecs_zsh(self):
        subprocess.run(['git','init','-b','main',str(self.root)],capture_output=True,check=True)
        result=subprocess.run(['bash',str(SCRIPTS/'lane-audit.sh'),str(self.root),str(self.lane)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('setopt: command not found',result.stderr)


class KeeperRepeatsTest(unittest.TestCase):
    setUp = fixtures.RetireTest.setUp
    lane = fixtures.FrictionTest.lane

    @mock.patch.object(lk, "owns_process", return_value=True)
    def test_nested_live_driver_prevents_short_session_and_uses_nested_progress(self, owner):
        C,d=self.lane(); nested=d/'processing-driver'; nested.mkdir()
        (d/'codex.log').write_text('discovery ended'); (d/'HANDOFF.md').write_text('next: process queue')
        (d/'fresh-at').write_text(str(self.t-20)); (d/'session-at').write_text(str(self.t-10))
        (nested/'driver.pid').write_text(str(os.getpid())); (nested/'driver.cmd').write_text(json.dumps({'cwd':str(self.tmp),'args':[str(nested),'fixture','30','--','true']}))
        for p in d.iterdir():
            if p.is_file(): os.utime(p,(self.t-20000,self.t-20000))
        with mock.patch.object(self.K,'waiting_heavy',return_value=False), mock.patch.object(self.K,'run') as run:
            self.K.tend(C,d)
        run.assert_not_called(); self.assertFalse(self.K.st['events'])
        self.K.cfg['campaigns']=[str(C)]; self.assertEqual(self.K.running_lanes(),[d])
        with mock.patch.object(self.K,'run'):
            self.K.launch(C,d,'fixture')
        self.assertNotIn('SHORT-SESSION',[e['kind'] for e in self.K.st['events'].values()])

    @mock.patch.object(lk, "owns_process", return_value=True)
    def test_reports_use_nested_liveness_progress_and_failure(self, owner):
        import dashboard
        import report
        C,d=self.lane(); child=d/'processing-driver'; child.mkdir()
        (child/'driver.pid').write_text(str(os.getpid())); (child/'driver.log').write_text('admission 20/70\n')
        self.K.cfg['campaigns']=[str(C)]
        with mock.patch.dict(sys.modules,{type(self.K).__module__:lk}):
            self.assertIn('driver',report.lanes(self.K)[0]['state'])
        with mock.patch.object(dashboard,'owns_process',return_value=True):
            row=dashboard.scan([C])[0]
        self.assertTrue(row['driver']); self.assertEqual(row['progress'],'admission 20/70')
        (child/'driver.pid').write_text('99999998'); (child/'DRIVER-FAILED.md').write_text('missing intake.jsonl')
        (d/'DONE.md').write_text('failed')
        with mock.patch.dict(sys.modules,{type(self.K).__module__:lk}):
            self.assertEqual(report.lanes(self.K)[0]['state'],'DRIVER-FAILED')
        self.assertEqual(dashboard.scan([C])[0]['state'],'DRIVER-FAILED')


    def test_next_step_handoff_without_live_driver_is_reported(self):
        C,d=self.lane(); (d/'HANDOFF.md').write_text('Next I will implement admissions.')
        with mock.patch.object(self.K,'launch'): self.K.tend(C,d)
        self.assertIn('HANDOFF-IDLE',[e['kind'] for e in self.K.st['events'].values()])

    def test_missing_contract_blocks_done_and_reports_named_item(self):
        C,d=self.lane(); (d/'DONE.md').write_text('failed staging'); (d/'READY-1.md').write_text('not accepted'); (d/'EXPECTS').write_text('{"files":["accepted.json"]}')
        self.K.tend(C,d); events=list(self.K.st['events'].values())
        self.assertEqual([e['kind'] for e in events],['DRIVER-FAILED']); self.assertIn('accepted.json',events[0]['detail'])

    def test_nested_interrupted_driver_restarts_even_with_pending_contract(self):
        C,d=self.lane(); child=d/'processing-driver'; child.mkdir()
        (child/'driver.pid').write_text('99999998')
        args=['--once',str(child),'fixture','30','--','true']
        (child/'driver.cmd').write_text(json.dumps({'args':args,'cwd':str(self.tmp)}))
        (d/'EXPECTS').write_text('{"files":["final.json"]}'); (d/'SESSION-EXIT').write_text('0')
        with mock.patch.object(lk.subprocess,'run',return_value=mock.Mock(returncode=1)), mock.patch.object(self.K,'run') as run:
            self.K.tend(C,d)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][-len(args):],args)
        self.assertEqual(run.call_args.args[2],child/'driver.log')


    def test_nested_failed_driver_reports_root_lane_and_actual_failure(self):
        C,d=self.lane(); child=d/'processing-driver'; child.mkdir()
        (child/'driver.pid').write_text('99999998'); (child/'DRIVER-FAILED.md').write_text('missing file: admissions.jsonl')
        with mock.patch.object(self.K,'resume') as resume: self.K.tend(C,d)
        event=next(iter(self.K.st['events'].values()))
        self.assertEqual(event['lane'],str(d)); self.assertEqual(event['kind'],'DRIVER-FAILED')
        self.assertIn('admissions.jsonl',event['detail']); resume.assert_called_once()

    def test_nested_driver_only_exit_is_a_root_lane_event(self):
        C,d=self.lane(); (d/'pid').unlink(); child=d/'processing-driver'; child.mkdir()
        (child/'driver.pid').write_text('99999998'); (child/'DRIVER-EXIT').write_text('0')
        self.K.cfg['campaigns']=[str(C)]; self.assertEqual(list(self.K.lanes())[0][1],[d])
        self.K.tend(C,d); self.assertEqual(next(iter(self.K.st['events'].values()))['kind'],'JOB-FINISHED')


class NestedRunawayTest(unittest.TestCase):
    setUp=fixtures.RunawayTest.setUp
    keeper=fixtures.RunawayTest.keeper
    proc=fixtures.RunawayTest.proc
    scan=fixtures.RunawayTest.scan
    kinds=fixtures.RunawayTest.kinds

    def test_nested_live_wrapper_and_children_are_never_reaped_as_orphans(self):
        child=self.d/'processing-driver'; child.mkdir()
        (self.d/'driver.pid').write_text('99999999'); (child/'driver.pid').write_text('500'); (child/'driver.cmd').write_text('{}')
        self.proc(500,f'bash drive-loop.sh {child}'); self.proc(501,'python3 process_queue.py',ppid=500)
        K=self.keeper(reap_orphans=True); self.scan(K)
        self.assertNotIn('ORPHAN',self.kinds(K)); self.assertEqual(self.killed,[])


if __name__=='__main__': unittest.main()
