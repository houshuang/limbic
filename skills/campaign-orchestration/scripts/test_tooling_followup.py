"""Refusing regressions for aliased ownership and forgotten parked lanes; offline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS = Path(os.environ.get('TOOLING_SCRIPTS', Path(__file__).resolve().parent))
sys.path.insert(0, str(SCRIPTS))
import deliverables
import test_lane_keeper as fixtures
lk = fixtures.lk


def marker(folder, owner=None, cwd=None, pid=99999998):
    (folder/'driver.pid').write_text(str(pid))
    (folder/'driver.cmd').write_text(json.dumps({'cwd':str(cwd or folder),
        'args':['--once',str(owner or folder),'fixture','30','--','true']}))


class OwnershipTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.tmp=Path(temp.name).resolve()
        self.source=self.tmp/'intake'; self.source.mkdir()
        self.target=self.tmp/'stopped'; self.target.mkdir()
        marker(self.target)
        (self.target/'STOP').write_text('preserve stopped evidence')
        self.safe=self.tmp/'safe'; self.safe.write_text('#!/bin/bash\nshift 5; exec "$@"\n');self.safe.chmod(0o755)

    def run_drive(self):
        return subprocess.run(['bash',str(SCRIPTS/'drive-loop.sh'),'--once',str(self.source),'fixture','30','--','true'],
                              env={**os.environ,'SLEEPSAFE':str(self.safe)},capture_output=True,text=True)

    def test_alias_refuses_before_pid_or_command_write_preserving_target(self):
        before={n:(self.target/n).read_bytes() for n in ('driver.pid','driver.cmd')}
        for name in before: (self.source/name).symlink_to(self.target/name)
        result=self.run_drive()
        self.assertEqual(result.returncode,65,result.stdout+result.stderr)
        self.assertIn('aliased driver marker',result.stderr)
        for name,raw in before.items():self.assertEqual((self.target/name).read_bytes(),raw)
        self.assertFalse((self.source/'DRIVER-EXIT').exists())

    def test_hardlinked_markers_are_also_refused_without_truncating_evidence(self):
        before=(self.target/'driver.pid').read_bytes()
        os.link(self.target/'driver.pid',self.source/'driver.pid')
        result=self.run_drive()
        self.assertEqual(result.returncode,65,result.stderr)
        self.assertEqual((self.target/'driver.pid').read_bytes(),before)

    def test_process_cwd_and_recorded_owner_must_both_agree(self):
        from driver_state import owns_process
        marker(self.source)
        command=f'bash drive-loop.sh --once {self.source} fixture 30 -- true'
        self.assertTrue(owns_process(self.source,123,command,cwd=str(self.source)))
        self.assertFalse(owns_process(self.source,123,'codex exec --prompt '+command,cwd=str(self.source)))
        self.assertFalse(owns_process(self.source,123,command,cwd=str(self.target)))
        self.assertFalse(owns_process(self.source,123,command.replace(str(self.source),str(self.target)),cwd=str(self.source)))

    def test_copied_foreign_owner_refuses_before_start(self):
        marker(self.source,owner=self.target)
        result=self.run_drive()
        self.assertEqual(result.returncode,65,result.stdout+result.stderr)
        self.assertIn('state owner',result.stderr)

    def test_nested_discovery_and_handoff_reject_aliases(self):
        nested=self.source/'processing-driver';nested.mkdir()
        for name in ('driver.pid','driver.cmd'):(nested/name).symlink_to(self.target/name)
        self.assertNotIn(nested,deliverables.driver_dirs(self.source))
        (self.source/'driver.pid').symlink_to(self.target/'driver.pid')
        (self.source/'driver.cmd').symlink_to(self.target/'driver.cmd')
        self.assertFalse(deliverables.live_driver(self.source))

    def test_stopped_alias_cannot_reap_live_foreign_driver_or_its_child(self):
        self.safe.write_text('#!/bin/bash\nshift 5; exec "$@"\n')
        cmd=['bash',str(SCRIPTS/'drive-loop.sh'),'--once',str(self.source),'fixture','30','--',sys.executable,'-c','import time; time.sleep(30)']
        process=subprocess.Popen(cmd,env={**os.environ,'SLEEPSAFE':str(self.safe)},start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        def cleanup():
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        self.addCleanup(cleanup)
        deadline=time.time()+3
        while not (self.source/'driver.cmd').exists() and time.time()<deadline:time.sleep(.01)
        self.assertIsNone(process.poll())
        for name in ('driver.pid','driver.cmd'):
            (self.target/name).unlink();(self.target/name).symlink_to(self.source/name)
        os.utime(self.target/'STOP',(time.time()-900,time.time()-900))
        cfg=self.tmp/'config.json';cfg.write_text(json.dumps({'repo':str(self.tmp),'state':str(self.tmp/'keeper-state'),'campaigns':[str(self.tmp)],'reap_orphans':True}))
        keeper=lk.Keeper(cfg)
        row={'cmd':' '.join(cmd),'ppid':1,'age':1000,'rss':1000}
        table={process.pid:row,process.pid+100000:{'cmd':'python child','ppid':process.pid,'age':999,'rss':1000}}
        with mock.patch.object(lk,'ps_table',return_value=table),mock.patch.object(lk,'proc_cwds',return_value={}),mock.patch.object(keeper,'reap') as reap:
            keeper.runaways()
        reap.assert_not_called();self.assertIsNone(process.poll())



class RemoteScanTest(unittest.TestCase):
    def test_stdin_remote_scan_in_isolated_python_has_shared_owner_helper(self):
        import dashboard
        with tempfile.TemporaryDirectory() as temp:
            lane=Path(temp)/'lane';lane.mkdir();marker(lane)
            result=subprocess.run([sys.executable,'-I','-','scan','--dirs',temp],
                input=dashboard.remote_source(),capture_output=True,cwd=temp)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            report=json.loads(result.stdout)
            self.assertEqual(report['lanes'][0]['lane'],'lane')


class ForeignDescendantTest(unittest.TestCase):
    setUp=fixtures.RunawayTest.setUp
    keeper=fixtures.RunawayTest.keeper
    proc=fixtures.RunawayTest.proc
    scan=fixtures.RunawayTest.scan

    def test_stop_preserves_independently_owned_descendant_lane(self):
        other=self.C/'other';other.mkdir()
        marker(self.d,cwd=self.repo,pid=700);marker(other,cwd=self.repo,pid=800)
        self.proc(700,f'bash drive-loop.sh --once {self.d} fixture 30 -- true')
        self.proc(800,f'bash drive-loop.sh --once {other} fixture 30 -- true',ppid=700)
        self.proc(801,'python source_recovery.py',ppid=800)
        self.cwd.update({700:str(self.repo),800:str(self.repo)})
        (self.d/'STOP').write_text('stop only this lane');os.utime(self.d/'STOP',(time.time()-900,time.time()-900))
        with mock.patch.object(lk,'alive',side_effect=lambda f,pid=None:f.name=='driver.pid'):
            self.scan(self.keeper(reap_orphans=True))
        self.assertEqual(self.killed,[[700]])


class ParkedTest(unittest.TestCase):
    setUp=fixtures.RetireTest.setUp
    lane=fixtures.FrictionTest.lane

    def test_old_stop_is_actionable_once_per_24h_even_after_ack_and_reload(self):
        C,d=self.lane();(d/'STOP').write_text('wait for release')
        os.utime(d/'STOP',(self.t-90000,self.t-90000))
        with mock.patch.object(lk,'now',return_value=self.t):self.K.tend(C,d)
        events=list(self.K.st['events'].values());self.assertIn('PARKED-STALE',[e['kind'] for e in events])
        self.assertNotIn('PARKED-STALE',lk.INFO_KINDS)
        self.K.st['events'].clear();self.K.save();self.K=lk.Keeper(self.K.cfg_path)
        with mock.patch.object(lk,'now',return_value=self.t+3600):self.K.tend(C,d)
        self.assertFalse(self.K.st['events'])
        with mock.patch.object(lk,'now',return_value=self.t+86401):self.K.tend(C,d)
        self.assertIn('PARKED-STALE',[e['kind'] for e in self.K.st['events'].values()])

    def test_old_idle_handoff_reports_but_retired_and_recent_do_not(self):
        C,d=self.lane();(d/'HANDOFF.md').write_text('wait for integration')
        os.utime(d/'HANDOFF.md',(self.t-90000,self.t-90000))
        with mock.patch.object(self.K,'launch') as launch:self.K.tend(C,d)
        launch.assert_not_called();self.assertIn('PARKED-STALE',[e['kind'] for e in self.K.st['events'].values()])
        self.K.st['events'].clear();(d/'RETIRED').write_text('superseded')
        self.K.tend(C,d);self.assertNotIn('PARKED-STALE',[e['kind'] for e in self.K.st['events'].values()])


class RetirementTest(unittest.TestCase):
    setUp=fixtures.RetireTest.setUp
    lane=fixtures.FrictionTest.lane

    def retired_history(self):
        C,d=self.lane()
        marker(d,cwd=self.tmp/'removed')
        (d/'DRIVER-FAILED.md').write_text('historical failed method')
        (d/'EXPECTS').write_text('{"files":["never-accepted.json"]}')
        (d/'HANDOFF.md').write_text('old continuation')
        for p in d.iterdir():os.utime(p,(self.t-100,self.t-100))
        (d/'RETIRED').write_text('Reviewed: superseded by integrated successor; evidence retained')
        return C,d

    def test_retired_removed_cwd_and_failed_contract_do_not_reemit_owner(self):
        C,d=self.retired_history()
        with mock.patch.object(self.K,'run') as run:self.K.tend(C,d)
        self.assertFalse(self.K.st['events']);run.assert_not_called()
        # Retirement classifies history only; direct launch ownership remains refused.
        from driver_state import valid
        self.assertFalse(valid(d))
        self.assertTrue(deliverables.missing(d))

    def test_unretired_missing_cwd_is_still_actionable(self):
        C,d=self.retired_history();(d/'RETIRED').unlink()
        self.K.tend(C,d)
        self.assertIn('DRIVER-OWNER',[e['kind'] for e in self.K.st['events'].values()])

    def test_newer_unfinished_execution_cannot_hide_behind_old_retirement(self):
        C,d=self.retired_history()
        os.utime(d/'driver.cmd',(self.t+10,self.t+10))
        self.K.tend(C,d)
        self.assertIn('DRIVER-OWNER',[e['kind'] for e in self.K.st['events'].values()])

    def test_independently_owned_live_child_is_not_hidden_or_restarted(self):
        C,d=self.retired_history();child=d/'processing-driver';child.mkdir()
        marker(child,cwd=self.tmp,pid=os.getpid())
        with mock.patch.object(lk,'owns_process',return_value=True), mock.patch.object(self.K,'waiting_heavy',return_value=False), mock.patch.object(self.K,'run') as run:
            self.K.tend(C,d)
        self.assertFalse(self.K.st['events']);run.assert_not_called()
        self.K.cfg['campaigns']=[str(C)]
        with mock.patch.object(lk,'owns_process',return_value=True):self.assertEqual(self.K.running_lanes(),[d])


class ParkedLaunchTest(unittest.TestCase):
    def test_stale_registry_refused_before_mutating_handoff_or_spawning(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lane=root/'campaign/worker';lane.mkdir(parents=True)
            registry=root/'registry';registry.mkdir()
            (registry/'worker.json').write_text(json.dumps({'path':str(root/'removed'),'branch':'sh/worker','released':False}))
            (lane/'HANDOFF.md').write_text('continue at removed checkout')
            sentinel=root/'spawned'
            result=subprocess.run([sys.executable,str(SCRIPTS/'session_guard.py'),str(lane),'--',sys.executable,'-c',f'open({str(sentinel)!r},"w").close()'],
                env={**os.environ,'LANE_REGISTRY_DIR':str(registry)},capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('removed',result.stdout+result.stderr)
            self.assertFalse(sentinel.exists());self.assertFalse((lane/'session-at').exists())
            self.assertEqual((lane/'HANDOFF.md').read_text(),'continue at removed checkout')


if __name__=='__main__':unittest.main()
