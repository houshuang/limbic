"""Regressions for 1 Nov 12:28, 17:15, 21:25 and 21:30 incidents."""
import datetime, json, os, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock
import test_lane_keeper as fixture
import deliverables
import checkpoint_queue
from driver_state import owns_process
S = Path(__file__).resolve().parent

class Incidents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name).resolve(); self.safe = self.d/'safe'
        self.safe.write_text('#!/bin/bash\nshift 5; exec "$@"\n'); self.safe.chmod(0o755)
    def spawn(self):
        cmd=['bash',str(S/'drive-loop.sh'),'--once',str(self.d),'incident','30','--',sys.executable,'-c','import time; time.sleep(30)']
        p=subprocess.Popen(cmd,env={**os.environ,'SLEEPSAFE':str(self.safe)},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        def cleanup():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=3)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
        self.addCleanup(cleanup)
        deadline=time.monotonic()+5
        while (not (self.d/'driver.cmd').exists() or not owns_process(self.d, str(p.pid))) and p.poll() is None and time.monotonic()<deadline: time.sleep(.01)
        self.assertTrue((self.d/'driver.cmd').exists())
        return p,cmd
    def test_real_driver_and_duplicate_refusal(self):
        p,cmd=self.spawn()
        pid=(self.d/'driver.pid').read_text(); self.assertTrue(pid.endswith('\n'))
        self.assertTrue(owns_process(self.d,pid))
        second=subprocess.run(cmd,env={**os.environ,'SLEEPSAFE':str(self.safe)},capture_output=True,text=True,timeout=5)
        self.assertEqual(second.returncode,65,second.stderr)
        self.assertEqual(int((self.d/'driver.pid').read_text()),p.pid)
        self.assertFalse((self.d/'DRIVER-FAILED.md').exists())
    def test_flag_required_for_continuation(self):
        (self.d/'EXPECTS').write_text(json.dumps({'files':['missing']}))
        with mock.patch.object(deliverables,'live_driver',return_value=True):
            with mock.patch.object(sys,'argv',['deliverables.py','check',str(self.d)]): self.assertEqual(deliverables.main(),65)
            with mock.patch.object(sys,'argv',['deliverables.py','check',str(self.d),'--allow-continuation']): self.assertEqual(deliverables.main(),0)
    def test_lease_limit_and_renewal(self):
        cfg=self.d/'cfg'; cfg.write_text(json.dumps({'repo':str(self.d),'state':str(self.d/'state')})); k=fixture.lk.Keeper(cfg)
        with self.assertRaises(ValueError): k.set_lease('lane',61)
        k.set_lease('lane',60); self.assertIsNotNone(k.lease())

    def test_driver_owner_deduplicated_after_ack_and_restart(self):
        cfg=self.d/'cfg'; cfg.write_text(json.dumps({'repo':str(self.d),'state':str(self.d/'state')})); k=fixture.lk.Keeper(cfg)
        d=self.d/'recall'; d.mkdir(); (d/'driver.cmd').symlink_to(self.d/'historic')
        (self.d/'historic').write_text('{}')
        now=time.time()
        with mock.patch.object(fixture.lk,'now',return_value=now): k.tend_driver(self.d,d,d)
        self.assertEqual([e['kind'] for e in k.st['events'].values()],['DRIVER-OWNER'])
        k.st['events'].clear(); k.save(); k=fixture.lk.Keeper(cfg)
        with mock.patch.object(fixture.lk,'now',return_value=now+3600): k.tend_driver(self.d,d,d)
        self.assertFalse(k.st['events'])
        with mock.patch.object(fixture.lk,'now',return_value=now+86401): k.tend_driver(self.d,d,d)
        self.assertEqual([e['kind'] for e in k.st['events'].values()],['DRIVER-OWNER'])

    def test_hourly_token_limit_notifies_and_expires(self):
        cfg=self.d/'cfg'; cfg.write_text(json.dumps({'repo':str(self.d),'state':str(self.d/'state'),
            'notify':{'cmd':'unused'},'coordinator':{'engine':'codex','max_tokens_per_hour':100,'min_gap_minutes':0}}))
        k=fixture.lk.Keeper(cfg,dry=True); now=time.time()
        ts=datetime.datetime.fromtimestamp(now-1,datetime.timezone.utc).isoformat()
        log=k.k/'coordinator-test.log'; log.write_text(json.dumps({'type':'turn.completed','timestamp':ts,'usage':{'input_tokens':90,'output_tokens':10}})+'\n')
        k.st['events']={'ready':{'lane':str(self.d),'kind':'READY','detail':'checkpoint'}}
        with mock.patch.object(fixture.lk,'now',return_value=now), mock.patch.object(k,'coordinator_live',return_value=None): k.maybe_start_coordinator()
        self.assertEqual(k.st['coordinator'].get('starts',0),0)
        self.assertIn('COORDINATOR-RATE-LIMIT',[e['kind'] for e in k.st['events'].values()])
        self.assertTrue(k.st['notify']['queue'])
        with mock.patch.object(fixture.lk,'now',return_value=now+3601): self.assertEqual(k.coordinator_hour_tokens(),0)

    def test_queue_parks_failure_and_runs_next_once(self):
        heavy=self.d/'heavy'; heavy.write_text('#!/bin/bash\nexec "$@"\n'); heavy.chmod(0o755)
        manifest=self.d/'manifest.json'; state=self.d/'queue'
        first=self.d/'first'; first.mkdir(); second=self.d/'other'; second.mkdir(); lease=self.d/'lease'; lease.write_text(json.dumps({'owner':'executor','until':time.time()+300}))
        manifest.write_text(json.dumps([{'id':'bad','cwd':str(first),'argv':[sys.executable,'-c','raise SystemExit(7)']},
            {'id':'good','cwd':str(second),'argv':[sys.executable,'-c',"open('ran','a').write('x')"]}]))
        self.assertEqual(checkpoint_queue.run(manifest,state,heavy,lease,'executor'),1)
        outcomes=json.loads((state/'results.json').read_text())['items']
        self.assertEqual(outcomes['bad']['exit'],7); self.assertEqual(outcomes['good']['status'],'completed')
        self.assertEqual(checkpoint_queue.run(manifest,state,heavy,lease,'executor'),1)
        self.assertEqual((second/'ran').read_text(),'x')
        manifest.write_text('[]')
        with self.assertRaises(ValueError): checkpoint_queue.run(manifest,state,heavy,lease,'executor')

    def test_interrupted_queue_waits_for_surviving_child(self):
        heavy=self.d/'heavy'; heavy.write_text('#!/bin/bash\nexec "$@"\n'); heavy.chmod(0o755)
        work=self.d/'work'; work.mkdir(); state=self.d/'queue'; lease=self.d/'lease'
        lease.write_text(json.dumps({'owner':'executor','until':time.time()+300}))
        manifest=self.d/'manifest.json'
        manifest.write_text(json.dumps([{'id':'one','cwd':str(work),'argv':[sys.executable,'-c',
            "import time,pathlib; pathlib.Path('started').touch(); time.sleep(2)"]}]))
        command=[sys.executable,str(S/'checkpoint_queue.py'),str(manifest),str(state),'--heavy',str(heavy),'--lease',str(lease),'--owner','executor']
        p=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: p.kill() if p.poll() is None else None)
        deadline=time.monotonic()+3
        while not (work/'started').exists() and time.monotonic()<deadline: time.sleep(.01)
        self.assertTrue((work/'started').exists())
        p.kill(); p.wait()
        with self.assertRaises(BlockingIOError): checkpoint_queue.run(manifest,state,heavy,lease,'executor')
        time.sleep(2.1)
        self.assertEqual(checkpoint_queue.run(manifest,state,heavy,lease,'executor'),1)
        self.assertEqual(json.loads((state/'results.json').read_text())['items']['one']['status'],'parked')

    def test_queue_refuses_shared_workspace_and_expired_lease(self):
        heavy=self.d/'heavy'; heavy.write_text('#!/bin/bash\nexec "$@"\n'); heavy.chmod(0o755)
        lease=self.d/'lease'; lease.write_text(json.dumps({'owner':'other','until':time.time()+10}))
        manifest=self.d/'manifest'; item={'id':'one','cwd':str(self.d),'argv':['true']}
        manifest.write_text(json.dumps([item,{**item,'id':'two'}]))
        with self.assertRaises(ValueError): checkpoint_queue.run(manifest,self.d/'s',heavy,lease,'executor')
        manifest.write_text(json.dumps([item]))
        self.assertEqual(checkpoint_queue.run(manifest,self.d/'s',heavy,lease,'executor'),1)
        self.assertIn('lease',json.loads((self.d/'s/results.json').read_text())['items']['one']['reason'])

    def test_live_nested_driver_prevents_short_session_and_launch(self):
        child=self.d/'processing-driver'; child.mkdir()
        original=self.d; self.d=child
        p,_=self.spawn(); self.d=original
        (self.d/'driver.pid').symlink_to(child/'driver.pid')
        (self.d/'driver.cmd').symlink_to(child/'driver.cmd')
        (self.d/'codex.log').write_text('short session')
        (self.d/'session-at').write_text(str(time.time()-10))
        (self.d/'fresh-at').write_text(str(time.time()-15))
        cfg=self.d/'cfg'; cfg.write_text(json.dumps({'repo':str(self.d),'state':str(self.d/'state')}))
        k=fixture.lk.Keeper(cfg)
        with mock.patch.object(k,'run') as run:
            k.launch(self.d,self.d,'handoff')
        run.assert_not_called(); self.assertFalse(k.st['events']); self.assertIsNone(p.poll())

    def test_lease_rechecked_after_heavy_admission(self):
        lease=self.d/'lease'; lease.write_text(json.dumps({'owner':'executor','until':time.time()+300}))
        heavy=self.d/'heavy'; heavy.write_text('#!/bin/bash\nprintf "{}" > "$LEASE_FIXTURE"\nexec "$@"\n'); heavy.chmod(0o755)
        manifest=self.d/'manifest'; manifest.write_text(json.dumps([{'id':'one','cwd':str(self.d),'argv':[sys.executable,'-c',"open('mutated','w').write('bad')"]}]))
        with mock.patch.dict(os.environ,{'LEASE_FIXTURE':str(lease)}):
            self.assertEqual(checkpoint_queue.run(manifest,self.d/'queue',heavy,lease,'executor'),1)
        self.assertFalse((self.d/'mutated').exists())
        result=json.loads((self.d/'queue/results.json').read_text())['items']['one']
        self.assertEqual(result['exit'],75)

if __name__ == "__main__": unittest.main()
