"""Subscription-free integration tests using a fake Codex JSONL executable."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPTS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('keeper_for_codex',SCRIPTS/'lane-keeper.py')
lk=importlib.util.module_from_spec(spec);spec.loader.exec_module(lk)

FAKE='''#!PYTHON
import json,os,sys
from pathlib import Path
Path('child-proof.json').write_text(json.dumps({'stdin':sys.stdin.read(), 'own_session':os.getsid(0)==os.getpid(), 'has_api_key':any(key in os.environ for key in ('OPENAI_API_KEY','CODEX_API_KEY','ANTHROPIC_API_KEY')), 'argv':sys.argv[1:]}))
mode=os.environ.get('FAKE_MODE','ok')
if mode!='no-session': print(json.dumps({'type':'thread.started','thread_id':os.environ['FAKE_SID']}),flush=True)
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'handoff'}}),flush=True)
if mode=='wait':
 import subprocess,time
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'])
 Path('grandchild.pid').write_text(str(child.pid))
 time.sleep(300)
print(json.dumps({'type':'turn.failed' if mode=='failed' else 'turn.completed'}),flush=True)
sys.exit(23 if mode=='nonzero' else 0)
'''


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.keeper=self.root/'state/keeper';self.keeper.mkdir(parents=True)
        self.registry=self.root/'registry';self.registry.mkdir()
        self.repo=self.root/'repo';self.repo.mkdir()
        self.prompt=self.root/'prompt.txt';self.prompt.write_text('write handoff and retire')
        self.out=self.root/'run.log';self.session=self.root/'run.session'
        self.sid='019a0011-1234-5678-9abc-012345678901'
        self.reservation='pending-reservation-1'
        (self.keeper/'coordinator.session').write_text(self.reservation+'\n')
        binary=self.root/'bin';binary.mkdir();codex=binary/'codex'
        codex.write_text(FAKE.replace('PYTHON',sys.executable));codex.chmod(0o755)
        self.env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH'], 'FAKE_SID':self.sid,
                  'OPENAI_API_KEY':'fixture', 'CODEX_API_KEY':'fixture','ANTHROPIC_API_KEY':'fixture'}

    def command(self):
        return [sys.executable,str(SCRIPTS/'codex-coordinator.py'),'--repo',str(self.repo),'--keeper',str(self.keeper),
              '--registry',str(self.registry),'--prompt',str(self.prompt),'--out',str(self.out),
              '--session-file',str(self.session),'--reservation',self.reservation]

    def invoke(self, mode='ok'):
        with self.out.open('w') as log:
            result=subprocess.run(self.command(),env={**self.env,'FAKE_MODE':mode},stdout=log,stderr=subprocess.STDOUT,timeout=10)
        return result.returncode

    def test_records_actual_session_subscription_environment_and_work_liveness(self):
        self.assertEqual(self.invoke(),0,self.out.read_text())
        self.assertEqual(self.session.read_text().strip(),self.sid)
        self.assertEqual((self.keeper/'coordinator.session').read_text().strip(),self.sid)
        registration=json.loads((self.registry/(self.sid+'.json')).read_text())
        self.assertEqual(registration,{'repo':str(self.repo),'engine':'codex','transcript':str(self.out)})
        proof=json.loads((self.repo/'child-proof.json').read_text())
        self.assertEqual(proof['stdin'],'');self.assertTrue(proof['own_session']);self.assertFalse(proof['has_api_key'])
        self.assertIn('forced_login_method="chatgpt"',proof['argv'])
        self.assertGreater(lk.transcript_activity(self.out),time.time()-10)
        self.assertIn('COORDINATOR-SESSION '+self.sid,(self.keeper/'keeper.log').read_text())

    def test_missing_header_failed_turn_and_nonzero_exit_are_failures(self):
        for mode,expected in [('no-session',2),('failed',2),('nonzero',23)]:
            with self.subTest(mode=mode):
                case=AdapterTests();case.setUp()
                try:self.assertEqual(case.invoke(mode),expected,case.out.read_text())
                finally:case.doCleanups()

    def test_retired_or_replaced_reservation_never_launches(self):
        (self.keeper/'retired').write_text('2026-10-31 23:00:00 '+self.reservation+'\n')
        self.assertEqual(self.invoke(),2)
        self.assertFalse((self.repo/'child-proof.json').exists())
        (self.keeper/'retired').unlink();(self.keeper/'coordinator.session').write_text('successor\n')
        self.assertEqual(self.invoke(),2)
        self.assertFalse((self.repo/'child-proof.json').exists())

    def test_retirement_removes_codex_registration_and_cannot_revive(self):
        self.assertEqual(self.invoke(),0,self.out.read_text())
        config=self.root/'coordinator.json';config.write_text(json.dumps({'repo':str(self.repo),'state':str(self.root/'state'),'campaigns':[]}))
        keeper=lk.Keeper(config)
        from unittest.mock import patch
        with patch.object(lk,'REG',self.registry):
            self.assertIsNotNone(keeper.coordinator_live())
            keeper.retire(self.sid)
            self.assertFalse((self.registry/(self.sid+'.json')).exists())
            # Late output from a retired stream cannot re-register a live owner.
            self.assertIsNone(keeper.coordinator_live())

    def test_metadata_events_do_not_extend_liveness(self):
        self.out.write_text(json.dumps({'type':'thread.started','thread_id':self.sid,'timestamp':lk.dt.datetime.now(lk.dt.timezone.utc).isoformat()})+'\n')
        self.assertEqual(lk.transcript_activity(self.out),0)

    def test_adapter_termination_kills_codex_and_command_grandchild(self):
        import signal
        with self.out.open('w') as log:
            process=subprocess.Popen(self.command(),env={**self.env,'FAKE_MODE':'wait'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                marker=self.repo/'grandchild.pid'
                for _ in range(100):
                    if marker.exists(): break
                    time.sleep(.05)
                self.assertTrue(marker.exists(),self.out.read_text())
                grandchild=int(marker.read_text())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=5),143,self.out.read_text())
                for _ in range(40):
                    status=subprocess.run(['ps','-o','stat=','-p',str(grandchild)],capture_output=True,text=True).stdout.strip()
                    if not status or status.startswith('Z'): break
                    time.sleep(.05)
                self.assertTrue(not status or status.startswith('Z'),f'grandchild {grandchild} survived')
            finally:
                if process.poll() is None:
                    process.terminate();process.wait(timeout=5)
