import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD, NEW = '11111111-1111-1111-1111-111111111111', '22222222-2222-2222-2222-222222222222'


class CostguardTest(unittest.TestCase):
    def test_appended_log_is_charged_to_its_latest_session(self):
        with tempfile.TemporaryDirectory() as d:
            home, camp = Path(d) / 'home', Path(d) / 'camp'
            sessions = home / '.codex' / 'sessions' / '2026' / '11' / '02'
            sessions.mkdir(parents=True); (camp / 'lane').mkdir(parents=True)
            ts = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
            for sid, ctx in ((OLD, 400_000), (NEW, 20_000)):
                row = {'timestamp': ts, 'payload': {'type': 'token_count', 'info': {
                    'total_token_usage': {'total_tokens': ctx}, 'last_token_usage': {'input_tokens': ctx}}}}
                (sessions / f'rollout-{sid}.jsonl').write_text(json.dumps(row) + '\n')
            (camp / 'lane' / 'codex.log').write_text(f'session id: {OLD}\nold work\nsession id: {NEW}\nnew work\n')
            r = subprocess.run([sys.executable, str(HERE / 'costguard.py'), 'lane'], capture_output=True, text=True,
                               env={**os.environ, 'HOME': str(home), 'CAMPAIGN_DIR': str(camp)})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('20,000', (camp / 'COST.md').read_text())


if __name__ == '__main__':
    unittest.main()
