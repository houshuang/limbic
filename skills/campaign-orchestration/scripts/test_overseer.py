import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import overseer  # noqa: E402


class OverseerTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        (self.d / 'PROMPT.md').write_text('now {NOW}; report {REPORT}')
        self.cfg = self.d / 'c.json'
        self.cfg.write_text(json.dumps({'repo': str(self.d), 'overseer': {
            'prompt': str(self.d / 'PROMPT.md'), 'state': str(self.d / 'st'), 'report': str(self.d / 'R.md')}}))

    def test_budget_refusal_is_reported_not_run(self):
        with mock.patch.object(overseer, 'budget_ok', return_value=(False, 'claude: HOLD reserve')), \
             mock.patch.object(overseer.subprocess, 'run') as run:
            overseer.run(self.cfg)
            run.assert_not_called()
        self.assertIn('overseer skipped: claude: HOLD reserve', (self.d / 'R.md').read_text())

    def test_run_fills_prompt_and_records_exit(self):
        seen = {}
        def fake(cmd, **kw):
            seen['prompt'] = cmd[2]; seen['cwd'] = kw['cwd']
            return mock.Mock(returncode=0)
        with mock.patch.object(overseer, 'budget_ok', return_value=(True, 'claude: OK')), \
             mock.patch.object(overseer.subprocess, 'run', side_effect=fake):
            overseer.run(self.cfg)
        self.assertIn(str(self.d / 'R.md'), seen['prompt'])
        self.assertNotIn('{NOW}', seen['prompt'])
        self.assertEqual(seen['cwd'], str(self.d))
        self.assertEqual(json.loads((self.d / 'st' / 'last.json').read_text())['exit'], 0)

    def test_budget_line_parsed_from_real_status_format(self):
        out = 'autonomy: x\ncodex: OK: codex 57% used\nclaude: OK: claude 0% used of a 11% line\n'
        with mock.patch.object(overseer.subprocess, 'run', return_value=mock.Mock(stdout=out)):
            self.assertEqual(overseer.budget_ok(), (True, 'claude: OK: claude 0% used of a 11% line'))


if __name__ == '__main__':
    unittest.main()
