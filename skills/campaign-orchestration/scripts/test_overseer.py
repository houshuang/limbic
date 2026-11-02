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

    def review_cfg(self):
        req, rev = self.d / 'REVIEW-REQUEST.md', self.d / 'REVIEW.md'
        cfg = json.loads(self.cfg.read_text())
        cfg['overseer']['reviews'] = [{'request': str(req), 'review': str(rev)}]
        self.cfg.write_text(json.dumps(cfg))
        return req, rev

    def test_review_runs_when_request_is_newer_and_not_when_answered(self):
        req, rev = self.review_cfg()
        rev.write_text('old'); req.write_text('please'); os.utime(rev, (1, 1))
        calls = []
        def fake(cmd, **kw):
            calls.append(cmd); rev.write_text('## Verdict\nAPPROVE'); return mock.Mock(returncode=0)
        with mock.patch.object(overseer, 'budget_ok', return_value=(True, 'claude: OK')), \
             mock.patch.object(overseer, 'launch', side_effect=overseer.review_one), \
             mock.patch.object(overseer.subprocess, 'run', side_effect=fake):
            overseer.review(self.cfg)
            overseer.review(self.cfg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index('--model') + 1], 'fable')
        self.assertIn(str(req), calls[0][2])

    def test_failed_review_is_reported_and_not_retried_at_once(self):
        req, rev = self.review_cfg()
        req.write_text('please')
        with mock.patch.object(overseer, 'budget_ok', return_value=(True, 'claude: OK')), \
             mock.patch.object(overseer, 'launch', side_effect=overseer.review_one), \
             mock.patch.object(overseer.subprocess, 'run', return_value=mock.Mock(returncode=1)) as run:
            overseer.review(self.cfg)
            overseer.review(self.cfg)
            self.assertEqual(run.call_count, 1)
        self.assertIn('failed (1, attempt 1 of 3)', (self.d / 'R.md').read_text())
        seen = json.loads((self.d / 'st' / 'reviews.json').read_text())
        seen[str(req)]['t'] -= 1801
        self.assertTrue(overseer.review_due({'request': str(req), 'review': str(rev)}, seen, __import__('time').time()))

    def test_lane_requests_are_found_and_reviewed_in_parallel(self):
        lanes = self.d / 'campaign'
        for name in ('a', 'b', 'c', 'd'):
            (lanes / name).mkdir(parents=True); (lanes / name / 'REVIEW-REQUEST.md').write_text('please')
        cfg = json.loads(self.cfg.read_text())
        cfg['overseer']['review_globs'] = [str(lanes / '*' / 'REVIEW-REQUEST.md')]
        cfg['overseer']['review_parallel'] = 3
        self.cfg.write_text(json.dumps(cfg))
        started = []
        with mock.patch.object(overseer, 'budget_ok', return_value=(True, 'claude: OK')), \
             mock.patch.object(overseer, 'launch', side_effect=lambda c, i: started.append(i)):
            overseer.review(self.cfg)
        self.assertEqual(len(started), 3)
        self.assertEqual(started[0]['review'], str(lanes / 'a' / 'REVIEW.md'))


if __name__ == '__main__':
    unittest.main()
