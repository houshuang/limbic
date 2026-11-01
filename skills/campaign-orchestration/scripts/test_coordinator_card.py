import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('card', HERE / 'coordinator_card.py')
card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card)


class CoordinatorCardTest(unittest.TestCase):
    def test_realistic_queue_and_incident_headings(self):
        with tempfile.TemporaryDirectory() as d:
            q, i = Path(d)/'QUEUE.md', Path(d)/'INCIDENTS.md'
            q.write_text('## P0\n- [~] **Fix** → lane fix-20261101 (code): details.\n'
                         '- [!] **Release**: waits on fix-20261101 READY; then test.\n'
                         '- [ ] **Inventory** (Codex, code lane): read fixture tests.\n'
                         '- [~] **Recovery** → lane old-123. Superseded by next-456; waits on RESULTS.json and serial owner next-456.\n'
                         '- [x] **Done** → lane old-1\n')
            i.write_text('# old\n## one\n## two\n## three\n## four\nbody\n')
            result = card.make_card(q, i)
            self.assertIn('[~] **Fix** · owner: fix-20261101', result)
            self.assertIn('[!] **Release** · owner: ? · wait: fix-20261101 READY', result)
            self.assertIn('[ ] **Inventory** · owner: ?', result)
            self.assertNotIn('Done', result)
            self.assertIn('wait: RESULTS.json and serial owner next-456', result)
            self.assertIn('may be stale or superseded', result)
            self.assertNotIn('## one\n', result)
            self.assertIn('## two\n## three\n## four', result)
            self.assertIn('Explicit user instructions', result)
            self.assertIn('If instructions require a file, read it', result)
            self.assertLessEqual(len(result.encode()), 4096)

    def test_overflow_is_explicit_and_missing_sources_are_safe(self):
        with tempfile.TemporaryDirectory() as d:
            q, i = Path(d)/'QUEUE.md', Path(d)/'INCIDENTS.md'
            missing = card.make_card(q, i)
            self.assertIn('queue unavailable', missing)
            self.assertIn('incidents unavailable', missing)
            q.write_text(''.join(f'- [ ] **Item {n}**\n' for n in range(300)))
            i.write_text('## last\n')
            result = card.make_card(q, i)
            self.assertIn('queue index exceeds card limit', result)
            self.assertLessEqual(len(result.encode()), 4096)

    def test_keeper_prompt_inlines_card(self):
        # The keeper test supplies its real startup fixture and captures its generated prompt.
        import test_lane_keeper as keeper_tests
        case = keeper_tests.RetireTest('test_codex_successor_uses_adapter_and_run_specific_retirement')
        case.setUp()
        try:
            (case.K.state_dir / 'QUEUE.md').write_text('- [~] **Card fixture** → lane owner-123\n')
            (case.K.state_dir / 'INCIDENTS.md').write_text('## recent incident\n')
            case.K.cfg['coordinator'] = {'engine': 'codex'}
            case.K.st['events'] = {'k': {'lane': '/lane', 'kind': 'READY', 'detail': 'checkpoint'}}
            with mock.patch.object(case.K, 'coordinator_live', return_value=None), mock.patch.object(keeper_tests.lk.subprocess, 'Popen') as popen:
                popen.return_value.pid = 99999999
                case.K.maybe_start_coordinator()
            prompt = (case.K.k / 'coordinator-prompt.txt').read_text()
            self.assertIn('[~] **Card fixture** · owner: owner-123', prompt)
            self.assertIn('## recent incident', prompt)
        finally:
            case.tearDown()

if __name__ == '__main__':
    unittest.main()
