import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import tmp_sweep  # noqa: E402


class TmpSweepTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        old = time.time() - 3 * 3600
        for name in ('job-stale', 'job-open', 'job-recent'):
            d = os.path.join(self.root, name, 'sub')
            os.makedirs(d)
            f = os.path.join(d, 'x.bin')
            with open(f, 'wb') as fh:
                fh.write(b'0' * 8192)
            if name != 'job-recent':
                for p in (f, d, os.path.dirname(d)):
                    os.utime(p, (old, old))
        locked = os.path.join(self.root, 'job-stale', 'sub')
        os.chmod(locked, 0o500)
        self.pattern = os.path.join(self.root, 'job-*')
        self.opened = {os.path.join(self.root, 'job-open', 'sub', 'x.bin')}

    def tearDown(self):
        for root, dirs, _ in os.walk(self.root):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o700)
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_dry_run_removes_nothing(self):
        r = tmp_sweep.sweep([self.pattern], 60, execute=False, opened=self.opened)
        self.assertEqual(r['removed'], 1)
        self.assertTrue(os.path.exists(os.path.join(self.root, 'job-stale')))

    def test_execute_removes_only_stale_unopened(self):
        r = tmp_sweep.sweep([self.pattern], 60, execute=True, opened=self.opened)
        self.assertEqual(r['removed'], 1)
        self.assertEqual(r['kept'], {'open': 1, 'recent': 1})
        self.assertFalse(os.path.exists(os.path.join(self.root, 'job-stale')))
        self.assertTrue(os.path.exists(os.path.join(self.root, 'job-open')))
        self.assertTrue(os.path.exists(os.path.join(self.root, 'job-recent')))

    def test_lsof_failure_removes_nothing(self):
        orig = tmp_sweep.open_paths
        tmp_sweep.open_paths = lambda: None
        try:
            r = tmp_sweep.sweep([self.pattern], 60, execute=True)
        finally:
            tmp_sweep.open_paths = orig
        self.assertIn('error', r)
        self.assertTrue(os.path.exists(os.path.join(self.root, 'job-stale')))


if __name__ == '__main__':
    unittest.main()
