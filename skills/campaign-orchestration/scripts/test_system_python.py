import glob
import os
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PYTHON = '/usr/bin/python3'


class SystemPythonTest(unittest.TestCase):
    """launchd runs the keeper and dashboard with /usr/bin/python3 (3.9 on this Mac); on 1 Nov a
    3.12-only f-string in an imported module stopped every keeper tick for 48 minutes."""

    @unittest.skipUnless(os.path.exists(SYSTEM_PYTHON), 'no system python')
    def test_every_script_compiles_under_system_python(self):
        for path in sorted(glob.glob(os.path.join(HERE, '*.py'))):
            r = subprocess.run([SYSTEM_PYTHON, '-c', 'import sys; compile(open(sys.argv[1]).read(), sys.argv[1], "exec")', path],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f'{os.path.basename(path)}: {r.stderr.strip()[-300:]}')


if __name__ == '__main__':
    unittest.main()
