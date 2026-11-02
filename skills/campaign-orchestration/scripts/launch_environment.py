"""Prepare the shared launch environment before consuming any lane state."""
import os
from pathlib import Path
import shlex
import shutil
import socket
import sys


def environment(lane, engine=None):
    exports = {}
    # Both the keeper service and manual ssh launches must use alif's login.
    if socket.gethostname().split('.')[0].startswith('alif') or str(Path(lane).absolute()).startswith(('/srv/agents/skard/', '/srv/media/agents/skard/')):
        exports['CODEX_HOME'] = '/opt/alif/.codex'
    if engine:
        executable = shutil.which(engine)
        if not executable:
            raise ValueError(engine + ' is not executable on PATH; no launch state consumed')
        # Codex's npm entry point uses /usr/bin/env node. An absolute CLI path
        # alone does not repair a non-login shell's missing Node PATH.
        with open(executable, 'rb') as stream:
            header = stream.readline(256)
        if b'/usr/bin/env node' in header and not shutil.which('node'):
            raise ValueError('node is not executable on PATH for ' + executable)
        exports['LANE_EXECUTABLE'] = str(Path(executable).absolute())
    return exports


if __name__ == '__main__':
    try:
        for key, value in environment(*sys.argv[1:]).items():
            print('export ' + key + '=' + shlex.quote(value))
    except (OSError, ValueError) as error:
        print('refused: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
