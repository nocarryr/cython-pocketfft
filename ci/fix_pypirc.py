"""Set "username" values to "__token__"

According to PyPI:

    * To use an API token:
     * Set your username to __token__
     * Set your password to the token value, including the pypi- prefix

But the TwineAuthenticateV1 task sets the username to "build":
https://github.com/microsoft/azure-pipelines-tasks/blob/ce15c9039fd3cfd340eaad82a5062fcf1d0de1e2/Tasks/TwineAuthenticateV1/authentication.ts#L101
"""
import sys
from pathlib import Path

CONF_FILE = Path(sys.argv[1])

lines = []
replaced = 0

with CONF_FILE.open() as f:
    for line in f:
        if line.lower().startswith('username') and 'build' in line:
            line = 'username: __token__\n'
            replaced += 1
        lines.append(line)
print(f'replaced {replaced} lines')
if replaced:
    CONF_FILE.write_text(''.join(lines))
