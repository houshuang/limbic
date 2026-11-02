import json,os,subprocess,sys
from pathlib import Path
S=Path(__file__).resolve().parent
W=Path("/Users/stian/src/skard/.claude/worktrees/audit-admission-2255")
expected="0be21515639c46aba67fc93dd069349ad3ece1cf"
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==expected
os.environ.update(SKARD_ROOT=str(W),SKARD_SHARED_ROOT='/Users/stian/src/skard',SKARD_NATIVE_SNAPSHOT='/Users/stian/src/skard/data/derived/delivery/request',SKARD_TEST_JOBS='2',SKARD_CONTENTS_AUDIT_FROZEN='/Users/stian/src/skard-state/campaign-20261023/contents-pooled-audit-20261101-2304/RESULTS.json')
commands=[['/usr/bin/python3','-m','py_compile','scripts/contents_audit.py','tests/test_contents_audit.py','tests/test_contents_audit_frozen.py'],['/usr/bin/python3','-m','unittest','tests.test_contents_audit','tests.test_contents_audit_frozen','-q'],['bash','scripts/heavy.sh','python3','scripts/preflight.py','--base','0871a15b63bdcfdf2f1856befa22148f8f2cffe6','--rev',expected,'--json']]
results=[]
for i,cmd in enumerate(commands):
 with (S/f'check-{i}.log').open('w') as f:
  p=subprocess.run(cmd,cwd=W,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(command=cmd,returncode=p.returncode,log=str(S/f'check-{i}.log')))
 if p.returncode:break
(S/'RESULTS.json').write_text(json.dumps(dict(revision=expected,status='PASS' if len(results)==3 and all(r['returncode']==0 for r in results) else 'FAIL',checks=results),indent=2)+'\n')
sys.exit(0 if len(results)==3 and all(r['returncode']==0 for r in results) else 1)
