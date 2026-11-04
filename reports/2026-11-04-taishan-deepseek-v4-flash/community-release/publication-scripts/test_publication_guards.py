"""Offline publication-gate checks using synthetic credentials and a fake API."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types

BASE=Path('/publish')
for scenario in ('wrong_namespace','audit_failed','incomplete_audit'):
    spec=importlib.util.spec_from_file_location('publisher_test',BASE/'publish_after_audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=Path(tempfile.mkdtemp(prefix='guard-test-',dir=BASE))
    module.ROOT=folder
    module.CANDIDATE=folder/'candidate';module.CANDIDATE.mkdir()
    module.REPO=module.CANDIDATE/'repository';module.REPO.mkdir()
    (folder/'.upload-auth').write_text('synthetic-credential-for-offline-test')
    (module.CANDIDATE/'status.json').write_text(json.dumps({'phase':'failed' if scenario=='audit_failed' else 'ready_for_review'}))
    (module.REPO/'validation').mkdir()
    (module.REPO/'validation/full-audit.json').write_text(json.dumps({'passed':False,'layers':4,'expert_projections_checked':3072}))
    calls=[]
    class FakeApi:
        def __init__(self,**kwargs):pass
        def whoami(self):return types.SimpleNamespace(username='different-user' if scenario=='wrong_namespace' else 'kmon03f')
        def repo_exists(self,*args):calls.append('repo_exists');return False
        def create_repo(self,*args,**kwargs):calls.append('create_repo');raise AssertionError('Must not create repository')
        def upload_folder(self,*args,**kwargs):calls.append('upload_folder');raise AssertionError('Must not upload')
    sys.modules['modelscope_hub']=types.SimpleNamespace(HubApi=FakeApi)
    captured=io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):module.main()
        raise AssertionError('Invalid gate unexpectedly passed')
    except (ValueError,RuntimeError,AssertionError):
        assert calls==[],calls
        assert 'synthetic-credential-for-offline-test' not in captured.getvalue()
        assert not (folder/'.upload-auth').exists()
print(json.dumps({'namespace_mismatch_blocks_publication':True,'audit_failure_blocks_publication':True,'incomplete_full_audit_blocks_publication':True,'synthetic_credentials_not_logged':True,'staging_credentials_removed':True}))
