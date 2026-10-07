import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('entry',ROOT/'revrender.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

def test_input_pixels_preserved_and_malformed_shape_refused(tmp_path):
    x=np.arange(96*768,dtype=np.uint8).reshape(96,768);p=tmp_path/'image.npy';np.save(p,x)
    np.testing.assert_array_equal(entry.read_image(p),x)
    np.save(p,np.zeros((97,768),dtype=np.uint8))
    with pytest.raises(ValueError,match='shape'):entry.read_image(p)

def test_missing_asset_prevents_prediction_without_loading_torch(tmp_path):
    from types import SimpleNamespace
    out=tmp_path/'already.npz';out.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):entry.predict(SimpleNamespace(output=out))
    assert out.read_bytes()==b'preserve'

@pytest.mark.parametrize('command',[[],['train'],['predict'],['freeze'],['evaluate']])
def test_help_without_gpu_or_private_data(command):
    result=subprocess.run([sys.executable,str(ROOT/'revrender.py'),*command,'--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'usage:' in result.stdout

def test_published_source_fingerprints_and_static_dependencies():
    import re
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for row in manifest['files']:
        p=ROOT/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.Constant) and isinstance(n.value,str) and '\n' not in n.value:
                for name in re.findall(r'\b\d{3}_[a-zA-Z0-9_]+\.py\b',n.value):assert (ROOT/'scripts'/name).is_file(),name
