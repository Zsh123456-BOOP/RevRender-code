"""Materialize ONLY new train/dev rows from frozen 832 indices, and verify them.

Separate C832 schema and loader; never edit old B hashes or relax its loader.
The full private index contains test IDs for exclusion checks, not test pixels.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hme-revision-c832-train-dev-v1'
COMMON_TEST_SHA = '7d53501f33b7e40c55076df155e92e4ef352b804824faa1ee80d94bbe6fe1188'
MAPPINGS = {
    'group_random_s42': '4a9bfaa45ff4cd92f9c6cc0327148250a684b07a19c57267c27397b0cbc7fd8d',
    'type_source_s42': '734a6286f7cf6416e03bc07499dc36cfca1f9812724779f79f34f46854e81e02',
    'morphology_s42': '097f767f58f78d2f1c495e4b78c71d7b79483dad86c6f8e05c9a06e7603cc382',
    'writer_morphology_s42': '18eee22907cd56a0957b387a336ecc2093fbaeac4f0c3f8c285f1853c0745ea6',
    'writer_morphology_s2354': '1800ea0a5fb01d559bf86e06fb634a7b4585724483ae7427567c95ac15277163',
}
WRITER_COUNTS = {'group_random_s42': [354, 43, 44], 'type_source_s42': [350, 47, 44],
                 'morphology_s42': [346, 51, 44], 'writer_morphology_s42': [353, 44, 44],
                 'writer_morphology_s2354': [353, 44, 44]}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location('candidate833_' + name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def read_mapping(path):
    mapping = json.loads(Path(path).read_text())
    name = mapping.get('name')
    require(name in MAPPINGS and digest(path) == MAPPINGS[name], 'unregistered 832 mapping')
    rows = mapping['rows']
    require(len(rows) == 4760 and len({r['sample_id'] for r in rows}) == 4760, 'sample identity mismatch')
    require(Counter(r['split'] for r in rows) == {'train': 3808, 'dev': 476, 'test': 476}, 'split count mismatch')
    writers = {s: {r['group_uuid'] for r in rows if r['split'] == s} for s in ('train', 'dev', 'test')}
    require([len(writers[s]) for s in ('train', 'dev', 'test')] == WRITER_COUNTS[name], 'writer counts changed')
    require(not (writers['train'] & writers['dev'] or writers['train'] & writers['test'] or writers['dev'] & writers['test']), 'writer leakage')
    ids = sorted(r['sample_id'] for r in rows if r['split'] == 'test')
    test_sha = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(test_sha == COMMON_TEST_SHA == mapping['common_test_sample_ids_sha256'], 'common test changed')
    return mapping


def validate_records(records, mapping):
    allowed = {r['sample_id']: r for r in mapping['rows'] if r['split'] in ('train', 'dev')}
    require(len(records) == len(allowed) and {r['sample_id'] for r in records} == set(allowed),
            'missing/duplicate/test sample in train/dev bundle')
    suffixes = [r['sample_id'].split(':', 1)[-1] for r in records]
    require(len(set(suffixes)) == len(suffixes), 'archive suffix IDs collide')
    for i, row in enumerate(records):
        source = allowed[row['sample_id']]
        require(row['index'] == i and row['split'] == source['split'] and row['group_uuid'] == source['group_uuid'], 'row identity/split changed')
        require(row['source_origin'] == source['origin'] and row['source_index'] == source['index'], 'source pointer changed')
        require(type(row['is_positive']) is bool, 'invalid positive flag')
        expected = ('fit' if row['split'] == 'train' else 'validation') + ('_positive' if row['is_positive'] else '_clean')
        require(row['legacy_split'] == expected, 'sampler pool mismatch')


def build(mapping_path, output):
    require(not output.exists() and not output.is_symlink(), 'create-only output exists')
    require('work' in output.resolve().parts, 'private bundle belongs under work/')
    mapping = read_mapping(mapping_path)
    sources, records = {}, {}
    for origin, source in mapping['inputs'].items():
        root = Path(source['path'])
        require(digest(root/'run_manifest.json') == source['manifest_sha256'], 'source manifest changed')
        for filename, entry in source['files'].items():
            require(digest(root/filename) == (entry if isinstance(entry, str) else entry['sha256']), 'source bytes changed')
        records[origin] = json.loads((root/'records.json').read_text())
    import numpy as np
    for origin, source in mapping['inputs'].items():
        root = Path(source['path'])
        sources[origin] = {key: np.load(root/(key+'.npy'), mmap_mode='r', allow_pickle=False) for key in ('images', 'masks')}
    selected = [r for r in mapping['rows'] if r['split'] in ('train', 'dev')]
    selected.sort(key=lambda r: (r['split'] != 'train', not records[r['origin']][r['index']]['is_positive'], r['sample_id']))
    output.mkdir(parents=True, exist_ok=False)
    images = np.lib.format.open_memmap(output/'images.npy', mode='w+', dtype=np.uint8, shape=(4284, 3, 96, 768))
    masks = np.lib.format.open_memmap(output/'masks.npy', mode='w+', dtype=np.bool_, shape=(4284, 96, 768))
    out = []
    for i, row in enumerate(selected):
        record = records[row['origin']][row['index']]
        require(record['sample_id'] == row['sample_id'] and record['group_uuid'] == row['group_uuid'], 'source index identity mismatch')
        images[i] = sources[row['origin']]['images'][row['index']]
        masks[i] = sources[row['origin']]['masks'][row['index']]
        legacy = ('fit' if row['split'] == 'train' else 'validation') + ('_positive' if record['is_positive'] else '_clean')
        out.append({**record, 'index': i, 'split': row['split'], 'legacy_split': legacy,
                    'source_origin': row['origin'], 'source_index': row['index']})
    images.flush()
    masks.flush()
    validate_records(out, mapping)
    write_json(output/'records.json', out)
    (output/'index_manifest.json').write_bytes(Path(mapping_path).read_bytes())
    manifest = {'schema': SCHEMA, 'candidate': mapping['name'], 'mapping_sha256': digest(mapping_path),
        'common_test_sample_ids_sha256': COMMON_TEST_SHA, 'counts': dict(Counter(r['legacy_split'] for r in out)),
        'corrected_dev_components': sum(r['instance_count'] for r in out if r['split'] == 'dev'),
        'new_test_rows_in_bundle': 0, 'old_test_repooling_authorized': True,
        'git_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'files': {n: {'sha256': digest(output/n), 'bytes': (output/n).stat().st_size}
                  for n in ('images.npy', 'masks.npy', 'records.json', 'index_manifest.json')}}
    write_json(output/'run_manifest.json', manifest)
    data = load_bundle(output, digest(output/'run_manifest.json'))
    write_json(output/'complete.json', {'manifest_sha256': digest(output/'run_manifest.json'),
        'lines': {s: len(data[s]) for s in ('train', 'dev')}, 'verified': True})
    return {'candidate': mapping['name'], 'path': str(output.resolve()), 'sha256': digest(output/'run_manifest.json')}


def manifest_at(path, expected_sha):
    root = Path(path).resolve()
    require(digest(root/'run_manifest.json') == expected_sha, 'new bundle manifest changed')
    manifest = json.loads((root/'run_manifest.json').read_text())
    require(manifest['schema'] == SCHEMA and manifest['new_test_rows_in_bundle'] == 0, 'not C832 train/dev')
    require(manifest['candidate'] in MAPPINGS and manifest['mapping_sha256'] == MAPPINGS[manifest['candidate']], 'wrong candidate')
    return manifest


def load_bundle(path, expected_sha):
    root = Path(path).resolve()
    manifest = manifest_at(root, expected_sha)
    for name, entry in manifest['files'].items():
        require(not (root/name).is_symlink() and (root/name).stat().st_size == entry['bytes'] and digest(root/name) == entry['sha256'], 'bundle file changed')
    mapping = read_mapping(root/'index_manifest.json')
    require(mapping['name'] == manifest['candidate'], 'manifest/mapping mismatch')
    records = json.loads((root/'records.json').read_text())
    validate_records(records, mapping)
    import numpy as np
    native = load('514_revision_native_dataset.py')
    images = np.load(root/'images.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(root/'masks.npy', mmap_mode='r', allow_pickle=False)
    require(images.shape == (4284, 3, 96, 768) and images.dtype == np.uint8, 'image shape/dtype changed')
    require(masks.shape == (4284, 96, 768) and masks.dtype == np.bool_, 'mask shape/dtype changed')
    result = {'schema_version': native.SCHEMA, 'manifest': manifest, 'train': [], 'dev': []}
    for i, row in enumerate(records):
        instances = native.instances_from_mask(masks[i])
        require(len(instances) == row['instance_count'] and bool(len(instances)) == row['is_positive'], 'label mismatch')
        instances.flags.writeable = False
        result[row['split']].append({**row, 'image': images[i], 'mask': masks[i], 'instances': instances})
    require(dict(Counter(r['legacy_split'] for r in records)) == manifest['counts'], 'sampler counts changed')
    require(sum(len(r['instances']) for r in result['dev']) == manifest['corrected_dev_components'], 'dev components changed')
    return result


def dataset_for(dataset, bundle_dir, expected_sha):
    root = Path(bundle_dir).resolve()
    manifest = manifest_at(root, expected_sha)
    dataset.COUNTS = manifest['counts']
    dataset.DEV_COMPONENTS = manifest['corrected_dev_components']
    def loader(path):
        require(Path(path).resolve() == root, 'unexpected dataset path')
        return load_bundle(root, expected_sha)
    dataset.load_bundle = loader
    return dataset


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--indices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.indices, args.output)))


if __name__ == '__main__':
    main()
