"""Split B: pool every revision-annotated line and re-split it 8:1:1 by student, stratified by revision type.

The first protocol round (涂改定位流程.md) kept the three splits as they came:
training and validation lines were picked in earlier annotation rounds, while
the test lines are every line of the 151 M0 test pages.  The three therefore
differ in kind as well as in size -- lines carrying a blot are 53% of the
training revisions, 47% of validation and 35% of test -- and, counted in
revised lines, the split is 76 : 6 : 18.  The user asked (2026-09-24) to try
"方案 B": pool all three and re-split them 8:1:1.

What is kept is the unit of the split, the student: a writer's hand is never
on both sides of it.  What is balanced is the share every split holds of five
counts -- lines whose revision is only strokes, only blots, or both; clean
lines; and lines from the full-page test annotation -- so each split holds the
same mix of revision types and of annotation sources.  The students of each
source are split on their own, greedily, largest first, and the union must
come within 0.03 of 8:1:1 on all five counts or the run is refused.  Every mask is the one its gold already holds
(788 training, 784 validation, 795 test); the eight training rows the annotator
emptied, which 788 had to keep because 514 fixed the positive count, become
the clean rows the annotator made them.

The price, stated rather than hidden: the pool includes the 28 M0 test
students, so a model trained on split B has seen some of them and may not be
used in the M6 pipeline for the main task without retraining on a split that
keeps them out.  514 refuses a bundle that says it touched the frozen test
split, and this one says so, so ``load_bundle`` here is 514's check list with
that one line replaced by split B's declaration, and ``dataset_for`` hands it
to an entry script for this bundle's digest only.

This script deliberately does NOT: change, redraw or drop any mask or image;
use any model output or any reading of any model, on any split, to decide an
assignment; split a student across splits; or contact the network.  Its
output holds student handwriting and belongs under work/.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hme-revision-split-b-v1'
TEST_SCHEMA = 'hme-revision-test-gold-split-b-v1'
PROTOCOL_BUNDLE_SHA = 'cbd833ebe4f08c3d8f1398d40bb4f01b8e96875ea599942e1b6455b2e5528baf'   # 792: 788 train + 784 validation
TEST_GOLD_SHA = 'f4a45632e8e90da29e5441ae96387690a77d7d039b0f706b6b2f7ee8d09546c8'        # 795
EXPORTS = {'train': '19c402e0560323ce620a537b4a6c23bfc2e1f5dc1a1a11ee5e4bcf8e7fd84d5f',
           'validation': '608cedaaa301564d5f8c038f4858d4d68f975b8a130c6dd12ffc6f64885bd868',
           'test': 'd54bae8e14ebb5dbca89b3c895f176d4592a446462246b1c8145be02f27881a0'}
SHARES = {'train': .8, 'dev': .1, 'test': .1}
CRITERIA = ('line_only', 'blot_only', 'mixed', 'clean', 'full_page')
SALT = 'split-b-seed-42'
TOLERANCE = .03
SPLIT_B_BUNDLE_SHA = '3d35b81f732e3b244aa93741199eedac17ff7e7342b55e920abff57070740b11'   # revision_split_b798_v2/bundle


class SplitBError(RuntimeError):
    pass


def digest(path):
    accumulator = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            accumulator.update(block)
    return accumulator.hexdigest()


def load(name, key):
    spec = importlib.util.spec_from_file_location(key, ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def revision_type(marks):
    kinds = {'blot' if m.get('type') == 'blot' else 'line' for m in marks}
    return 'clean' if not kinds else 'line_only' if kinds == {'line'} else 'blot_only' if kinds == {'blot'} else 'mixed'


def assign(students, criteria=CRITERIA):
    """Greedy, largest student first: each goes where the shares end closest to 8:1:1.

    ``students`` maps a student to its counts over ``criteria``.  Ties are
    broken by a salted hash of the student, so the split is a pure function of
    the counts.
    """
    totals = {c: sum(v[c] for v in students.values()) or 1 for c in criteria}
    held = {s: {c: 0 for c in criteria} for s in SHARES}
    order = sorted(students, key=lambda s: (-sum(students[s][c] for c in criteria),
                                            hashlib.sha256((SALT+s).encode()).hexdigest()))
    out = {}

    def cost(state):
        return sum((state[s][c]/totals[c]-SHARES[s])**2 for s in SHARES for c in criteria)

    for student in order:
        best = None
        for split in SHARES:
            trial = {s: dict(v) for s, v in held.items()}
            for c in criteria:
                trial[split][c] += students[student][c]
            candidate = (cost(trial), list(SHARES).index(split))
            if best is None or candidate < best[0]:
                best = (candidate, split, trial)
        out[student], held = best[1], best[2]
    return out, held


def assign_by_source(students):
    """8:1:1 within each annotation source, then checked over all five counts.

    A full-page student carries tens of lines, most of them clean; assigned in
    one greedy pass with everyone else they all land in training before the
    many small students arrive.  So the two sources are split on their own, on
    the four type counts, and the union is refused unless every one of the five
    shares is within TOLERANCE of 8:1:1.
    """
    types = CRITERIA[:4]
    out = {}
    for full_page in (True, False):
        group = {s: v for s, v in students.items() if bool(v['full_page']) == full_page}
        out.update(assign(group, types)[0])
    held = {s: {c: 0 for c in CRITERIA} for s in SHARES}
    for student, split in out.items():
        for c in CRITERIA:
            held[split][c] += students[student][c]
    for c in CRITERIA:
        total = sum(held[s][c] for s in SHARES) or 1
        for split, share in SHARES.items():
            if abs(held[split][c]/total-share) > TOLERANCE:
                raise SplitBError(f'{c} in {split} is {held[split][c]/total:.3f}, not within {TOLERANCE} of {share}')
    return out, held


def load_bundle(dataset, path, counts, dev_components):
    """514's load_bundle for split B: every check 514 makes, with split B's provenance line.

    514 requires ``frozen_test_touched is False``; split B holds M0 test
    students and says so, so this requires the split-B declaration instead.
    Everything else -- schema, files and digests, array shapes, record order,
    labels against masks, component counts, writer isolation -- is 514's check,
    with 514's own helpers, against split B's counts.
    """
    import numpy as np
    require = dataset.require
    root = dataset.safe_path(Path(path))
    manifest_path = root/'run_manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'bundle manifest absent/symlinked')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('schema_version') == dataset.SCHEMA, 'unsupported bundle schema')
    require(manifest.get('frozen_test_touched') is True and manifest.get('split_b', {}).get('schema') == SCHEMA
            and manifest.get('writer_overlap_train_dev') == 0, 'not a split-B bundle that declares its provenance')
    require(manifest.get('counts') == counts and manifest.get('corrected_dev_components') == dev_components,
            'bundle count contract changed')
    require(manifest.get('gold_correction', {}).get('applications_per_positive') == 1
            and manifest.get('gold_correction', {}).get('connectivity') == 8, 'gold correction contract changed')
    for name in ('images.npy', 'masks.npy', 'records.json'):
        item = root/name
        require(item.is_file() and not item.is_symlink() and dataset.safe_path(item).is_relative_to(root),
                f'missing/symlinked bundle file: {name}')
        expected = manifest['files'][name]
        require(item.stat().st_size == expected['bytes'] and dataset.digest(item) == expected['sha256'],
                f'bundle integrity mismatch: {name}')
    records = json.loads((root/'records.json').read_text(encoding='utf-8'))
    images = np.load(root/'images.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(root/'masks.npy', mmap_mode='r', allow_pickle=False)
    count = sum(counts.values())
    require(images.shape == (count, *dataset.IMAGE_SHAPE) and images.dtype == np.uint8, 'unexpected bundle image array')
    require(masks.shape == (count, *dataset.MASK_SHAPE) and masks.dtype == np.bool_, 'unexpected bundle mask array')
    require(isinstance(records, list) and len(records) == count, 'invalid bundle record count')
    result = {'schema_version': dataset.SCHEMA, 'train': [], 'dev': [], 'manifest': manifest}
    seen, groups = set(), {'train': set(), 'dev': set()}
    observed = {key: 0 for key in counts}
    components = 0
    for index, record in enumerate(records):
        require(type(record.get('index')) is int and record['index'] == index, 'bundle index/order mismatch')
        name = record['legacy_split']
        require(name in counts, 'undeclared/frozen split')
        split = 'train' if name.startswith('fit_') else 'dev'
        require(record['split'] == split and type(record['is_positive']) is bool
                and record['is_positive'] == name.endswith('positive'), 'bundle split/label mismatch')
        require(record['sample_id'] not in seen, 'duplicate bundle sample ID')
        seen.add(record['sample_id'])
        groups[split].add(record['group_uuid'])
        observed[name] += 1
        mask = masks[index]
        require(bool(mask.any()) == record['is_positive'], 'bundle foreground/label mismatch')
        instances = dataset.instances_from_mask(mask)
        require(len(instances) == record['instance_count'], 'instance count differs from archived mask')
        if split == 'dev':
            components += len(instances)
        instances.flags.writeable = False
        result[split].append({**record, 'image': images[index], 'mask': mask, 'instances': instances})
    require(observed == counts and components == dev_components, 'loaded bundle counts changed')
    require(not groups['train'] & groups['dev'], 'loaded bundle writer overlap')
    return result


def dataset_for(dataset, bundle_dir):
    """514 as an entry script must see it for split B, for the pinned split-B bundle only."""
    root = Path(bundle_dir)
    if SPLIT_B_BUNDLE_SHA is None or digest(root/'run_manifest.json') != SPLIT_B_BUNDLE_SHA:
        raise SplitBError('the split-B contract applies only to the pinned split-B bundle')
    if dataset.COUNTS != {'fit_positive': 1515, 'fit_clean': 1319, 'validation_positive': 126, 'validation_clean': 379} \
            or dataset.DEV_COMPONENTS != 221:
        raise SplitBError('514 no longer declares the registered contract this replaces')
    manifest = json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
    counts, components = dict(manifest['counts']), manifest['corrected_dev_components']
    dataset.COUNTS, dataset.DEV_COMPONENTS = counts, components
    original = dataset.load_bundle

    def loader(path):
        if Path(path).resolve() != root.resolve():
            return original(path)
        return load_bundle(dataset, path, counts, components)
    dataset.load_bundle = loader
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--protocol-bundle', type=Path, required=True, help="792's bundle")
    parser.add_argument('--test-gold', type=Path, required=True, help="795's output")
    parser.add_argument('--train-export', type=Path, required=True)
    parser.add_argument('--validation-export', type=Path, required=True)
    parser.add_argument('--test-export', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for path in (args.bundle_dir, args.test_dir):
            if path.exists():
                raise SplitBError(f'create-only output exists: {path}')
        if digest(args.protocol_bundle/'run_manifest.json') != PROTOCOL_BUNDLE_SHA or \
                digest(args.test_gold/'run_manifest.json') != TEST_GOLD_SHA:
            raise SplitBError('inputs are not 792 and 795')
        exports = {}
        for name, path in (('train', args.train_export), ('validation', args.validation_export), ('test', args.test_export)):
            if digest(path) != EXPORTS[name]:
                raise SplitBError(f'the {name} export is not the registered one')
            exports[name] = json.loads(path.read_text(encoding='utf-8'))['records']
        import numpy as np
        dataset = load('514_revision_native_dataset.py', 'splitb798_514')
        protocol = load('792_revision_protocol_bundle.py', 'splitb798_792')
        loaded = protocol.dataset_for(dataset, args.protocol_bundle).load_bundle(args.protocol_bundle)
        source_manifest = loaded['manifest']
        emptied = {e['line_id'] for e in source_manifest['reannotation_788']['rows_emptied_kept_stored_mask']}
        if len(emptied) != 8:
            raise SplitBError('788 does not list the eight emptied rows')
        test_manifest = json.loads((args.test_gold/'run_manifest.json').read_text())
        for name, sha in test_manifest['files'].items():
            if digest(args.test_gold/name) != sha:
                raise SplitBError('795 changed: '+name)
        test_images = np.load(args.test_gold/'images.npy', mmap_mode='r')
        test_masks = np.load(args.test_gold/'masks.npy', mmap_mode='r')
        test_records = json.loads((args.test_gold/'records.json').read_text())

        pool = []
        for split, source, records in (('train', 'train788', loaded['train']), ('dev', 'validation784', loaded['dev'])):
            export = exports['train' if split == 'train' else 'validation']
            for r in records:
                line = str(r['line_id'])
                mask = np.asarray(r['mask'], bool)
                if line in emptied:
                    mask = np.zeros_like(mask)
                if mask.any() and line not in export:
                    raise SplitBError(f'{line} carries a revision its export does not hold')
                kind = revision_type(export[line]['marks']) if mask.any() else 'clean'
                pool.append({'sample_id': r['sample_id'], 'line_id': line, 'image_id': r['image_id'],
                             'group_uuid': r['group_uuid'], 'source': source, 'image': r['image'], 'mask': mask,
                             'revision_type': kind})
        for r in test_records:
            mask = np.asarray(test_masks[r['index']], bool)
            pool.append({'sample_id': r['sample_id'], 'line_id': r['line_id'], 'image_id': r['image_id'],
                         'group_uuid': r['group_uuid'], 'source': 'test795', 'image': test_images[r['index']],
                         'mask': mask, 'revision_type': revision_type(exports['test'][r['line_id']]['marks'])
                         if mask.any() else 'clean'})
        if len({r['sample_id'] for r in pool}) != len(pool):
            raise SplitBError('a line is in two golds')
        if any((r['revision_type'] == 'clean') == bool(r['mask'].any()) for r in pool):
            raise SplitBError('a revision type disagrees with its mask')
        students = {}
        for r in pool:
            counts = students.setdefault(r['group_uuid'], {c: 0 for c in CRITERIA})
            counts[r['revision_type']] += 1
            counts['full_page'] += int(r['source'] == 'test795')
        assignment, held = assign_by_source(students)

        splits = {s: [r for r in pool if assignment[r['group_uuid']] == s] for s in SHARES}
        bundle_rows = sorted(splits['train'], key=lambda r: (not r['mask'].any(), r['sample_id'])) + \
            sorted(splits['dev'], key=lambda r: (not r['mask'].any(), r['sample_id']))
        args.bundle_dir.mkdir(parents=True)
        images = np.stack([np.asarray(r['image'], np.uint8) for r in bundle_rows])
        masks = np.stack([r['mask'] for r in bundle_rows])
        records, counts = [], {'fit_positive': 0, 'fit_clean': 0, 'validation_positive': 0, 'validation_clean': 0}
        for index, r in enumerate(bundle_rows):
            split = assignment[r['group_uuid']]
            positive = bool(r['mask'].any())
            legacy = ('fit_' if split == 'train' else 'validation_')+('positive' if positive else 'clean')
            counts[legacy] += 1
            records.append({'index': index, 'sample_id': r['sample_id'], 'line_id': r['line_id'],
                            'image_id': r['image_id'], 'group_uuid': r['group_uuid'], 'split': split,
                            'legacy_split': legacy, 'is_positive': positive,
                            'instance_count': len(dataset.instances_from_mask(r['mask'])),
                            'source': r['source'], 'revision_type': r['revision_type']})
        np.save(args.bundle_dir/'images.npy', images, allow_pickle=False)
        np.save(args.bundle_dir/'masks.npy', masks, allow_pickle=False)
        (args.bundle_dir/'records.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
        dev_components = sum(r['instance_count'] for r in records if r['split'] == 'dev')
        stats = {s: {c: held[s][c] for c in CRITERIA} | {'students': sum(1 for v in assignment.values() if v == s),
                                                          'lines': len(splits[s]),
                                                          'revised_lines': sum(bool(r['mask'].any()) for r in splits[s])}
                 for s in SHARES}
        m0_test_students = {r['group_uuid'] for r in pool if r['source'] == 'test795'}
        manifest = {key: source_manifest[key] for key in ('schema_version', 'gold_correction', 'image_dtype',
                                                          'image_layout', 'image_shape', 'mask_dtype', 'mask_shape')}
        manifest.update(counts=counts, corrected_dev_components=dev_components, writer_overlap_train_dev=0,
                        frozen_test_touched=True, create_only=True, training_performed=False,
                        files={n: {'bytes': (args.bundle_dir/n).stat().st_size, 'sha256': digest(args.bundle_dir/n)}
                               for n in ('images.npy', 'masks.npy', 'records.json')},
                        split_b={'schema': SCHEMA, 'shares': SHARES, 'criteria': CRITERIA, 'salt': SALT, 'tolerance': TOLERANCE,
                                 'method': 'each annotation source split 8:1:1 on its own, greedy largest first',
                                 'unit': 'student', 'statistics': stats,
                                 'm0_test_students_by_split': {s: len({r['group_uuid'] for r in splits[s]} & m0_test_students)
                                                               for s in SHARES},
                                 'inputs': {'protocol_bundle': PROTOCOL_BUNDLE_SHA, 'test_gold': TEST_GOLD_SHA,
                                            'exports': EXPORTS},
                                 'emptied_training_rows_made_clean': sorted(emptied),
                                 'authorisation': 'user 2026-09-24 (方案 B); task_card.md blocker #18',
                                 'm6_caveat': 'holds M0 test students; not for the M6 main-task pipeline without retraining',
                                 'script_sha256': digest(Path(__file__).resolve()), 'python': platform.python_version()})
        (args.bundle_dir/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')

        test_rows = sorted(splits['test'], key=lambda r: r['sample_id'])
        args.test_dir.mkdir(parents=True)
        np.save(args.test_dir/'images.npy', np.stack([np.asarray(r['image'], np.uint8) for r in test_rows]), allow_pickle=False)
        np.save(args.test_dir/'masks.npy', np.stack([r['mask'] for r in test_rows]), allow_pickle=False)
        test_out = [{'index': i, 'sample_id': r['sample_id'], 'line_id': r['line_id'], 'image_id': r['image_id'],
                     'group_uuid': r['group_uuid'], 'split': 'test', 'is_positive': bool(r['mask'].any()),
                     'instance_count': len(dataset.instances_from_mask(r['mask'])), 'source': r['source'],
                     'revision_type': r['revision_type']} for i, r in enumerate(test_rows)]
        (args.test_dir/'records.json').write_text(json.dumps(test_out, ensure_ascii=False), encoding='utf-8')
        (args.test_dir/'run_manifest.json').write_text(json.dumps({
            'schema': TEST_SCHEMA, 'split_b_bundle_manifest_sha256': digest(args.bundle_dir/'run_manifest.json'),
            'files': {n: digest(args.test_dir/n) for n in ('images.npy', 'masks.npy', 'records.json')},
            'lines': len(test_out), 'revised_lines': sum(r['is_positive'] for r in test_out),
            'components': sum(r['instance_count'] for r in test_out), 'statistics': stats['test'],
            'locked': "read only by a test pass that follows a model's frozen validation choices",
            'frozen_test_touched': True, 'create_only': True}, ensure_ascii=False, indent=1, sort_keys=True)+'\n')
        summary = {'bundle_sha256': digest(args.bundle_dir/'run_manifest.json'), 'counts': counts,
                   'dev_components': dev_components, 'test_revised_lines': sum(r['is_positive'] for r in test_out),
                   'test_components': sum(r['instance_count'] for r in test_out), 'statistics': stats,
                   'm0_test_students_by_split': manifest['split_b']['m0_test_students_by_split']}
    except SplitBError as exc:
        raise SystemExit('798 split B refused: '+str(exc))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
