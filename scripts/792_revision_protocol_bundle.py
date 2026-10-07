"""The protocol's bundle: 788's training split with 784's validation gold in its validation split.

涂改定位流程.md fixes the three golds a model under the protocol reads:
training = 788 (the user's full review of the 1,515 training positive rows),
validation = 784 (the user's review of the 126 validation positive rows), test
= the locked test gold.  788 copied the validation split byte for byte from
the registered bundle, so a run on it chooses its checkpoint against the old
validation answers.  This bundle is 788 with every validation mask replaced by
784's, so a model that validates inside its training loop -- the native
baselines -- chooses against the same answers the protocol reports on.

784's validation gold has 211 components, not the 221 514 was written for, and
514 refuses a bundle whose count moved ("no second correction allowed").  That
refusal is the point of 514's contract, so it is not edited: ``dataset_for``
hands an entry script a 514 whose count is 211, after asserting it was 221,
and only for this bundle's digest.

This script deliberately does NOT: touch a training row, an image or the
positive/clean membership of any row (784 emptied none); modify 788, 784 or
the registered bundle; train, evaluate or select anything; read any model
output; read, transcribe or otherwise consume the test split; or contact the
network.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hme-revision-protocol-bundle-v1'
TRAIN_BUNDLE_SHA = '01784272074cfd089d406e774fb76ebe01c7cbb1affa7527ecd84ae0c47091dd'     # 788
VALIDATION_GOLD_SHA = '9a952dc88468ff051a46726e927735e6061e2f9ece8c6009fadfaf8ee9e52050'  # 784 run_manifest
REGISTERED_DEV_COMPONENTS, PROTOCOL_DEV_COMPONENTS = 221, 211
PROTOCOL_BUNDLE_SHA = 'cbd833ebe4f08c3d8f1398d40bb4f01b8e96875ea599942e1b6455b2e5528baf'   # this bundle's run_manifest


class ProtocolBundleError(RuntimeError):
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


def dataset_for(dataset, bundle_dir):
    """514 as it must read this bundle: validation components 211, asserted from 221."""
    if dataset.DEV_COMPONENTS != REGISTERED_DEV_COMPONENTS:
        raise ProtocolBundleError('514 no longer declares the registered 221 validation components')
    if PROTOCOL_BUNDLE_SHA is None or digest(Path(bundle_dir)/'run_manifest.json') != PROTOCOL_BUNDLE_SHA:
        raise ProtocolBundleError('the 211-component contract applies only to the pinned protocol bundle')
    dataset.DEV_COMPONENTS = PROTOCOL_DEV_COMPONENTS
    return dataset


def validation_masks(records, gold_dir, dataset):
    """784's mask for every validation record, in the bundle's own order."""
    import numpy as np
    manifest = json.loads((Path(gold_dir)/'run_manifest.json').read_text(encoding='utf-8'))
    for name, sha in manifest['files'].items():
        if digest(Path(gold_dir)/name) != sha:
            raise ProtocolBundleError('784 changed: '+name)
    gold = {r['sample_id']: r['index'] for r in json.loads((Path(gold_dir)/'records.json').read_text(encoding='utf-8'))}
    masks = np.load(Path(gold_dir)/'dev_masks_v2.npy', mmap_mode='r')
    out = {}
    for record in records:
        if record['split'] != 'dev':
            continue
        mask = np.asarray(masks[gold[record['sample_id']]], bool)
        if bool(mask.any()) != bool(record['is_positive']):
            raise ProtocolBundleError('784 would change a validation row\'s positive/clean membership')
        out[record['index']] = (mask, len(dataset.instances_from_mask(mask)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--train-bundle', type=Path, required=True, help="788's bundle")
    parser.add_argument('--validation-gold', type=Path, required=True, help="784's output directory")
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        dataset = load('514_revision_native_dataset.py', 'protocol792_514')
        output = args.output_dir.expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise ProtocolBundleError('create-only output exists')
        source = dataset.safe_path(args.train_bundle)
        if digest(source/'run_manifest.json') != TRAIN_BUNDLE_SHA:
            raise ProtocolBundleError("the training bundle is not 788's")
        if digest(args.validation_gold/'run_manifest.json') != VALIDATION_GOLD_SHA:
            raise ProtocolBundleError("the validation gold is not 784's")
        import numpy as np
        dataset.load_bundle(source)   # 788's own contract, read at 221
        records = json.loads((source/'records.json').read_text(encoding='utf-8'))
        masks = np.load(source/'masks.npy', mmap_mode='r', allow_pickle=False)
        replaced = validation_masks(records, args.validation_gold, dataset)
        if len(replaced) != 505 or sum(c for _, c in replaced.values()) != PROTOCOL_DEV_COMPONENTS:
            raise ProtocolBundleError('784 does not give the 505 validation rows 211 components')

        output.mkdir(parents=True)
        fresh = np.array(masks, dtype=bool)
        updated = [dict(r) for r in records]
        for index, (mask, count) in replaced.items():
            fresh[index] = mask
            updated[index]['instance_count'] = count
            updated[index]['validation_gold_784'] = True
        train = [r['index'] for r in records if r['split'] == 'train']
        if not np.array_equal(fresh[train], masks[train]):
            raise ProtocolBundleError('a training mask moved')
        np.save(output/'masks.npy', fresh, allow_pickle=False)
        shutil.copyfile(source/'images.npy', output/'images.npy')
        (output/'records.json').write_text(json.dumps(updated, ensure_ascii=False), encoding='utf-8')
        manifest = json.loads((source/'run_manifest.json').read_text(encoding='utf-8'))
        manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['corrected_dev_components'] = PROTOCOL_DEV_COMPONENTS
        manifest['files'] = {name: {'bytes': (output/name).stat().st_size, 'sha256': digest(output/name)}
                             for name in ('images.npy', 'masks.npy', 'records.json')}
        manifest['protocol_792'] = {'schema_version': SCHEMA, 'train_bundle_sha256': TRAIN_BUNDLE_SHA,
                                    'validation_gold_sha256': VALIDATION_GOLD_SHA,
                                    'validation_components': PROTOCOL_DEV_COMPONENTS,
                                    'training_split_untouched': True, 'protocol': '涂改定位流程.md',
                                    'script_sha256': digest(Path(__file__).resolve()), 'python': platform.python_version()}
        (output/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
        dataset.DEV_COMPONENTS = PROTOCOL_DEV_COMPONENTS
        loaded = dataset.load_bundle(output)
        summary = {'bundle_sha256': digest(output/'run_manifest.json'), 'output_dir': str(output),
                   'train_rows': len(loaded['train']), 'dev_rows': len(loaded['dev']),
                   'dev_components': sum(len(r['instances']) for r in loaded['dev'])}
    except ProtocolBundleError as exc:
        raise SystemExit('792 protocol bundle refused: '+str(exc))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
