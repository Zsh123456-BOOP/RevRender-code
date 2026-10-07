"""C832: reuse 675 point_context / 793 native recipes and early stopping.

Only the data contract changes. point_context/MaskDINO use torchvision R50;
PointRend retains its native MSRA R50 pretraining. No old trained checkpoints,
no new test rows. Their existing
normalization, optimizer groups, losses and checkpoint tie-breakers stay intact.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('candidate834_' + name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ours(picked, rest, data):
    os.environ['HME_ALIGN_REFINE'] = 'point_context'
    architecture = load('816_revision_alignment_refinement.py')
    train = architecture.COARSE.RESNET.repoint(load('675_revision_region_point_train.py'), architecture.COARSE.RESNET.MIT_B0_SHA)
    original = train.load
    def loader(name):
        module = original(name)
        if name.startswith('655'):
            if module.BUNDLE_SHA != 'd9cacdf88844768cf415e1b954ac4234f7908bb6c5896f6cbe1961da6f54f250':
                raise ValueError('unexpected original trainer data contract')
            module.BUNDLE_SHA = picked.bundle_sha
        if name.startswith('514'):
            data.dataset_for(module, picked.bundle, picked.bundle_sha)
        return module
    train.load = loader
    args = train.parse(rest + ['--bundle', str(picked.bundle)], arms=architecture.ARMS)
    if args.seed != 42 or args.early_stop_patience != 6 or not args.speed_v1:
        raise ValueError('requires seed42, patience6 and verified speed path')
    train.run(args, architecture_file='816_revision_alignment_refinement.py', extra_dependencies=(
        '685_revision_mask_iou_score.py', '687_revision_region_point_maskiou.py',
        '750_revision_resnet50_base.py', '767_revision_coarse_mask.py', '775_revision_mechanism_trials.py',
        '833_revision_candidate_bundle.py', Path(__file__).name))


def native(picked, rest, data):
    train = load('793_revision_native_protocol_train.py')
    original = train.load
    adapter = SimpleNamespace(__file__=data.__file__, dataset_for=lambda ds, path: data.dataset_for(ds, path, picked.bundle_sha))
    def loader(name, key):
        if name == '798_revision_split_b.py':
            return adapter
        module = original(name, key)
        if name == '516_revision_native_instance_train.py':
            write = module.write_json
            def write_json(path, value):
                if Path(path).name == 'run_manifest.json':
                    value['source_sha256'][str(Path(__file__))] = data.digest(Path(__file__))
                    value['candidate_dataset'] = data.manifest_at(picked.bundle, picked.bundle_sha)['candidate']
                    value['new_test_accessed'] = False
                return write(path, value)
            module.write_json = write_json
        return module
    train.load = loader
    if picked.model not in ('maskdino', 'pointrend'):
        raise ValueError('undeclared native model')
    args = train.parser().parse_args(rest + ['--dataset-dir', str(picked.bundle), '--method', picked.model])
    if args.seed != 42 or args.microbatch != 2:
        raise ValueError('preserve native seed42/microbatch2 recipe')
    args.split = 'C832'
    train.GOLDS['C832'] = ('split_c832', 'split_c832')
    train.run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--model', choices=('point_context', 'maskdino', 'pointrend'), required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    picked, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    data = load('833_revision_candidate_bundle.py')
    data.manifest_at(picked.bundle, picked.bundle_sha)
    (ours if picked.model == 'point_context' else native)(picked, rest, data)


if __name__ == '__main__':
    main()
