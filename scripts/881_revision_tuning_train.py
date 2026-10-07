"""Train 880's tuning mode on the corrected data, seed 42, through 853's verified path.

The data are 853's ``corrected`` variant (the 3,808 training lines plus 861's
2,127 corrected rows) on whatever train/dev bundle is handed in -- the tuning
campaign uses 878's bundle with validation gold v2, so in-loop checkpoint
choice and early stopping read gold v2.  The architecture is 880 in the declared
``--tune`` mode; recipe, gates, early stopping and the speed path are 853's and
675's.  ``--tune control`` on 878's bundle must reproduce the corrected run's
training steps value for value (only validation differs).

This script deliberately does NOT: change 853's data handling or 675's loop;
admit a validation or test writer; read the frozen common test; or contact the
network.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ('control', 'bb01', 'wd1e3', 'lr05', 'frozenbn', 'drop01', 'ssl', 'nofine', 'nocontext', 'nodense', 'nofine_nodense')


def load(name):
    spec = importlib.util.spec_from_file_location('tune881_' + name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--tune', choices=MODES, required=True)
    parser.add_argument('--ssl-checkpoint', type=Path)
    parser.add_argument('--ssl-sha')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--corrected-bundle', type=Path, required=True)
    parser.add_argument('--corrected-sha', required=True)
    picked, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if (picked.tune == 'ssl') != bool(picked.ssl_checkpoint and picked.ssl_sha):
        raise SystemExit('881 refused: --tune ssl needs --ssl-checkpoint and --ssl-sha, and only it')
    # 880 reads its mode at import; set it before anything loads it.
    os.environ['HME_TUNE'] = picked.tune
    if picked.tune == 'ssl':
        os.environ['HME_TUNE_SSL'], os.environ['HME_TUNE_SSL_SHA'] = str(picked.ssl_checkpoint.resolve()), picked.ssl_sha
    arms = load('853_revision_generalization_arms.py')
    data_code = load('833_revision_candidate_bundle.py')
    data_code.manifest_at(picked.bundle, picked.bundle_sha)
    variant = argparse.Namespace(variant='corrected', bundle=picked.bundle, bundle_sha=picked.bundle_sha,
                                 pseudo_bundle=picked.corrected_bundle, pseudo_sha=picked.corrected_sha)
    arms.ours(variant, rest, data_code, architecture_file='880_revision_tuning.py', extra_files=(Path(__file__).name,))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
