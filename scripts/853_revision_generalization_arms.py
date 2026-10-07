"""Generalization arms on morphology_s42's point_context, seed 42: what 831/842/846 point at.

831 and 846 located the gap on unseen writers: point_context scores F1 0.96 on
its own training rows and 0.63-0.67 on validation writers; its candidate pool
covers 98.9% of training targets and 78.3% of validation's; and every scorer
trained on its in-sample pools (805, 844-846) learned errors too easy to
transfer.  This entry trains point_context exactly as 834 does (816's
architecture, the baselines' R50, seed 42, patience-6 early stopping from
14150, the same gates and freeze) under one declared variant:

  fold0..fold3  the training pool without one quarter of the training writers
                (a fixed hash-ordered, positive-balanced assignment).  855 later
                predicts the held-out quarter with this model, so every training
                line gets a candidate pool drawn by a model that never saw its
                writer -- the realistic errors a second stage has to learn from
  writer        positive and clean pools each cycled by writer, then by that
                writer's own rows (851's WriterCycle): the ten writers holding
                15.6% of positive rows stop dominating updates; no row is dropped
  style         after 357's augmentation (whose random stream is left exactly as
                it was), writer-style perturbation from an independent generator:
                slant shear +-15 degrees (p .5), a smooth elastic warp (p .3),
                stroke width by a 2x2 grey erosion or dilation (p .3) and ink
                gamma 0.7-1.4 (p .3); geometry moves image and mask together.
                None of these is in 357 or in 780's strong dose
  pseudo        854's rows added to the pools: lines a person marked revised,
                from training writers and from writers in no split, with the
                frozen point_context's confident masks as their targets; and
                lines a person marked clean, from writers in no split, with
                empty targets.  Validation and test writers are refused here too

  corrected     as pseudo, with 861's rows instead: the same lines after the user
                corrected them by hand (2026-09-27) -- 1,081 revised lines with the
                user's masks, the lines they emptied as clean, 854's clean lines of
                writers in no split; twelve lines the user rejected are absent
  corrected_pos corrected's revised lines only, none of its clean lines: next to
                corrected it separates what the positives bring from the clean lines
  control       nothing changed: a smoke run of it must equal 837's point_context
                smoke trace value for value, which proves the wiring above leaves
                the recipe alone; it is never trained formally

Everything else is point_context unchanged: matcher, losses, heads, schedule,
budget, gates and the 835 freeze.

This script deliberately does NOT: read the frozen common test; admit a
validation-writer or test-writer line into any pool; change 675, 816 or 834;
resume a run; or contact the network.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('control', 'fold0', 'fold1', 'fold2', 'fold3', 'writer', 'style', 'pseudo', 'corrected', 'corrected_pos')
FOLDS = 4
FOLD_SALT = '853fold'
# Lists, not tuples: the profile is compared after a JSON round trip.
STYLE = {'shear_p': .5, 'shear_degrees': 15., 'elastic_p': .3, 'elastic_alpha': [1.5, 3.], 'elastic_sigma': 6.,
         'stroke_p': .3, 'gamma_p': .3, 'gamma': [.7, 1.4]}
PSEUDO_SCHEMA = 'hme-revision-pseudo854-v1'
CORRECTED_SCHEMA = 'hme-revision-corrected861-v1'
EXTRA_SCHEMAS = {'pseudo': PSEUDO_SCHEMA, 'corrected': CORRECTED_SCHEMA, 'corrected_pos': CORRECTED_SCHEMA}


def load(name):
    spec = importlib.util.spec_from_file_location('arms853_' + name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def writer_folds(positive, clean, folds=FOLDS):
    """Writer -> fold: largest positive writers first, each to the fold with the fewest positive rows."""
    counts = {}
    for index, rows in enumerate((positive, clean)):
        for row in rows:
            require(row.get('split') == 'train', 'folds are drawn over training rows only')
            counts.setdefault(row['group_uuid'], [0, 0])[index] += 1
    order = sorted(counts, key=lambda w: (-counts[w][0], -counts[w][1],
                                          hashlib.sha256((FOLD_SALT+w).encode()).hexdigest()))
    load_ = [[0, 0] for _ in range(folds)]
    assignment = {}
    for writer in order:
        fold = min(range(folds), key=lambda i: (load_[i][0], load_[i][1], i))
        assignment[writer] = fold
        load_[fold][0] += counts[writer][0]
        load_[fold][1] += counts[writer][1]
    return assignment


def held_out_ids(rows, assignment, fold):
    return sorted(r['sample_id'] for r in rows if assignment[r['group_uuid']] == fold)


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


class WriterCycle:
    """851's writer-then-row cycle, exposing the writer cycle as the sampler state 655 snapshots."""

    def __init__(self, rows, rng, cycle_type):
        groups = {}
        for row in rows:
            require(row.get('split') == 'train', 'writer sampler accepts train only')
            groups.setdefault(row['group_uuid'], []).append(row)
        require(len(groups) >= 4, 'four distinct writers per polarity required')
        self.rows, self.rng = rows, rng
        self.writers = cycle_type([{'writer': w} for w in sorted(groups)], rng)
        self.cycles = {w: cycle_type(groups[w], rng) for w in sorted(groups)}

    @property
    def order(self):
        return self.writers.order

    @property
    def position(self):
        return self.writers.position

    def take_distinct(self, count):
        require(0 <= count <= len(self.writers.rows), 'writer request exceeds pool')
        picked = []
        while len(picked) < count:
            cycle = self.writers
            if cycle.position >= len(cycle.order):
                order = list(range(len(cycle.rows)))
                self.rng.shuffle(order)
                # Delay repeats at a cycle boundary; never discard an exposure.
                chosen = {r['writer'] for r in picked}
                cycle.order = ([i for i in order if cycle.rows[i]['writer'] not in chosen]
                               + [i for i in order if cycle.rows[i]['writer'] in chosen])
                cycle.position = 0
            picked.append(cycle.rows[cycle.order[cycle.position]])
            cycle.position += 1
        return [self.cycles[r['writer']].take(1)[0] for r in picked]


def style_generator(rng):
    """Seeded by the stream's state after 357's call; drawing from it leaves the stream alone."""
    value = hashlib.sha256(('853style'+repr(rng.getstate())).encode()).digest()
    return random.Random(int.from_bytes(value[:8], 'big'))


def shear(picture, target, degrees):
    """Slant about the row's vertical centre: x_in = x + k (y - cy)."""
    import numpy as np
    from PIL import Image
    height, width = picture.shape
    k = math.tan(math.radians(degrees))
    matrix = (1., k, -k*height/2., 0., 1., 0.)
    warp = lambda a, resample, fill: np.asarray(Image.fromarray(a).transform(
        (width, height), Image.Transform.AFFINE, matrix, resample=resample, fillcolor=fill), np.uint8).copy()
    return warp(picture, Image.Resampling.BILINEAR, 255), warp(target, Image.Resampling.NEAREST, 0)


def elastic(picture, target, generator):
    import numpy as np
    from scipy import ndimage
    source = np.random.default_rng(generator.getrandbits(63))
    alpha = generator.uniform(*STYLE['elastic_alpha'])
    height, width = picture.shape
    fields = [ndimage.gaussian_filter(source.uniform(-1, 1, (height, width)), STYLE['elastic_sigma'], mode='constant')
              for _ in range(2)]
    fields = [f*(alpha/max(1e-6, float(np.abs(f).max()))) for f in fields]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    coords = [y+fields[0], x+fields[1]]
    warped = ndimage.map_coordinates(picture.astype(np.float32), coords, order=1, mode='constant', cval=255.)
    moved = ndimage.map_coordinates(target, coords, order=0, mode='constant', cval=0)
    return np.clip(np.rint(warped), 0, 255).astype(np.uint8), moved.astype(np.uint8)


def stroke(picture, generator):
    from scipy import ndimage
    operation = ndimage.grey_erosion if generator.random() < .5 else ndimage.grey_dilation
    return operation(picture, size=(2, 2))


def gamma(picture, value):
    import numpy as np
    return np.clip(np.rint(255.*(picture.astype(np.float32)/255.)**value), 0, 255).astype(np.uint8)


def install_style(extension):
    original = extension.paired_affine_with_aux

    def styled(image, mask, skeleton, aux, rng):
        picture, target, centreline, auxiliary = original(image, mask, skeleton, aux, rng)
        generator = style_generator(rng)
        if generator.random() < STYLE['shear_p']:
            picture, target = shear(picture, target, generator.uniform(-STYLE['shear_degrees'], STYLE['shear_degrees']))
        if generator.random() < STYLE['elastic_p']:
            picture, target = elastic(picture, target, generator)
        if generator.random() < STYLE['stroke_p']:
            picture = stroke(picture, generator)
        if generator.random() < STYLE['gamma_p']:
            picture = gamma(picture, generator.uniform(*STYLE['gamma']))
        return picture, target, centreline, auxiliary

    extension.paired_affine_with_aux = styled
    return dict(STYLE, stream='advanced exactly as point_context (357 is called first)')


def excluded_writers(bundle_dir):
    """Validation and test writers of the registered 832 mapping, for refusal only."""
    mapping = json.loads((Path(bundle_dir)/'index_manifest.json').read_text())
    return {r['group_uuid'] for r in mapping['rows'] if r['split'] in ('dev', 'test')}


def pseudo_rows(path, expected_sha, excluded, schema=PSEUDO_SCHEMA):
    """854's rows as 655 pool rows; refused if any belongs to a validation or test writer."""
    import numpy as np
    root = Path(path).resolve()
    require(digest(root/'run_manifest.json') == expected_sha, 'pseudo-label manifest changed')
    manifest = json.loads((root/'run_manifest.json').read_text())
    require(manifest['schema'] == schema and manifest['test_touched'] is False, 'not a '+schema+' bundle')
    for name, sha in manifest['files'].items():
        require(Path(name).name == name and digest(root/name) == sha, 'pseudo-label file changed: '+name)
    records = json.loads((root/'records.json').read_text())
    images = np.load(root/'images.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(root/'masks.npy', mmap_mode='r', allow_pickle=False)
    require(images.shape == masks.shape == (len(records), 96, 768), 'pseudo-label arrays do not match records')
    require(not ({r['group_uuid'] for r in records} & excluded), 'a validation or test writer is in the pseudo-label pool')
    rows = {'fit_positive': [], 'fit_clean': []}
    for i, record in enumerate(records):
        mask = np.asarray(masks[i], bool)
        require(bool(mask.any()) == (record['pool'] == 'fit_positive'), 'pseudo-label pool disagrees with its mask')
        rows[record['pool']].append({'sample_id': 'pseudo854:'+record['line_id'], 'line_id': record['line_id'],
                                     'group_uuid': record['group_uuid'], 'split': 'train', 'legacy_split': record['pool'],
                                     'is_positive': record['pool'] == 'fit_positive', 'source': record['source'],
                                     'image': np.asarray(images[i]), 'mask': mask.astype(np.uint8)*255,
                                     'line_skeleton': np.zeros(mask.shape, np.uint8)})
    return rows, {'bundle_sha256': expected_sha, 'positive': len(rows['fit_positive']), 'clean': len(rows['fit_clean']),
                  'writers': len({r['group_uuid'] for r in records}), 'policy': manifest['policy']}


def apply_variant(variant, data, bundle_dir, pseudo=None):
    """Change the training pools as the variant declares; return what changed."""
    positive, clean = data['fit_positive'], data['fit_clean']
    train_lines = {r['line_id'] for r in positive+clean}
    if variant.startswith('fold'):
        fold = int(variant[4:])
        assignment = writer_folds(positive, clean)
        held = held_out_ids(positive+clean, assignment, fold)
        data['fit_positive'] = [r for r in positive if assignment[r['group_uuid']] != fold]
        data['fit_clean'] = [r for r in clean if assignment[r['group_uuid']] != fold]
        return {'fold': fold, 'folds': FOLDS, 'salt': FOLD_SALT,
                'held_out_writers': sum(v == fold for v in assignment.values()), 'held_out_rows': len(held),
                'held_out_positive': sum(assignment[r['group_uuid']] == fold for r in positive),
                'held_out_sample_ids_sha256': ids_digest(held),
                'training_rows': len(data['fit_positive'])+len(data['fit_clean'])}
    if variant in EXTRA_SCHEMAS:
        require(pseudo is not None, variant+' requires its extra rows')
        rows, info = pseudo
        for pool in (('fit_positive',) if variant == 'corrected_pos' else ('fit_positive', 'fit_clean')):
            require(not ({r['line_id'] for r in rows[pool]} & train_lines), 'pseudo rows collide with training rows')
            data[pool] = data[pool]+sorted(rows[pool], key=lambda r: r['line_id'])
        return dict(info, training_rows=len(data['fit_positive'])+len(data['fit_clean']))
    if variant == 'writer':
        return {'positive_writers': len({r['group_uuid'] for r in positive}),
                'clean_writers': len({r['group_uuid'] for r in clean})}
    return {}


def ours(picked, rest, data_code, architecture_file='816_revision_alignment_refinement.py', extra_files=()):
    """Train one variant; 866 hands in its own architecture file (which must build on 816)."""
    os.environ['HME_ALIGN_REFINE'] = 'point_context'
    architecture = load('816_revision_alignment_refinement.py')
    train = architecture.COARSE.RESNET.repoint(load('675_revision_region_point_train.py'), architecture.COARSE.RESNET.MIT_B0_SHA)
    excluded = excluded_writers(picked.bundle)
    pseudo = (pseudo_rows(picked.pseudo_bundle, picked.pseudo_sha, excluded, EXTRA_SCHEMAS[picked.variant])
              if picked.variant in EXTRA_SCHEMAS else None)
    record = {'variant': picked.variant}
    original = train.load

    def loader(name):
        module = original(name)
        if name.startswith('655'):
            require(module.BUNDLE_SHA == 'd9cacdf88844768cf415e1b954ac4234f7908bb6c5896f6cbe1961da6f54f250',
                    'unexpected original trainer data contract')
            module.BUNDLE_SHA = picked.bundle_sha
            rows_of, stream_of = module.make_rows, module.make_stream

            def make_rows(bundle):
                data = rows_of(bundle)
                require(not ({r['group_uuid'] for r in data['fit_positive']+data['fit_clean']} & excluded),
                        'a validation or test writer is in the training pool')
                record.update(apply_variant(picked.variant, data, picked.bundle, pseudo))
                return data

            def make_stream(fast, base, extension, data, args, rng):
                stream = stream_of(fast, base, extension, data, args, rng)
                if picked.variant == 'writer':
                    stream.positive_cycle = WriterCycle(data['fit_positive'], rng, base.ShuffledCycle)
                    stream.clean_cycle = WriterCycle(data['fit_clean'], rng, base.ShuffledCycle)
                    stream.consumed_state = stream._snapshot()
                return stream

            module.make_rows, module.make_stream = make_rows, make_stream
        if name.startswith('514'):
            data_code.dataset_for(module, picked.bundle, picked.bundle_sha)
        if name.startswith('357') and picked.variant == 'style':
            record['style'] = install_style(module)
        if name == architecture_file:
            configuration = module.configuration
            # What the variant is enters the profile, so gate evidence of one
            # variant can never pass for another's.
            module.configuration = lambda arm: dict(configuration(arm), generalization853={
                'variant': picked.variant, 'pseudo_sha256': picked.pseudo_sha if picked.variant in EXTRA_SCHEMAS else None,
                'style': STYLE if picked.variant == 'style' else None, 'folds': FOLDS, 'fold_salt': FOLD_SALT})
        return module

    train.load = loader
    args = train.parse(rest + ['--bundle', str(picked.bundle)], arms=architecture.ARMS)
    require(args.seed == 42 and args.early_stop_patience == 6 and args.speed_v1,
            'requires seed42, patience6 and the verified speed path')
    require(args.augment == 'current', 'the variant, not 780, decides augmentation')
    train.run(args, architecture_file=architecture_file, extra_dependencies=(
        '685_revision_mask_iou_score.py', '687_revision_region_point_maskiou.py',
        '750_revision_resnet50_base.py', '767_revision_coarse_mask.py', '775_revision_mechanism_trials.py',
        '816_revision_alignment_refinement.py', '833_revision_candidate_bundle.py', '834_revision_candidate_train.py',
        Path(__file__).name, *extra_files))
    variant_file = args.output_dir/'variant853.json'
    variant_file.write_text(json.dumps(record, indent=1, sort_keys=True)+'\n')
    print(json.dumps({'variant': picked.variant, 'stage': args.stage, **{k: v for k, v in record.items() if k != 'style'}},
                     sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--pseudo-bundle', type=Path)
    parser.add_argument('--pseudo-sha')
    picked, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    require((picked.variant in EXTRA_SCHEMAS) == (picked.pseudo_bundle is not None and picked.pseudo_sha is not None),
            'the extra-row bundle goes with the pseudo and corrected variants and only with them')
    data_code = load('833_revision_candidate_bundle.py')
    data_code.manifest_at(picked.bundle, picked.bundle_sha)
    ours(picked, rest, data_code)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
