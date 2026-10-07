"""Augmentation dose for the r50_ctrl training stream: off, current, strong.

The same model scores F1 0.885 on training rows and 0.607 on dev rows, and
every failure stage drops together when it leaves the rows it trained on.  The
only augmentation it has ever seen is 357's paired affine: with probability 0.8
a rotation of +-3 degrees, scale +-6 percent, shift +-4 percent, contrast and
brightness +-12 percent, a 20 percent chance of a light blur and a 15 percent
chance of JPEG.  One hypothesis is that even this hurts (the model trains on
warped rows and is read on unwarped ones); the other is that it is far too weak
for two thousand training instances.  Reading three doses on one seed answers
both, and a check made before this was written already bounds the first: over
4,545 augmented draws of the training rows the instance structure the loss
sees changed in 1.0 percent (19 merges, 8 splits, 10 pushed out of frame).

Both new doses call 357's function first, so the sampler's random stream --
batch composition, order and every later draw -- is r50_ctrl's exactly, and the
arms differ from revision_r50_ctrl_s42 only in what the rows look like:

  off     the stream advances as in current, and the unaugmented row is returned.
  strong  current's output, then extra operations whose generator is seeded by a
          digest of the stream state after the call (deterministic, replayable,
          and it never touches the stream):
            copy-paste (p .5, positive rows only): the box around one training
              revision component, with the ink around it, is laid by darkest-
              pixel compositing into a blank span of the row at its own height;
              the pasted components join the mask (Ghiasi et al., CVPR 2021);
            a second paired affine: rotation +-4 degrees, scale 0.85-1.15;
            stroke width (p .3): grey erosion or dilation by one pixel, image only;
            Gaussian noise (p .2), sigma 2-8 grey levels, image only.

This script deliberately does NOT: change the model, the losses, the sampler,
the schedule, the budget or the gate (the learning gate reads fixed rows with no
augmentation, so it is the same for every dose); paste into clean rows; paste
over ink or over another revision; draw pastes from dev rows; read, transcribe
or otherwise consume the frozen test split; or contact the network.
"""
import hashlib
import math
import random

import numpy as np

DOSES = ('current', 'off', 'strong')
PASTE_P, STROKE_P, NOISE_P = .5, .3, .2
PASTE_TRIES, PASTE_MARGIN, PASTE_MAX_WIDTH = 8, 4, 320
BLANK_LEVEL, BLANK_SHARE = 200, .97


def stream_generator(rng):
    """A generator determined by the stream's state; drawing from it leaves the stream alone."""
    digest = hashlib.sha256(repr(rng.getstate()).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def paste_pool(rows):
    """(image patch, mask patch, top) for each training revision component, with margin."""
    from scipy import ndimage
    pool = []
    for row in rows:
        mask = np.asarray(row['mask']) > 0
        if not mask.any():
            continue
        image = np.asarray(row['image'])
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
        for box in ndimage.find_objects(labels):
            top, bottom = max(0, box[0].start-PASTE_MARGIN), min(mask.shape[0], box[0].stop+PASTE_MARGIN)
            left, right = max(0, box[1].start-PASTE_MARGIN), min(mask.shape[1], box[1].stop+PASTE_MARGIN)
            if right-left > PASTE_MAX_WIDTH:
                continue
            pool.append((image[top:bottom, left:right].copy(), mask[top:bottom, left:right].copy(), top))
    return pool


def copy_paste(image, mask, pool, generator):
    """Lay one pool patch into a blank span at its own height; unchanged if none fits."""
    if not pool or not (mask > 0).any():
        return image, mask
    patch, patch_mask, top = pool[generator.randrange(len(pool))]
    height, width = patch.shape
    if top+height > image.shape[0] or width >= image.shape[1]:
        return image, mask
    occupied = mask > 0
    for _ in range(PASTE_TRIES):
        left = generator.randrange(image.shape[1]-width)
        region = (slice(top, top+height), slice(max(0, left-2), left+width+2))
        if occupied[region].any() or (image[region] >= BLANK_LEVEL).mean() < BLANK_SHARE:
            continue
        image, mask = image.copy(), mask.copy()
        window = (slice(top, top+height), slice(left, left+width))
        image[window] = np.minimum(image[window], patch)
        mask[window] = np.where(patch_mask, 255, mask[window]).astype(mask.dtype)
        return image, mask
    return image, mask


def second_affine(image, mask, generator):
    from PIL import Image
    height, width = image.shape
    angle, scale = generator.uniform(-4., 4.), generator.uniform(.85, 1.15)
    radians = math.radians(angle)
    cosine, sine = math.cos(radians)/scale, math.sin(radians)/scale
    cx, cy = width/2., height/2.
    matrix = (cosine, sine, cx-cosine*cx-sine*cy, -sine, cosine, cy+sine*cx-cosine*cy)

    def warp(array, resample, fill):
        return np.asarray(Image.fromarray(array).transform((width, height), Image.Transform.AFFINE,
                                                            matrix, resample=resample, fillcolor=fill))
    return (warp(image, Image.Resampling.BILINEAR, 255).copy(),
            warp(mask, Image.Resampling.NEAREST, 0).copy())


def stroke_and_noise(image, generator):
    from scipy import ndimage
    if generator.random() < STROKE_P:
        image = (ndimage.grey_erosion if generator.random() < .5 else ndimage.grey_dilation)(image, size=(2, 2))
    if generator.random() < NOISE_P:
        noise = np.random.default_rng(generator.getrandbits(32)).normal(0., generator.uniform(2., 8.), image.shape)
        image = np.clip(image.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    return image


def install(extension, dose, train_rows):
    """Replace 357's paired affine on this loaded module with the declared dose."""
    if dose not in DOSES:
        raise ValueError('undeclared augmentation dose')
    if dose == 'current':
        return {'dose': 'current'}
    original = extension.paired_affine_with_aux
    pool = paste_pool(train_rows) if dose == 'strong' else []

    def dosed(image, mask, skeleton, aux, rng):
        result = original(image, mask, skeleton, aux, rng)
        if dose == 'off':
            return (np.asarray(image, np.uint8).copy(), np.asarray(mask, np.uint8).copy(),
                    np.asarray(skeleton, np.uint8).copy(), aux)
        picture, target, centreline, auxiliary = result
        generator = stream_generator(rng)
        if generator.random() < PASTE_P:
            picture, target = copy_paste(picture, target, pool, generator)
        picture, target = second_affine(picture, target, generator)
        picture = stroke_and_noise(picture, generator)
        return picture, target, centreline, auxiliary

    extension.paired_affine_with_aux = dosed
    return {'dose': dose, 'paste_pool': len(pool), 'stream': 'advanced exactly as current (357 is called first)'}
