# Data contracts

The public repository contains schemas and loaders, not research data. An approved data release must supply matching train/validation and corrected-training bundles. These are frozen research-protocol formats, not a general-purpose COCO training adapter.

## Train/validation bundle

The authoritative loader is `scripts/833_revision_candidate_bundle.py` (`manifest_at`, `load_bundle`, and `validate_records`). It verifies the manifest hash, source hashes, writer separation, row ordering and split membership before returning rows. Images have shape `(3, 96, 768)` with the same grayscale channel repeated. Masks have shape `(96, 768)`; instance arrays have shape `(N, 96, 768)`. Sample/writer identities use UUIDs.

Read the schema in that loader rather than inventing a compatible-looking manifest. The paper's bundle includes validation annotation v2 and corrected training data. The training path refuses validation or test writers in the corrected pool. Do not disable these checks to make another dataset pass.

## Corrected training bundle

`scripts/853_revision_generalization_arms.py:pseudo_rows` validates the corrected schema and its manifest digest, then reads `images.npy`, `masks.npy`, and row metadata. The paper adds 2,127 corrected training rows to the 3,808-row training pool. These private arrays are not part of this release.

## Frozen operating point

`revrender.py freeze` writes a record containing the validation bundle hash, checkpoint hash, mask threshold, confidence threshold and mask-NMS setting. Its `test_touched` field must be false. Paths in a freeze record refer to the user's own local assets; update path strings when moving a run, keeping all content hashes and selected thresholds unchanged.

## Standalone inference

A grayscale `uint8` array of shape `(96, 768)` is sufficient as the image input, together with the separately supplied model checkpoint, backbone file and freeze record. Predictions are returned as boolean masks, scores and query indices. Empty predictions are valid. No student example is needed to run the synthetic unit tests.
