# ARIC offline global-feature baseline

The task is **image-level multilabel behavior presence** using existing shallow global visual features. A behavior is counted once per image even when multiple people exhibit it.

This does not reproduce ARIC's original individual-feature method and is not person detection, per-person classification, attendance counting, or engagement modeling. Individual features are missing. Exact encoder weights, shallow output layer, normalization, and spatial token order of the global features remain unverified. Do not apply this head directly to RGB or classroom video. `contract.json` fixes `offline_research_only` and `raw_rgb_compatible=false`.

Inputs come only from local `ARIC_supplement/ARIC_CLIP_global_view_features.tar.gz`. Inspected `.npy` arrays were finite `float32` with shape `(1,401,768)`. Mean pooling over all 401 tokens followed by L2 normalization yields 768 dimensions; no 20×20 grid is assumed. `numpy.load(..., allow_pickle=False)` prevents object deserialization. Tar streaming avoids extracting the 41.5 GB archive, caching only selected reduced vectors.

Targets come only from `annotations/*.json` `instances[].label`. Captions, class names in instance paths, boxes, audio, and original train/test index labels are not classifier inputs. Old indexes are audited only for original-image overlap.

## Split and evaluation

- Select 2,048 images by default from all annotation IDs using seed/SHA-256 sorting, rather than consecutive archive entries.
- Resplit by original image ID: 60% train, 20% validation, 20% test, with no ID overlap. Missing lesson/video IDs prevent claims of cross-classroom independence; adjacent images and the same room may cross sets.
- Select classes using training positives/negatives only, at least 15 of each by default. `metrics.json` includes excluded classes and split support for retained classes.
- Fit dimension-wise means/standard deviations and class weights on training data only. Train independent logistic heads with Adam and L2; class weights are capped at 10.
- Train 240 epochs, choosing minimum validation BCE. Choose per-class thresholds by validation F1 on 0.10–0.90 in 0.05 steps. Use 0.5 when validation has no positives or no negatives. Evaluate the test set only at the end.
- Report Macro-F1, Micro-F1, mAP, exact match, Hamming loss, per-class precision/recall/F1/AP, and `tn/fp/fn/tp`. AP is `null` without both positive/negative support; report the class count contributing to mAP.
- Compare training-frequency, all-negative, and all-positive baselines. Frequency uses threshold 0.5 by default; all-positive shows constant-prediction F1 behavior. AP for constant frequency scores equals test-label prevalence. Also retain model results at a fixed 0.5 threshold.
- Class-weighted sigmoid outputs are uncalibrated scores, not independently calibrated behavior probabilities.

## Reproduce the baseline

Dependencies are Python and NumPy; tests additionally need pytest. Artifacts record the actual environment/timing. The small CPU head does not require a GPU. From `backend/`:

```powershell
python scripts/train_aric.py `
  --dataset-root './local-datasets' `
  --output artifacts/learning/aric --max-images 2048 --seed 41 --epochs 240
& .\.venv-media\Scripts\python.exe -m pytest tests/test_aric_learning.py -q
```

The first cache build scans sequentially because gzip lacks random indexing. Reuse verifies image IDs, resolved absolute archive path, size, nanosecond modification time, reduction method, and tensor shape before overwriting audit files. Any mismatch requires a new output directory. This detects ordinary replacement; it is not a cryptographic digest of the full 41.5 GB archive. Use separate outputs when changing source, sampling, or seed.

A missing source archive cannot bypass verification through the cache path. Use `--from-bundle` for an explicitly exported portable bundle without the archive. Copy `learning/aric.py`, `learning/__init__.py`, `scripts/train_aric.py`, `training_bundle.npz`, and adjacent `contract.json` to a NumPy environment:

```bash
python scripts/train_aric.py --from-bundle /path/to/training_bundle.npz --output /path/to/rerun --seed 41 --epochs 240
```

The bundle contains reduced features, IDs, labels, and explicit split row indexes. Duplicate, overlapping, missing indexes or duplicate image IDs are rejected. Another encoder's 768-dimensional vectors are not interchangeable inputs.

## Outputs and prediction

Default location: `artifacts/learning/aric/`.

| File | Contents |
| --- | --- |
| `data_audit.json` | Scale, class support, old-index image overlap |
| `split_manifest.json` | Selected IDs, targets, annotation SHA-256, split assignments |
| `features.npz` / `features.json` | 768-dimensional cache; source member hashes, sizes, paths |
| `training_bundle.npz` | Portable features, all candidate classes, split row indexes |
| `preparation.json` | Preparation/cache timing, cache/bundle hashes, environment |
| `model.npz` | Weights, biases, training normalization, classes, validation thresholds |
| `contract.json` | Input contract, task, known/unknown preprocessing, prohibition on unverified RGB use |
| `metrics.json` | Results, baselines, support, configuration, CPU/timing, model hash, export paths |
| `test_predictions.npz` | Test IDs, targets, scores, thresholds, classes |

```python
import numpy as np
from learning.aric import load_head, predict_reduced

model = load_head('artifacts/learning/aric/model.npz')
with np.load('artifacts/learning/aric/features.npz', allow_pickle=False) as cache:
    scores = predict_reduced(model, cache['features'][:2])
present = scores >= model['thresholds']
```

Cache/model reload checks reproducibility for the same reduced inputs, not new-video inference, image encoding, or cross-lesson generalization. Competition display and redistribution rights require separate confirmation.
