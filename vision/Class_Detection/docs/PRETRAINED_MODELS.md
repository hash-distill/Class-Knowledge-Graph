# Pretrained weights and provenance

The three-class entry downloads existing weights without local training. `configs/pretrained_models.json` pins files. The downloader uses only the Python standard library and neither deserializes models nor starts inference. Sources/digests were checked on 2026-10-25; this is a historical provenance record, not a claim that the weights are bundled.

## Pinned files

Paths are relative to `vision/Class_Detection/`.

| Key | Local file | Pinned source | Bytes |
| --- | --- | --- | ---: |
| `behavior_scb` | `artifacts/models/scb_yolov9e.pt` | Author-linked Hugging Face Space, commit `dd6c9381099400c4c64135335cb531bdad35ed1d`, original `best.pt` | 117275802 |
| `pose_coco` | `artifacts/models/yolo26n-pose.pt` | Official Ultralytics assets release `v8.4.0`, asset `340096236` | 7878574 |

```text
behavior_scb  4ff890b66e69ed809f7bcbfa56eb4348404a1298457d772d728291d4027f6561
pose_coco     eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9
```

Behavior provenance follows the deployment link in [the author's repository](https://github.com/nontaphatfirm/Classroom-Behavior-Detector) to [Student-Behavior-Detector](https://huggingface.co/spaces/nontaphat/Student-Behavior-Detector). The project pins a commit rather than `main`; size/SHA-256 come from its [Git LFS pointer](https://huggingface.co/spaces/nontaphat/Student-Behavior-Detector/raw/dd6c9381099400c4c64135335cb531bdad35ed1d/best.pt). Renaming to `scb_yolov9e.pt` changes no bytes. Internal architecture is `yolov9e.yaml`, hence YOLOv9e.

Pose provenance is [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0), matching the [release-asset API](https://api.github.com/repos/ultralytics/assets/releases/assets/340096236) size/hash. Tags can change, so retain/check the fixed digest too.

## Download and verify

From `vision/Class_Detection/`:

```powershell
python scripts/fetch_pretrained.py
python scripts/fetch_pretrained.py --verify-only
python scripts/fetch_pretrained.py --model pose_coco
python scripts/fetch_pretrained.py --model behavior_scb --repair
```

- Default output is project-local `artifacts/models`, independent of the caller's working directory; override `--output-dir`.
- Existing files undergo full size/SHA-256 checks every time. Invalid same-name files fail instead of being trusted or silently overwritten.
- Downloads use a unique same-directory `.part` and incremental hashes. Atomic publication follows exact size/hash validation. Interruptions/mismatches clean up that attempt's temporary file.
- `--repair` replaces an old file only after full validation; failed downloads preserve it for diagnosis. Invalid old files remain unusable.
- `--verify-only` is offline and exits nonzero for missing/corrupt files.
- No automatic Ultralytics download or alternate model/class fallback is used. Integrity is not target-classroom accuracy.

Offline tests:

```powershell
python -m unittest discover -s tests -p test_model_download.py
```

In-memory responses/temporary files cover wrong sizes/hashes, interruption cleanup, cached checks, and preservation during failed repair. Tests do not download real weights.

## Model meaning and historical checks

Validated PyTorch ZIP files were statically inspected with `zipfile` and `pickletools`, without executing `data.pkl`:

| Model | Internal configuration | Classes/shape |
| --- | --- | --- |
| Behavior | `yolov9e.yaml` | `0: hand-raising`, `1: reading`, `2: writing` |
| Pose | `yolo26n-pose.yaml` | `0: person`; keypoints `[17,3]` |

The manifest also records this contract. Both models then loaded in `torch 2.8.0+cpu`, `ultralytics 8.4.188`, `dill 0.4.1`; assertions checked tasks, IDs/names, architecture, and keypoint shape. Behavior loading needs `dill`. `YOLO_AUTOINSTALL=false` prevented implicit dependency installs; `YOLO_CONFIG_DIR` was project-local `artifacts/ultralytics`.

On 2026-10-25, each ran once on the existing local `test_images/image_test.jpg` (3648×2736), using `imgsz=960`, `conf=0.25`, `save=False`. Behavior returned four boxes (three reading, one writing); pose returned 18 people with `[18,17,3]` keypoints. These separate one-image operability checks were not fused results or human-annotated accuracy tests. Counts vary with thresholds/input size and are not fixed performance metrics. The input image is excluded from source exports.

Exact classes/order come from checkpoint `names`, not dataset README guesses. This checkpoint is not a general seven-class model. Pose is not attention, comprehension, or learning effectiveness. Occlusion, small targets, seat views, and video behavior need independent annotation. Author-reported results are not this project's independent tests.

## Separate licensing layers

These are historical source statements, not new legal determinations:

1. The [pinned Space README](https://huggingface.co/spaces/nontaphat/Student-Behavior-Detector/blob/dd6c9381099400c4c64135335cb531bdad35ed1d/README.md) has `license: mit`. Space metadata alone does not establish MIT terms for weights, training data, and all upstream rights.
2. [SCB upstream](https://github.com/Whiffe/SCB-dataset#4-commercial-use-restrictions) states academic/personal/noncommercial restrictions and written permission for commercial use, explicitly including weights. Downstream MIT metadata does not automatically remove them.
3. [Ultralytics licensing](https://www.ultralytics.com/license) describes AGPL-3.0 and Enterprise options. Framework/model requirements need separate consideration.

Public download availability does not make these weights fully permissive or commercially authorized. No additional commercial permission was obtained in this integration, and integrity verification is not license clearance. Preserve upstream notices and check the actual downloaded revisions before use/redistribution.
