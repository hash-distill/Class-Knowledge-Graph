# Run pretrained classroom-behavior observation

Historical setup dated 2026-10-25: this entry used public pretrained weights without training, two pinned downloads, and a separate `.venv-inference` to avoid global Anaconda OpenMP conflicts. This three-class workflow is retained separately from the [newer seven-class workbench path](README.md). The installed environment is excluded from source releases.

## Components

| Stage | Implementation | Boundary |
| --- | --- | --- |
| People, pose, tracks | Official YOLO26n-pose + ByteTrack | 17-point poses and video-local IDs; no names/automatic teacher-student distinction |
| Behavior | Public SCB YOLOv9e checkpoint | Only hand raising, reading, writing; IDs verified from the checkpoint |
| Association | One-to-one behavior/person matching | Unmatched objects stay unknown; geometric hand raising has separate provenance |
| Temporal fusion | Four-second evidence window integrated over real seconds | Quality, valid duration, and task determine whether rules return an estimate |
| Teaching content | User-provided timed tasks/concepts; optional OCR | No automatic concepts or mastery inference from behavior scores |
| Historical viewer | JSON import, frame replay, trends, object heatmaps, provenance | Mock/real modes separate; unknown stays null/— |

The rule index is not a learned engagement probability. Three classes mainly provide positive-participation clues; there is no reliable low-engagement classifier. Unknowns are excluded from means, so a high value may cover few objects. Always inspect coverage. Target-classroom annotated accuracy was not evaluated.

## Run locally

From `vision/` in PowerShell, with your own local media:

```powershell
./run_pretrained.ps1 -Source './local-media/lesson.mp4' -MaxFrames 35 -Device cpu
```

Remove `-MaxFrames 35` for the full video. Defaults analyze about every 0.2 s and save snapshots about every second, plus the final frame. Analysis times snap to the nearest video frame. Analysis and export intervals are independent.

```powershell
./run_pretrained.ps1 -Source 'D:\recordings\lesson.mp4' -Task lecture -StudentRoi '0.05,0.1,0.95,1' -AnnotatedVideo
```

- `-Task`: `unspecified / lecture / exercise / discussion / presentation`; supplies rule context without changing weights. Default: unspecified.
- `-StudentRoi`: normalized `x1,y1,x2,y2`; count detections with centers inside. Set it for the camera. Full-frame default may include teachers/others.
- `-AnnotatedVideo`: additionally writes boxed video; default outputs JSON/frames only.
- `-PoseOnly`: person/pose and geometric hand-raising diagnostics without reading/writing; not the full workflow.
- `-SnapshotIntervalSec 0`: save every analyzed frame; default one second limits volume.
- `-Device auto`: selects `cuda:0` if usable, otherwise CPU. The historical installation/check was CPU-only. Explicit GPU selection fails if unavailable instead of silently switching.

Outputs use a new `Class_Detection/artifacts/results/pretrained_<timestamp>/` directory:

```text
snapshots.json             # JSON array for the compatible legacy viewer
snapshots_002.json         # Subsequent chunks for long videos
frames/frame_0000001.jpg   # Sampled source frame
observations.mp4           # Only with AnnotatedVideo
run_manifest.json         # Completion, chunks, model digests, configuration, counts
```

JSON chunks are at most 16 MiB, within the historical viewer's 25 MB, 20,000-frame, 300,000-object limits. It loads one chunk at a time; seconds retain source time. `analyzed_frames` differs from exported `snapshots`. Cross-frame counts are repeated observations, not unique students. Failed runs retain `.partial` diagnostics; only `run_manifest.json` with `completed=true` establishes completion.

## Historical standalone viewer

The original workflow used the frontend inference-JSON importer. Files were local; a browser cannot infer neighboring image paths from local JSON. With a compatible standalone viewer, copy the complete result to a new `frontend/public/data/visual/<run>/` directory and load it from the same origin:

```text
http://127.0.0.1:5173/?data=/data/visual/my-run/snapshots.json
```

This documents the legacy viewer contract, not the full application's default lesson/API route. The earlier 35-frame `pretrained-v2` sample was real inference on an existing demo recording, not an independent test set. Public exports exclude that media/output.

The original viewer began at the first frame with no score until evidence accumulated. Without concept annotations, its graph remained empty. Model provenance showed hashes/revisions/links. For current integration use the [backend lesson workflow](../backend/README.md).

## Task/concept alignment

After checking video times, create UTF-8 context JSON. This is a format example, not real annotation:

```json
[
  {"start_sec": 0, "end_sec": 120, "task": "lecture", "knowledge_point": "An actually taught concept"},
  {"start_sec": 120, "end_sec": 300, "task": "exercise"}
]
```

Pass `-ContextJson './context.json'`. Intervals are half-open, nonnegative, and non-overlapping; uncovered time uses the default task. Concepts are content annotations with null scores. This entry disables ASR, slide OCR, facial gaze, and LLMs by default. Legacy PaddleOCR is optional, needs another environment, and was outside the smoke check.

## Recreate the CPU environment

From `vision/`, with Python 3.11/uv:

```powershell
uv venv .venv-inference --python 3.11
uv pip install --python .venv-inference/Scripts/python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-inference/Scripts/python.exe -r requirements-inference.txt
& .venv-inference/Scripts/python.exe Class_Detection/scripts/fetch_pretrained.py
& .venv-inference/Scripts/python.exe Class_Detection/scripts/fetch_pretrained.py --verify-only
& .venv-inference/Scripts/python.exe -m pytest Class_Detection/tests -q
```

`requirements-inference.lock.txt` records that CPU environment, not CUDA. Downloads pin size/revision/SHA-256; inference rechecks task/classes/architecture. Corrupt files are rejected; only explicit `--repair` attempts replacement.

## Legacy and publication boundaries

Earlier code/manuals/formulas remain legacy comparisons and are not called by this entry. Old configuration is `Class_Detection/configs/pipeline_legacy.yaml`; do not mix its scores with this index. Historical backups are local and excluded.

See [Pretrained sources](Class_Detection/docs/PRETRAINED_MODELS.md) for URLs/hashes/classes/terms. SCB restrictions are not removed by downstream Space MIT metadata; Ultralytics has separate terms. This integration establishes no unrestricted commercial redistribution permission. Publish code/manifests/downloaders with separate provenance, excluding unchecked video/weights.
