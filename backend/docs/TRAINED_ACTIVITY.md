# Jingxi Insight: linking lesson content and visible behavior

Historical execution record dated 2026-10-26, with additions through 2026-10-27. The workbench retains speech, recorded slides, graph, independent courseware, and evidence Q&A, adding same-source behavior analysis. Vision runs offline in separate scripts and is verified before publication; the API does not load vision models. See [current scoring](CLASS_ENGAGEMENT_20261029.md) and [vision changes](VISION_IMPROVEMENT_20261029.md) for subsequent updates.

## Open a prepared demo

From `backend/`:

```powershell
.\start_trained_demo.ps1
```

The prepared default is the [synchronized Chinese chemistry demo](http://127.0.0.1:5175/?lesson=unified-chemistry-001-9&view=review): one 90-second recording supplies speech, real projection footage, and visible behavior. Earlier Chinese math/language/chemistry lessons remain selectable; English lessons are archived. Source releases exclude prepared lesson data and media.

The [chemistry behavior demo](http://127.0.0.1:5175/?lesson=ts-chemistry-001-9) has reading/writing and valid historical rule scores around 40 s; slide-only shots remain unknown. The [52-second math demo](http://127.0.0.1:5175/?lesson=ts-math-006-32) lacked sufficient temporal evidence, so its curve stayed empty while real hand-raising predictions and source frames remained available.

The launcher reuses running services and starts/checks the local model, API, frontend, and demo data. Use `-LlmBackend cpu` for CPU or `-RestartApi` after API changes. New machines need dependencies/models first; it does not download data. Source video remains in the local dataset; update `configs/lessons.json` when moving machines.

The historical five views were overview, speech/courseware, behavior, graph, and report, sharing one player/time with an on-demand assistant. Teaching improvement was added later. Behavior sections separate overview, heatmap, source frames, and model limits. Full exports retain original records.

## Three distinct outputs

| Output | Input/task | Relationship to the recording |
| --- | --- | --- |
| Visible behavior | Pretrained YOLO person/pose and reading/writing/hand-raising detection plus temporal rules | Actual source-video inference; these detectors were not retrained in this experiment |
| TS teaching context | Frozen ImageNet MobileNetV3 Small, 16 frames/clip, mean/std features, eight-label linear head | RGB inference produces a coarse whole-clip prediction, not per-second labels |
| ARIC research record | Global CLIP `(1,401,768)` mean pooling/L2 plus multilabel head | Offline image-level behavior presence; not per-person online video inference |

Teacher/student roles were unverified. Historical participation was an uncalibrated 0–100 rule quantity, not psychological engagement/mastery. No people, insufficient evidence, and waiting after a shot cut stayed unknown. Score coverage counted object observations with non-null rule scores, not immediately recognized states. Cross-frame totals are observations, not unique people; tracks reset across cuts.

## Recorded training/evaluation

| Item | TS | ARIC |
| --- | --- | --- |
| Locally available | 70 public clips, eight source-video groups, about 1.365 h | Global features/box annotations; original RGB and individual features missing |
| Used | 70 clips; 44 train, 10 validation, 16 test | Deterministic 2,048-image sample; 1,228 train, 410 validation, 410 test |
| Split unit | Original video; math006/art001 test, Chinese001 validation | Original image ID; old indexes overlapped and were resplit |
| Head | Eight multilabel classes | 25 training-supported labels; seven excluded |
| Test Macro-F1 | 0.227814 | 0.450011 |
| Test Micro-F1 | Not computed in recorded report | 0.686210 |
| Test mAP | 0.460947 | 0.472439 |
| Baseline | Training-prior Macro-F1 0.119048 | All-positive Macro-F1 0.284397; training-prior 0.122266 |
| Measured time | Local CPU: 1,120 frames, about 151.56 s; A6000 head: about 1.97 s | Preparation about 369.49 s, feature cache 350.87 s; CPU head 1.55 s; server CPU rerun 0.64 s |

Seconds-long fitting excludes environment setup, archive I/O, video processing, and detector fine-tuning. TS used one RTX A6000: physical GPU3 mapped to process `cuda:0`. ARIC used NumPy CPU, without a GPU-speed claim. Local reports: `artifacts/learning/ts-context/metrics.json`, `artifacts/learning/aric/metrics.json`, and ARIC `remote_metrics.json`.

Only 70 public TS examples were available, not the full 973 clips. Small test groups/low TS Macro-F1 make this a runnable baseline, not mature accuracy. Chemistry is in the TS **training set**, labeled as such in UI/Q&A. Math is test data, but clip prediction still does not validate per-person behavior. Weak ARIC labels such as `taking_bottle` remain visible in reports.

Normalization uses training data only and thresholds validation only. TS does not receive reference transcripts, behavior descriptions, or target text; ARIC does not receive captions. `model.npz` loading disables pickle; TS verifies metadata SHA-256. Data/personal media are excluded from source releases. Model publication needs applicable dataset/pretraining permission checks.

## Replay

From `backend/`, use NumPy, PyTorch, TorchVision, and OpenCV in the vision environment; ASR/OCR/bundling use the media environment:

```powershell
# Frozen-encoder RGB features; use a new output directory to preserve prior results.
..\vision\.venv-inference\Scripts\python.exe scripts/train_ts.py --dataset './local-datasets/TS-Interact' --output artifacts/learning/ts-retrain --device cpu --extract-only
# Copy the cache/manifest to the A6000 host; the operator chooses a free CUDA device.
python scripts/train_ts.py --features artifacts/learning/ts-retrain/features.npz --output artifacts/learning/ts-retrain --device cuda:0 --epochs 300 --seed 42

# Portable ARIC bundle replay without reading the large source archive.
python scripts/train_aric.py --from-bundle artifacts/learning/aric/training_bundle.npz --output artifacts/learning/aric-rerun

# Actual video processing, lesson build, behavior inference, and overlay publication.
.venv-media\Scripts\python.exe scripts/prepare_media.py --video '<source-video.mp4>' --output artifacts/my-lesson --mode all --roi 0 0 1 1 --sample-interval 5
.venv-media\Scripts\python.exe scripts/build_review.py --lesson-id my-lesson --video '<source-video.mp4>' --media-dir artifacts/my-lesson --title 'Lesson content and behavior'
..\vision\.venv-inference\Scripts\python.exe scripts/analyze_activity.py --video '<source-video.mp4>' --lesson-id my-lesson --media-dir artifacts/my-lesson --interval .5 --imgsz 960 --device cpu --context-model artifacts/learning/ts-context --aric-report artifacts/learning/aric/metrics.json
```

`--enrich-only` adds/updates TS and ARIC records in an existing complete `activity.json` without rerunning person detection, while still checking video/frame/time provenance. Rebuild/import/removal retains valid same-source overlays and original media/ASR. See [ARIC contract](ARIC_TRAINING.md).

## Acceptance

```powershell
# From backend/:
.venv-media\Scripts\python.exe -m pytest tests -q
..\vision\.venv-inference\Scripts\python.exe -m pytest tests/test_ts_learning.py -q
# Then from frontend/:
cd ../frontend
npm run test:review
npm run test:activity
npm run build
python scripts/verify-proofread-students.py
```

Browser checks require Playwright/Chromium and real API, media, and local models, without mocked routes. Frontend local `output/playwright/trained-activity-real` retains source-image clicks, unknowns, heatmaps, exports, behavior citations, mobile checks, and screenshots. This is functionality acceptance, not model-accuracy evaluation.

## Chinese lesson addition — 2026-10-27

`ts-chinese-001-2` is the complete 82-second source segment. It produced 165 behavior samples, six shot segments, 3,110 person detections across frames, and 46 scored observations. Chinese OCR at source resolution/full frame produced six page intervals and nine locatable concepts. Obvious OCR noise is filtered for display; raw recognition/provenance/metadata remain in full data.

Default VAD retained only about 4.352 s and produced no reliable transcript. A first-25-second diagnostic without filtering still failed, so no dialogue was invented. `--allow-no-speech` explicitly built a visual-evidence lesson with unrecognized speech; Q&A used actual slides/behavior. The main example retained the complete Chinese speech lesson titled “正弦定理” (“Sine rule”).

Select `-LessonId ts-chinese-001-2` or `-LessonId ts-chemistry-001-9`. Port 5175 replaced 5173 because another project used it at that time. Frontend `scripts/verify-chinese-demo.py` checked Chinese lessons, real speech Q&A, visual evidence, citations, and exports without rewriting source evidence.

## Synchronized-demo workflow

1. In overview, inspect three sources at the same time. Around 40 s there are reading/writing predictions and rule scores. At 43.5–47.5 s the Chinese projection is visible but people are not, so behavior stays unknown. Do not combine different shots as simultaneous events.
2. In speech/courseware, inspect 28 machine-ASR segments guided by observed slide vocabulary. The original 36 unguided segments remain in full data. Neither is human-verified, and chemistry terms still contain errors.
3. Independent courseware uses a companion file compiled from the original 45 s projection. Real matching covers only 43.5–47.5 s. Page two documents provenance and has no lesson time. Local `artifacts/unified-demo-courseware` PPTX/PDF are observation compilations, not the author's original slides.
4. Inspect behavior trends, heatmaps, and frames. The graph shows five model candidates with literal observed-slide support, retaining 12 originals separately. Rule scores are not psychological engagement.
5. The assistant can cite speech, courseware, and visible behavior; citations seek the shared player. Full report export retains original records/provenance.

Frontend `scripts/verify-unified-workbench.py` used real API/video/models and reimported the PDF through the browser. Historical acceptance checked shared video hashes, navigation while playing, back/forward, search/chat retention, reimport, citations, full exports, and mobile layout. Later English prompt translations require separate model-output evaluation.
