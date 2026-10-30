# Chinese proofreading and current visible states

After `start_trained_demo.ps1`, open the [local workbench](http://127.0.0.1:5175/). The prepared Chinese chemistry demo aligns speech, observed slides, and behavior on the source-video timeline. Switching views preserves the player/time. Prepared private demo data is excluded from source releases.

## Build-time proofreading

`build_review.py` creates proofreading before concept extraction using full context, observed courseware terms, and same-source secondary ASR. Replacements retain original terms, suggestions, evidence, and uncertainty. Formulas, numbers, negation, and syntax have separate guards. Text the model cannot fully handle remains original.

Speech defaults to machine proofreading, with an original-ASR comparison. Original T segments/times are unchanged; proofreading uses P IDs and continuous evidence PB IDs. Q&A/concepts use sourced proofreading; explicitly uncertain text is excluded from concepts. Replay is still required for human verification.

A lesson with prepared audio can add secondary Chinese ASR. From `backend/`, explicitly configure the remote environment first, as described in the [backend README](../README.md):

```powershell
.venv-media/Scripts/python.exe scripts/generate_chinese_reference.py --media-dir artifacts/unified-chemistry-001-9
```

The configured SSH environment runs Seaco Paraformer, VAD, and punctuation. There is no default private host. Only `asr.secondary.json` matching both audio/video digests is published; Whisper is retained. Install `requirements-asr-chinese.txt` remotely; caches retain models/revisions. Override `--host`, `--remote-python`, `--remote-root`, and `--gpu` as needed. Enabling this transfers lesson audio to that host.

For another lesson:

```powershell
./run_lesson.ps1 -Video 'D:\lesson-materials\lesson.mp4' -LessonId 'lesson-new' -Title 'New lesson' -ChineseReference
```

Secondary ASR and proofreading retain separate provenance. Cache checks include original ASR, courseware, other recognizers, and model identity; changed inputs regenerate proofreading.

## Current-state display and historical metrics

Overview prioritizes the sampled frame, boxes, state counts, and detected-action fraction. The behavior view has time trends; cumulative counts/coverage are in technical diagnostics.

The earlier pretrained action path recognizes reading, writing, and hand raising. Separate pose rules cover head up/down, hand near head, and standing with sufficient leg evidence. Rules record COCO17 points, confidence, and thresholds; absent categories are not filled with invented counts. The newer seven-class path is described in [Vision changes](VISION_IMPROVEMENT_20261029.md).

In the earlier path, detected-action fraction is reading/writing/hand-raising count divided by detected people. Its reference score averages only explicit action weights; geometric poses are unscored and missing/no-person frames remain unknown. These historical metrics differ from the [current overall behavior engagement](CLASS_ENGAGEMENT_20261029.md). Neither measures psychological engagement or mastery.

## Acceptance commands

From `backend/`, then switch to the flattened frontend directory:

```powershell
cd ../frontend
npm run test:review
npm run test:activity
npm run build
python scripts/verify-proofread-students.py
python scripts/verify-unified-workbench.py
```

The browser scripts use real local API, source video, and local models. The first checks proofreading comparison, current states, removed entry points, and mobile layout. The second checks playback across views, courseware import, three-source Q&A, and full export. Records/screenshots go to local `output/playwright/`. Historical browser acceptance does not validate later prompt translations or prove model accuracy.
