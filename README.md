# Jingxi Insight

A local classroom review assistant for teachers. It aligns classroom video, transcripts, courseware, and visible student behavior on one timeline for evidence replay, manual correction, cited Q&A, and teaching improvement records.

Repository: [hash-distill/Class-Knowledge-Graph](https://github.com/hash-distill/Class-Knowledge-Graph). The repository retains its historical name; the application is **Jingxi Insight**. The interface defaults to English. Original lesson language, transcripts, source quotations, and imported courseware remain in their source language.

## Features

| Workspace | Supported tasks |
| --- | --- |
| Classroom overview | View student footage alongside the current state, overall behavior engagement, and synchronized transcript |
| Speech and courseware | Compare original ASR, machine proofreading, recorded slides, and separate PDF/PPTX/PPT files; jump from evidence to the source video |
| Student behavior | Inspect boxes, state provenance, trends, and scoring support; add missed boxes, correct labels, and exclude false detections |
| Knowledge graph | Inspect lesson concepts and their original evidence; follow co-occurrence links to their sources |
| Review report | Export JSON and Markdown with original observations and separate derived scores |
| Teaching improvement | Save goals, teacher revisions, plans, implementation results, exercises, and actual answers |

Manual review keeps versions and source digests, with conflict detection, withdrawal, restoration, and classification-sample ZIP export. Courseware jobs persist in SQLite, retain interrupted status after a restart, and support retries. Model answers undergo citation validation. Model outages, missing evidence, and insufficient data have explicit states.

## Architecture

| Layer | Current implementation |
| --- | --- |
| Interface | Vue 3, TypeScript, Vite, and Pinia; blue theme and responsive layouts |
| API | Python 3.11, FastAPI, and Uvicorn; preserves the `/api/review` contract |
| Local storage | SQLite for teacher records, jobs, retrieval indexes, and manual review; JSON for original lesson evidence |
| Media | FFmpeg, OpenCV, and Tesseract; API media endpoints support Range/HEAD requests |
| Local Q&A | Lesson text/time retrieval, citation validation, and a separate llama.cpp-compatible service |
| Desktop | Tauri 2 with a Python sidecar; model runtimes and weights are installed separately |

Browsing processed lessons, editing records, and replaying evidence do not load a large model. Video processing, vision inference, and language models run through separate scripts or processes. On a computer with about 8 GB VRAM, load models sequentially by task; memory use depends on the model, context length, and configuration.

## Models and integration status

| Task | Existing path | Status |
| --- | --- | --- |
| Primary transcription | faster-whisper Small, CPU int8 | Default path in `prepare_media.py` |
| Chinese ASR upgrade | Qwen3-ASR-1.7B | Separate experiment; see [Qwen upgrade](backend/docs/QWEN_UPGRADE.md) |
| Secondary Chinese ASR | FunASR Seaco Paraformer, FSMN-VAD, CT-Punc | Optional SSH reference channel; requires an explicitly configured remote host |
| Person and pose detection | YOLO26n and YOLO26n-pose, with full-frame/tiled fusion | Person localization is separate from behavior classification; box counts are not class attendance |
| Seven visible behaviors | MobileNetV3 Small crop classifier | Experimental weights of about 3.15 MB; low-confidence and conflicting states remain unknown |
| Concepts and Q&A | Qwen3-4B GGUF with llama.cpp | Separate local service; generated citations must pass validation |
| Proofreading | Source rules, ChineseErrorCorrector4-4B, and Qwen3.5-4B comparisons | Original ASR is preserved; machine revisions require teacher review |

Qwen3-ASR-0.6B, PP-OCRv6 ONNX, and BGE-small in runtime manifests are recommended installs or pending adapters. A manifest entry alone does not establish deployment. Current Q&A uses text retrieval; a vector database and GraphRAG are not default components.

## Run from source

Install Python 3.11, [uv](https://docs.astral.sh/uv/), and Node.js 22 or 24. Run these commands from the repository root to start the application and API. They do not download models or classroom video.

```powershell
git clone https://github.com/hash-distill/Class-Knowledge-Graph.git
cd Class-Knowledge-Graph
uv venv backend/.venv-media --python 3.11
uv pip install --python backend/.venv-media/Scripts/python.exe -r backend/requirements-app.txt
cd frontend
npm ci
npm run build
cd ..
python scripts/workspace_setup.py
python scripts/run_workbench.py --port 8000 --open-browser
```

Open the [local workbench](http://127.0.0.1:8000/). [API documentation](http://127.0.0.1:8000/docs) is available at `/docs`; the health endpoint is `/api/review/health`. The service binds to `127.0.0.1`.

The first run creates an empty lesson registry and local data directories. An empty lesson list is expected. Source distributions exclude classroom video, transcripts, student images, model weights, and personal workspaces. Initialization preserves an existing registry.

On Linux/macOS, use `backend/.venv-media/bin/python` for dependency installation. The source launcher recognizes both virtual environment layouts. See [Desktop and runtimes](backend/docs/DESKTOP.md) for Windows package and PowerShell media-entry support.

### Frontend/backend development

Run `python scripts/run_workbench.py --api-only --port 8000` in one terminal. In another, enter `frontend` and run `npm run dev -- --host 127.0.0.1 --port 5175`. Vite proxies `/api/review` to the backend. `npm run preview` checks the static build; use the source launcher above for the full application.

If the API started before the first frontend build, stop it with `Ctrl+C` in its original terminal, then launch the full application. The launcher refuses to reuse an older service without static pages and leaves that service running.

`python scripts/run_workbench.py --check` checks paths, tools, and build status. Use `--python`, `--registry`, and `--data-dir` to override the interpreter, lesson registry, and state directory.

## Use your own lessons

Register processed lessons in the private file `backend/configs/lessons.json`, for example:

```json
{
  "my-lesson": {
    "directory": "../artifacts/my-lesson",
    "source_video": "../local-media/my-lesson.mp4"
  }
}
```

Relative paths resolve from the registry file's directory; absolute paths are also supported. `directory` must contain a complete, source-consistent `bundle.json` and its evidence files. Registering a video path alone does not generate analysis. Keep original media and the registry local.

New video processing requires extra media dependencies, FFmpeg, Tesseract Chinese/English language packs, the relevant models, and the actual slide region for that recording. See [Backend and lesson processing](backend/README.md) and the [Vision module](vision/README.md). Uploading PDF/PPTX/PPT supplements an already registered lesson; it does not run the entire video inference pipeline.

See [Local models](backend/docs/LOCAL_LLM.md) for Q&A configuration. If the server explicitly sets an external `REVIEW_LLM_BASE_URL`, retrieved evidence is sent there. Optional secondary Chinese ASR sends the lesson WAV to an explicitly configured SSH host; no remote host is selected by default.

## Overall behavior engagement

The interface uses `overall_behavior_engagement_v1`: writing 85, reading 80, looking up 85, turning the head 30, raising a hand 95, standing 90, and discussion 75. After normalizing valid behavior weights, it computes `100 × mean × exp(-0.6 × population standard deviation)` and applies temporal smoothing within a continuous shot. Overview, detail, trends, and exports share the implementation.

One trusted behavior observation is sufficient for an estimate. Fewer than three scored objects or coverage below 60% produces a low-support label. No-person frames, absent valid behaviors, geometric poses, and unrecognized objects are not assigned zero. Manual review is stored separately and does not automatically recompute the lesson-wide model curve. See [Scoring details](backend/docs/CLASS_ENGAGEMENT_20261029.md) for weights, smoothing, coverage, and exports.

This is an uncalibrated rule estimate of detected objects' visible behavior. It does not measure individual grades, psychological engagement, or knowledge mastery. The classifier's fixed-crop test contains adjacent-frame overlap; its Macro-F1 is not independent classroom end-to-end accuracy. See the [Vision change record](backend/docs/VISION_IMPROVEMENT_20261029.md).

## Verification and release

```powershell
& backend/.venv-media/Scripts/python.exe -m pytest backend/tests -q
cd frontend
npm test
npm run build
cd ..
python scripts/prepare_release.py --check
python scripts/prepare_release.py --output release/github-ready
```

The exporter creates `source/`, `jingxi-insight-source.zip`, and a per-file SHA-256 manifest. It excludes local registries, databases, lesson artifacts, weights, environments, build directories, and nested Git metadata. It refuses to overwrite an existing output directory and stops on common credential patterns. Inspect file contents and material permissions before publication.

[GitHub publishing](docs/PUBLISHING.md) explains the unified layout, migration of historical vision files, and publication checks. GitHub Actions runs backend/frontend tests, production builds, and source audits. Model accuracy and real classroom acceptance require separate evaluation.

## Layout and documentation

```text
scripts/    Workspace initialization, source launcher, release exporter
backend/    FastAPI, media processing, evidence, teacher workspace, tests
frontend/   Vue workbench
desktop/    Tauri shell and packaging scripts
vision/     Person localization, state classification, historical vision pipeline
docs/       Release documentation
```

- [Manual vision review and sample export](backend/docs/VISUAL_REVIEW_20261029.md)
- [Local jobs and data architecture](backend/docs/LOCAL_ARCHITECTURE.md)
- [Independent courseware import and alignment](backend/docs/COURSEWARE.md)
- [Frontend development](frontend/README.md)
- [Contributing](CONTRIBUTING.md)
- [Licensing and third-party material](NOTICE)

Existing classroom materials, training data, third-party code, and models retain their respective terms. No single open-source license currently covers all application code. Public source availability does not grant redistribution rights for classroom materials or model weights.
