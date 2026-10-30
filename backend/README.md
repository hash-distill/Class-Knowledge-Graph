# Jingxi Insight backend

FastAPI, Uvicorn, and SQLite provide the `/api/review` API for the Vue workbench. Start the complete application using the [root README](../README.md). This document covers data, processing, and model configuration.

## Application environment

From the repository root:

```powershell
uv venv backend/.venv-media --python 3.11
uv pip install --python backend/.venv-media/Scripts/python.exe -r backend/requirements-app.txt
python scripts/workspace_setup.py
python scripts/run_workbench.py --api-only --port 8000
```

After building the frontend, omit `--api-only` to serve the interface on the same port. API documentation is at `/docs`; health is at `/api/review/health`. Application startup does not load GPU models. Install inference libraries and models separately for the tasks you need.

Relative lesson paths resolve from the registry's directory. Entry-point-specific arguments determine whether command-line options override environment variables.

| Setting | Purpose |
| --- | --- |
| `configs/lessons.json` | Private lesson registry; initially empty and excluded from source releases |
| `REVIEW_LESSON_REGISTRY` | Registry path for direct backend or desktop entry points |
| `REVIEW_DATA_DIR` | Databases for jobs, teacher records, reviews, and indexes |
| `REVIEW_FRONTEND_DIR` | Built interface directory |
| `REVIEW_RUNTIME_ROOT` | Model/runtime and public configuration root |
| `REVIEW_LLM_BASE_URL` | Compatible model endpoint; default `http://127.0.0.1:18081/v1` |
| `REVIEW_LLM_MODEL` / `REVIEW_LLM_API_KEY` | Server-side model alias and optional key; do not expose them to the frontend |

The root `scripts/run_workbench.py` has its own default registry, state, and runtime paths. Use `--registry` and `--data-dir` to override them. Initialization preserves existing registries and evidence.

## Process a lesson

New video processing requires FFmpeg, Tesseract (`chi_sim` and `eng`), and extra media dependencies. Run the following from `backend/` with your own local media:

```powershell
uv pip install --python .venv-media/Scripts/python.exe -r requirements-media.txt -r requirements-app.txt
& .venv-media/Scripts/python.exe scripts/fetch_local_llm.py --runtime cpu
./scripts/start_local_llm.ps1 -Backend cpu
./run_lesson.ps1 -Video ./local-media/my-lesson.mp4 -LessonId my-lesson -Title 'My lesson' -Roi 0.49,0.055,0.91,0.47 -SkipModelStart
```

For CUDA engines, choose a download/start configuration in [Local models](docs/LOCAL_LLM.md). Downloads require network access, disk space, and compliance with the applicable terms. The ROI above demonstrates the parameter format; replace it with the actual slide region for your camera. This media entry point currently targets Windows PowerShell.

The pipeline runs primary ASR and recorded-screen OCR, verifies video digests, time bases, and processing bounds, then produces proofreading, evidence, concepts, and reports before registering the lesson. `status != complete` and `.partial` files are incomplete outputs. Use a new lesson ID for a different video or configuration to avoid mixing caches.

| File | Contents |
| --- | --- |
| `asr.json` | Original transcript, timestamps, model version, and source digest |
| `slides.json` | Observed slide intervals and OCR from the recording |
| `raw_ocr.jsonl` / `ocr_frames/` | Replayable samples and raw OCR |
| `transcript.proofreading.json` | Separate machine proofreading; original ASR retained |
| `graph.json` | Evidence-backed concepts and co-occurrence relationships |
| `bundle.json` | Frontend/backend v3 lesson contract |
| `report.json` / `report.md` | Report at generation time |

Recorded screens and independent PDF/PPTX/PPT files are stored separately. Temporal overlap alone does not prove a semantic match; unmatched pages, ambiguity, and visual gaps remain explicit. See [Courseware](docs/COURSEWARE.md) for rendering dependencies and use.

## Model configuration

Primary ASR defaults to faster-whisper Small CPU int8. Qwen3-ASR-1.7B and Qwen3.5-4B are separate upgrade experiments; see [Qwen upgrade](docs/QWEN_UPGRADE.md). See [CEC4](docs/CEC4.md) for its proofreading role. Machine revisions are not human transcripts.

Seaco Paraformer remains an optional secondary Chinese-ASR channel with FSMN-VAD and CT-Punc. Scripts have no built-in server address, user directory, or GPU ID. Configure them explicitly:

```powershell
$env:REVIEW_ASR_SSH_HOST = 'user@your-server'
$env:REVIEW_ASR_REMOTE_PYTHON = '/path/to/asr-env/bin/python'
$env:REVIEW_ASR_REMOTE_ROOT = '/tmp/jingxi-asr'
$env:REVIEW_ASR_GPU = '0'
./run_lesson.ps1 -Video ./local-media/my-lesson.mp4 -LessonId my-lesson -Title 'My lesson' -ChineseReference
```

Install `requirements-asr-chinese.txt` in the remote environment separately. Enabling the channel transfers the lesson WAV over SSH to the specified host and writes a separate `asr.secondary.json` bound to video/WAV digests. With no host configured, the script exits before reading media or connecting.

Concept proposals and Q&A use Qwen3-4B through a separate local service. The server retrieves lesson evidence, limits citations, and validates them. Offline, timeout, truncation, and invalid-citation conditions return explicit errors. An explicitly configured external endpoint receives the retrieved text.

## Student behavior and manual review

The [Vision module](../vision/README.md) generates analysis offline; the workbench verifies provenance before reading it. The overview emphasizes student footage. Current frames, trends, and exports share the [overall behavior engagement rules](docs/CLASS_ENGAGEMENT_20261029.md). Missing observations are not zero-filled, and low-support estimates are labeled.

[Manual vision review](docs/VISUAL_REVIEW_20261029.md) supports label, role, and box correction; false-detection exclusion; missed-box addition; version restoration; and sample ZIP export. Reviews occupy separate tables in `workspace.sqlite3` and do not overwrite model results. Only explicitly confirmed students with known behavior produce classification crops. Partially labeled frames are not complete person-detection training labels.

Teacher goals, revisions, improvement plans, actual answers, and implementation results are stored separately. Courseware jobs use a SQLite queue and subprocesses; start a single Uvicorn worker. See [Local architecture](docs/LOCAL_ARCHITECTURE.md) for recovery.

## Tests and evidence boundaries

```powershell
& .venv-media/Scripts/python.exe -m pytest tests -q
```

Unit tests mainly use synthetic fixtures. Frozen-backend tests require an explicitly supplied built binary. These verify implementation, not real classroom recognition accuracy. Real acceptance records live in local `artifacts/verification` and are excluded from source distributions.

Graph edges represent sourced co-occurrence; prerequisite or causal relationships have not been validated. Overall behavior engagement is an uncalibrated visible-behavior rule estimate, not psychological engagement, knowledge mastery, or teaching effectiveness. Original video, timestamps, and raw ASR remain the reference evidence.
