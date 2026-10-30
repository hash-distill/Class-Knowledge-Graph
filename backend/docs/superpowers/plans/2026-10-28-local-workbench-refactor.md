# Local Classroom Workbench Refactor Implementation Plan

> **For agentic workers:** Use subagent-driven-development for isolated modules and execute continuously. Preserve existing source material and verify spec compliance before code quality. This workspace is not a Git checkout; create source snapshots instead of commits.

**Goal:** Refactor the working 4C application into a local-first FastAPI application with persistent SQLite state, independent background workers, teacher review and follow-up records, and a desktop launch boundary.

**Architecture:** Original lesson JSON and media remain immutable inputs. SQLite stores catalog metadata, evidence indexes, teacher revisions, follow-up actions, exercises, responses and durable job records. FastAPI composes small route modules; heavy import/index/model work runs outside request handlers. The existing Vue application and HTTP contracts remain usable while an additional teacher workspace exposes the new cycle.

**Tech Stack:** FastAPI/Uvicorn, Python standard-library SQLite and subprocess, Vue 3/TypeScript/Vite, existing independent model runtimes, Tauri 2 desktop shell where the build toolchain is available.

## Compatibility and contracts

- Preserve `/api/review/health`, `/lessons`, `/lessons/{id}`, `/chat`, `/media/{id}/{path}`, existing binary courseware upload/confirm/detach and `error.code/message`.
- Original transcript text, native timestamps, source SHA, behavior outputs and model results are never rewritten by teacher edits.
- Add `GET /api/review/lessons/{id}/workspace` returning `{lesson_id, source_revision, version, goals, corrections, actions, assessments, attempts}`. `source_revision` hashes original transcript and source-media identity; `version` is an optimistic concurrency counter.
- Mutation requests include `expected_version`. A stale request returns HTTP 409 `workspace_changed`. Source changes return HTTP 409 until a teacher explicitly reconciles the saved workspace.
- `POST .../workspace/goals`: `{text, expected_version}`.
- `POST .../workspace/corrections`: `{transcript_id, replacement, reason, expected_version}`. An empty replacement is rejected; a separate revert operation preserves audit history.
- `DELETE .../workspace/corrections/{transcript_id}?expected_version=N`: revoke an active correction, retain its audit record.
- `POST .../workspace/actions`: `{title, evidence_ids, expected_version}`.
- `PATCH .../workspace/actions/{action_id}`: `{status: planned|done, outcome, expected_version}`. A completed action requires the actual recorded outcome.
- `POST .../workspace/assessments`: `{question, reference_answer, evidence_ids, expected_version}`. Teacher-authored or teacher-confirmed exercises are persisted.
- `POST .../workspace/assessments/{assessment_id}/attempts`: `{answer, feedback, score: number|null, expected_version}`. Recorded answers are observations, never fabricated model outputs.
- Each mutation returns the complete updated workspace.
- `GET /api/review/tasks?lesson_id=ID`, `GET /api/review/jobs/{job_id}`, `POST /api/review/tasks/{id}/cancel` and `POST .../{id}/retry` expose persisted task status. Existing upload results continue to use `job_id` and `running/complete/failed`; additional queued/cancelled states are documented and handled in the UI.
- New derived teacher transcript/evidence fields are separate from original and machine-proofread fields. QA and retrieval prefer the active teacher-confirmed view, with provenance and original citations retained.

## Task 1: Persistent jobs and independent workers

Files: `review/jobs.py`, `review/job_worker.py`, `review/courseware_service.py`, `scripts/serve_review.py`, `tests/test_persistent_jobs.py`.

- [x] Write failing tests for persistence across controller recreation, a real subprocess job, cancel/retry, stale-source refusal, and recovery from an interrupted worker.
- [x] Implement SQLite transactions with bounded retries and an explicit allowed-kind dispatcher. Store request payloads privately; return safe status summaries.
- [x] Move production courseware computation to child processes. Preserve transaction locks, source checks and final-publication ordering. An explicit injected in-process executor remains available to embedding callers and synthetic contract tests.
- [x] Capture child failures, release locks and make interrupted work recoverable; never kill unrelated processes or report partial results as complete.
- [x] Run new worker tests and the original courseware service tests.

## Task 2: SQLite teacher workspace and evidence catalog

Files: `review/workspace.py`, `review/lesson_store.py`, `review/evidence_index.py`, `tests/test_workspace.py`, `tests/test_evidence_index.py`.

- [x] Write failing tests using a tiny immutable lesson fixture for save/reload, optimistic concurrency, teacher-correction audit/revert, evidence-ID validation, actions and actual exercise responses.
- [x] Add SQLite metadata and Chinese lexical evidence search. Maintain source and index fingerprints; update only when source or confirmed transcript changes.
- [x] Extract LessonStore without breaking existing Python imports and CLI consumers. Add explicit storage paths for tests and normal local runtime.
- [x] Overlay teacher evidence separately; verify that raw lesson JSON hashes do not change.

## Task 3: Modular API composition

Files: `review/api/{application,http,lessons,media,chat,workspace,tasks}.py`, `review/server.py`, `review/retrieval.py`, `tests/test_workspace_api.py`, `tests/test_fastapi_server.py`.

- [x] Keep `review.server.create_app`, `serve`, `LessonStore` and `byte_range` as compatibility exports.
- [x] Implement the contracts above using small routers and typed requests. Extend same-origin and body limits to PATCH requests.
- [x] Start and stop the worker controller through FastAPI lifespan. Use one GPU-task slot, no model copies in API workers.
- [x] Verify old HTTP contracts, readable OpenAPI, real Range/HEAD behavior and responsiveness while a child job runs.

## Task 4: Teacher workspace UI

Files: `frontend src/api`, `src/types`, `src/components/review/TeacherWorkspace.vue`, workbench navigation and validators, frontend scripts.

- [x] Add a distinct teaching-workspace view while retaining the single shared player and current navigation state.
- [x] Load persisted goals, teacher corrections, actions, exercises, responses and job states from the API. Preserve pending edits and report optimistic-concurrency conflicts with a reload action.
- [x] Support correction/revert, action creation/completion and teacher-recorded exercise responses. Empty and missing values remain explicit.
- [x] Show queued/running/failed/cancelled/completed task state and retry/cancel controls; never synthesize progress or outcomes.
- [x] Run TypeScript/build and existing review/activity tests, then verify new flows against the real local API in a browser.

## Task 5: Runtime/model and desktop boundaries

Files: `review/runtime.py`, `configs/runtime_profiles.json`, desktop launcher/build files, README and architecture documentation.

- [x] Describe actual installed models and missing optional assets; expose model roles and selected configuration without silently relabeling historical results.
- [x] Retain independent ASR/LLM processes and add explicit profile boundaries for Qwen3-ASR-0.6B, Qwen3.5-4B and ONNX OCR/embedding assets. A model is available only after local asset verification; configuration alone does not claim completed inference.
- [x] Serve the built frontend from the local application for a single production entry point.
- [x] Add desktop launch/build configuration and verify available build tools. Preserve normal browser launch when desktop packaging is unavailable.
- [x] Document install/start, DB and artifact locations, task recovery, model installation boundaries and rollback.

## Final acceptance

Run backend `pytest tests -q`, frontend `test:review`, `test:activity`, new workspace tests and `npm run build`. Use the real six-lesson catalog to verify unchanged raw artifacts, API restart persistence, an independently processed courseware job, teacher corrections and revert, saved follow-up outcomes and actual sample responses. Record commands and results under `artifacts/verification/architecture-refactor-*`. Do not claim ASR accuracy gains or desktop installers that have not been built.

Execution record: backend 317 passed and 4 skipped with the final frozen desktop backend; frontend review/activity checks and production build passed. Real API/browser acceptance and the real queued PDF import passed in an isolated lesson copy. See `artifacts/verification/architecture-refactor-20261028-154904/ACCEPTANCE.md` and `../desktop/build/release-manifest.json`. OCR/BGE inference adapters and new model accuracy evaluation remain outside this framework implementation.
