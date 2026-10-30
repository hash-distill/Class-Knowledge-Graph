# Local teaching workbench architecture

This describes the 2026-10-28 refactor for a single personal computer. Independent services/scripts perform model computation; the API process does not load model weights.

## Modules and storage

`review/server.py` preserves `create_app`, `serve`, `LessonStore`, and `byte_range`. HTTP implementations in `review/api/` handle lessons, media, Q&A, courseware, jobs, and teacher records. `review/lesson_store.py` reads the registry and original lesson bundles.

State defaults to `artifacts/runtime/`; override it with `REVIEW_DATA_DIR`.

| Location | Contents |
| --- | --- |
| `workspace.sqlite3` | Teaching goals, teacher revisions/undo history, improvement tasks, exercises, actual answers, and old-source archives |
| `evidence.sqlite3` | Lesson-isolated Chinese full-text evidence indexes and fingerprints |
| `jobs.sqlite3` | Jobs, progress, attempts, retry relationships, and publication recovery journal |
| Original lesson directories | Real media, raw ASR, machine proofreading, courseware alignment, and historical model outputs |

Connections close after each operation; writes use transactions. Teacher records use `expected_version` to prevent overwriting newer edits from another user/window. Old-source records remain viewable and are archived when a new source is confirmed.

Chinese retrieval uses SQLite FTS5 and character fragments, returning only real evidence from the current lesson. BGE/ONNX remains optional in the runtime manifest; semantic-vector inference was not enabled in this refactor. File presence, executable presence, and completed inference are separate states.

## Teacher revisions and records

Teaching improvement shares the player with other views. Teachers can save goals, replay and revise text, create plans, enter actual results, and save confirmed exercises and answers.

Revisions create separate `teacher_transcript` and `teacher_evidence_blocks`. Original `transcript`, timestamps, and machine proofreading remain unchanged. Unedited machine context remains unconfirmed, with uncertain terms and proofreading provenance. Q&A prioritizes the current revision view; undone revisions cease to be current evidence.

Completing a task requires an actual result. Answer scores can be absent; `0` is an explicitly recorded zero, distinct from unscored. Exercises and plans retain citation-time evidence snapshots. Exercises are teacher-entered or teacher-confirmed; the system does not fabricate student answers or infer mastery.

Changed original text/media identity suspends old overlays. Explicitly confirming the new source archives previous goals, revisions, and follow-up records rather than applying them to new evidence.

## Jobs and recovery

Production courseware imports use `JobQueue` with separate Python subprocesses. Frozen desktop workers use the same executable's `--run-worker` mode. Jobs retain queued, running, complete, failed, or cancelled states. A retry creates a new job while retaining the original ID and attempt relationship.

Enqueueing freezes digests of uploads, bundles, actual courseware images, and original audio/video. Workers recheck before processing and publication. Switching the registry to another source video also prevents old jobs from publishing.

Cancellation is checkpoint-based. Native rendering/matching displays a cancelling state until a safe checkpoint. SQLite write transactions prevent cancellation/publication races. Published results cannot be cancelled; a recovery write lock may require retrying later.

Before publication, a recovery journal stores the target bundle digest and old reports. If a worker exits after file publication, the controller verifies hashes and recovers completion. If it exits before commit, recovery acquires the lesson write lock, restores prior reports, and retains retryability. It never overwrites a different bundle published by another operation. The app only stops subprocesses whose handles it owns, never external processes by name.

Legacy direct `CoursewareService(store)` embedding retains thread execution. The main app explicitly injects the persistent queue and subprocess computation. The compatibility entry point is not production job recovery.

## API and startup

Existing `/api/review` routes remain. Additional relative routes include:

- `/lessons/{id}/workspace`: teacher records.
- `/lessons/{id}/workspace/history`: old-source archive.
- `/lessons/{id}/workspace/goals`, `corrections`, `actions`, `assessments`: teacher-record writes.
- `/lessons/{id}/evidence/search`: lesson evidence retrieval.
- `/tasks`, `/tasks/{id}/cancel`, `/tasks/{id}/retry`: persistent jobs.
- `/runtime`: read-only model-file and runtime inspection.

See `/docs` after API startup for request schemas. Writes validate origins, sizes, and field types. Errors retain `error.code` and `error.message`. The API serves the built frontend from one origin.

From the repository root:

```powershell
& .\desktop\start_desktop.ps1
```

The browser entry is `desktop/start_source.ps1`. The entries default to separate workspaces. To share records, explicitly select the same `DataDir` and reuse one backend. Desktop distributions exclude private lessons, model weights, and GPU environments.

## Historical verification and rollback

The pre-change snapshot is local `artifacts/verification/architecture-refactor-20261028-154904/source-before.zip`. Original lesson inputs were not overwritten or migrated; new teacher/job state is in SQLite.

Acceptance separately checked HTTP compatibility, real subprocess PDF imports, crash recovery, cancellation/retries, record persistence, frontend races, and desktop execution. Browser tests used an independent copy of a real lesson and a separate database. Test entries were labeled acceptance examples and did not modify production teacher records. These checks do not establish ASR, OCR, or behavior-accuracy gains.

Detailed results, operation scope, and model boundaries are in local `artifacts/verification/architecture-refactor-20261028-154904/ACCEPTANCE.md`, excluded from the source release.

For rollback, stop services started by the app, back up databases, then restore changed files from the source snapshot. Retaining the new SQLite files preserves teacher records; older code does not read those new records.
