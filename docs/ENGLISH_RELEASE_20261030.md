# English application release

The application is named **Jingxi Insight**. Source directories are `backend/`,
`frontend/`, `vision/`, and `desktop/`. Interface text, accessibility labels,
errors, generated report headings, launcher help, model instructions, comments,
and project documentation use English. Desktop installation defaults to English.

Original transcripts, OCR, teacher quotations, linguistic dictionaries, and
language-preservation fixtures keep their source language. Historical generated
labels remain only as compatibility identifiers. See
[language data notes](../backend/tests/LANGUAGE_DATA.md). Translating model
instructions changes prompt conditions; earlier model measurements do not
independently validate the translated prompts.

English questions now use the same whole-lesson sampling and multimodal evidence
balancing as Chinese questions, including case-insensitive intent matching.
Behavior overlays replace legacy generated report sections and stale missing-data
notices without modifying the saved source bundle.

## Windows crash recovery

Regression tests exposed SQLite error 1546, `SQLITE_IOERR_TRUNCATE`, during worker
crash recovery. It matches the upstream
[Windows mapped-SHM issue](https://sqlite.org/forum/info/ba7afbd0e81501118563617430fe06d0aca052e55ece2b0d457f8e16fcf85bba).
The task controller now retains an idle SQLite connection while workers and
polling connections run, and closes it after shutdown. Existing transactions,
publication recovery, and schema values are preserved; I/O errors are not retried
or suppressed. A deterministic Windows regression covers mapped-SHM reads,
controller restart, and handle release. An orphaned mapping already present at
fresh startup still requires the upstream SQLite runtime fix.

## Verification

- Backend integration: 353 passed and 6 skipped on Python 3.11.15 / SQLite 3.53.1.
- Additional mapped-SHM regression: passed; the previous implementation failed.
- Vision module: 82 tests passed.
- Frontend contracts, visual-review drafts, scoring, type checks, and production
  build passed.
- English empty-workspace startup and runtime inventory passed.
- Real lesson scoring matched the independent reference across 181 frames;
  42 rendered spacing checks passed without page errors.

Linux CI installs FFmpeg for synthetic-video integration tests. Model lifecycle
tests explicitly select the Windows adapter or POSIX rejection path without
starting a model or changing the host OS.

Checks use isolated application state. Source media, model weights, teacher
records, and original inference artifacts are excluded from publication.
This release does not include a rebuilt native installer or a new model-quality
benchmark.
