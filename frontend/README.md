# Jingxi Insight frontend

A lesson review workspace built with Vue 3, TypeScript, Vite, and Pinia. See the [application README](../README.md) for startup, models, and data preparation.

## Development and build

Use Node.js 22 or 24 and run the following in this directory:

```powershell
npm ci
npm run dev -- --host 127.0.0.1 --port 5175
```

In another terminal at the repository root, run `python scripts/run_workbench.py --api-only --port 8000`. Vite proxies `/api/review` to `127.0.0.1:8000`. After a production build, FastAPI serves the interface and API from the same origin:

```powershell
npm test
npm run build
```

`npm run build` runs the Vue/TypeScript checks and Vite build. `dist/` contains application and brand assets only; historical classroom images and snapshots from `public/` are not copied into the production bundle. `npm run preview` provides a static preview. Use the backend origin for the complete application.

## Interface and data

English is the default interface language. Original lesson transcripts, OCR quotations, and teacher-entered content retain their source language.

The overview focuses on student images, visible states, overall engagement estimates, and synchronized transcripts. Courseware is available on the Audio and courseware page. Student behavior includes provenance, timelines, manual boxes for missed detections, label corrections, false-detection exclusions, version history, and sample export. The workspace also includes a knowledge graph, review report, and teaching improvement records.

The lesson list comes from `/api/review/lessons`. Empty data, service failures, offline models, and unknown observations have explicit states; the interface does not automatically switch to demo data. The frontend stores no model secrets and does not start GPU inference directly.

| Code | Responsibility |
| --- | --- |
| `src/components/review/` | Lesson workspace and six feature views |
| `src/api/` | API clients, errors, and visual review requests |
| `src/utils/classEngagement.ts` | Shared scoring for frames, trends, and exports |
| `src/utils/studentStates.ts` | State labels and interpretation |
| `scripts/test-*.mjs` | Contracts, records, behavior, visual drafts, and scoring checks |

Derived scores are exported in a separate `derived_engagement` field without changing original observations. Unknown people are not assigned zero scores. Limited-support estimates and human review are displayed separately. See the [scoring notes](../backend/docs/CLASS_ENGAGEMENT_20261029.md) and [visual review notes](../backend/docs/VISUAL_REVIEW_20261029.md).

## Verification

`npm test` runs `test:review`, `test:activity`, `test:vision-review`, and `test:engagement`. Browser checks in `scripts/verify-*.py` require a separate Playwright environment and the corresponding local lesson data. Browser evidence stays in the ignored `output/` directory and is not part of the source package. Several older verification scripts target historical interfaces; their presence is not evidence that the current application passed those checks.

Historical data contracts and interface notes remain in [LESSON_REVIEW.md](LESSON_REVIEW.md). Existing classroom data and local test paths described there are development records, not materials included with a fresh clone.
