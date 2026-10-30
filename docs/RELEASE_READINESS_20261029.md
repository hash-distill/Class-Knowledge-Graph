# Source release readiness record — 2026-10-29

This historical record describes source entry points, documentation, and GitHub layout prepared for the complete Jingxi Insight application. Existing classroom and model files were preserved. The subsequent publication round includes the application source, English directory names, and an English-default interface; source-language lesson evidence is retained.

## Changes in the recorded preparation

- Added workspace initialization, read-only checks, source launch, and source export at the root; initialization preserves existing registries.
- Resolved lesson-relative paths from the registry file. Legacy evidence-directory entries remain readable; missing source-video entries produce explicit media errors.
- Allowed API-only startup before the frontend build. A service started before that build is not treated as serving static pages afterward. Configuration changes request a restart in the original terminal while preserving the running service.
- Separated application and media/model dependencies, included pytesseract for scanned courseware, and fixed the desktop packaging Pillow version conflict.
- Removed default private SSH hosts, interpreter paths, and GPU IDs from secondary Chinese ASR; remote transfer requires explicit configuration.
- Updated root, backend, frontend, and vision documentation, linking historical vision records to current scoring and manual review.
- Removed an unused Vue DevTools build dependency, upgraded Vite to 6.4.5, and updated compatible dependencies and lockfiles.
- Added GitHub Actions for backend/frontend tests, production builds, and source audits.

## Historical validation

A fresh Python 3.11 application environment reported **345 passed, 6 skipped** backend tests. Skips concerned separately configured frozen backends and optional conditions. This recorded round did not rerun the Windows installation wizard or train models.

Frontend contract, teaching-record, behavior-analysis, manual-review draft, and scoring tests passed, as did Vue/TypeScript checks and the Vite 6.4.5 production build. After dependency updates, `npm audit` reported **0 known vulnerabilities** for that query and lockfile; this is a dated result.

Independent review checked empty-workspace startup, health/lesson/runtime endpoints, service reuse, frontend build transitions, and asset integrity. No unresolved P1/P2 findings remained within the reviewed scope. Unit tests, builds, and source audits do not establish real classroom model accuracy.

Detailed local logs, README backups, JUnit results, and screenshots are in the ignored `backend/artifacts/verification/github-ready-20261029` directory. These records are excluded from public source packages. The source package has a separate per-file manifest; original classroom files are excluded.

## Publication scope

The GitHub README title and description use Jingxi Insight while the repository URL remains unchanged. Application source is prepared and pushed from a separate checkout while preserving existing remote commit history. Publication excludes private media, generated evidence, workspaces, and weights. The overall license remains for the maintainer to establish; existing third-party terms continue to apply.
