# Archived visual review implementation plan — 2026-10-29

The original execution guidance selected sequential checked tasks within the visual-review scope. This records completed work and does not start a new implementation.

**Goal:** Confirm each current-frame object, correct labels, exclude false detections, add missed boxes, retain reversible records, and export provenance-backed human-confirmed samples.

**Architecture:** Separate SQLite review tables preserve lesson bundles/model scores. Records bind video digests, full-frame model records, and original-image digests. Per-frame versions protect concurrent writes. The frontend locks the edited frame, isolates drafts by lesson/frame, and exports annotation manifests plus confirmed-student crops.

**Stack:** Existing FastAPI, SQLite, Pillow, Vue 3, TypeScript, and SVG pointer events.

- [x] Backend tests in tests/test_vision_review.py: saving, conflicts, source changes, withdrawal history, invalid coordinates, and confirmed-sample filtering.
- [x] review/vision_review.py: source binding, versions, history, ZIP; review/api/vision_review.py: routes; review/api/application.py: registration.
- [x] Frontend src/types/visionReview.ts, src/api/visionReview.ts, and src/components/review/VisionFrameReview.vue: types/services and blue editor with locks, box drawing, edits, withdrawal, errors, and conflicts.
- [x] LessonStudentState.vue: review entry preserving original model outputs/frames; automated writes use separate test databases.
- [x] Backend regressions, frontend build, browser checks, independent review/fixes, runnable local update, and documentation.

Export creates seven-class crops only for explicitly confirmed students with known states. Unknown roles/states and ignored objects stay in manifests without classification-truth claims. Unreviewed objects are not ground truth. Frames are partially annotated, not complete person-detection labels. Stale records are neither applied nor exported; history stays readable. Reviews do not automatically change lesson-wide engagement curves or establish model accuracy.
