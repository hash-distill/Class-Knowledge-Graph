# Recorded lesson review frontend (v3)

The home page loads recorded lessons from `GET /api/review/lessons`, then reads the selected lesson package. An empty lesson list shows an empty state. Service failures show an error and retry control without falling back to demo data. Audio, courseware, and graph evidence use original timestamps from the same lesson package.

## Usage

1. Start the review backend at `127.0.0.1:8000`, then run `npm run dev -- --host 127.0.0.1 --port 5175`.
2. Open the frontend home page. Selecting a lesson synchronizes the video, whole-lesson timeline, and audio/courseware panels.
3. Select a transcript segment, observed slide segment, concept, or evidence button to seek the source video. Multiple independent intervals for a concept remain separate; gaps between the first and last observations are not merged into one continuous interval.
4. The review assistant supports Whole lesson and Current segment scopes. Each request retains the lesson, time, and scope at submission. Retrying a failed request uses that same context.
5. Answers show the actual `source`, `model`, and `request_id`. Citations must exist in the current lesson evidence and match the source times. HTTP errors, timeouts, and citation validation failures show errors without generating local template answers.
6. Full JSON exports retain the complete lesson package. Markdown exports retain the report, original filename and SHA-256, evidence times, generation method, provenance, and limitations.

Uploaded courseware alignment accepts PDF/PPTX/PPT files, shows asynchronous processing status, previews source pages, and lets teachers confirm or exclude pages for actual observed intervals. Removing an association restores the original evidence. An unmatched interval shows no page in follow mode rather than falling back to page one. Courseware-assisted revisions separately show original text, suggestions, page sources, and replay links; JSON and Markdown exports retain the provenance. `coursewareValidation.ts` validates the correspondence between courseware citations, revision character ranges, raw transcripts, and complete retrieval blocks. See [COURSEWARE.md](../backend/docs/COURSEWARE.md) for backend requirements and usage.

Optional `evidence_blocks` are retrieval blocks built from consecutive native ASR segments (B IDs). Machine corrections are stored separately as P segments and PB evidence blocks, preserving original T segments, replacement text, supporting sources, and uncertainties. Corrected text is shown by default; Compare original transcript reveals the raw recognition output. Citations must match their evidence text and times. Teacher revisions have their own verification states; unchanged machine context is not treated as teacher verified.

Lesson duration comes directly from `lesson.duration_sec` and does not depend on person snapshot counts. In the historical `lesson1-real` package, the audio-based duration was 1145.092063 seconds (19:05 in the interface); the video stream was approximately 1144.433313 seconds, a difference of about 0.66 seconds. The current backend package determines the actual processing scope and limitations.

When synchronized behavior video from the same lesson is unavailable, the behavior panel explicitly reports missing data and creates no participation curve, engagement score, or mastery estimate. Machine transcription, OCR, source-extracted concepts, and model-proposed relationships are not human ground truth.

## Structure

- `src/components/review/LessonReviewWorkspace.vue`: workspace layout and lesson selection.
- `LessonPlayer.vue`: video, independent whole-lesson clock, and evidence timeline.
- `LessonEvidence.vue`: transcript search, follow playback, and observed slide segments.
- `LessonGraph.vue`: package-supplied nodes, relationships, states, and evidence; no random activity generation.
- `LessonAssistant.vue`: actual model Q&A, provenance, citations, and error retries.
- `LessonReport.vue`: report and provenance exports.
- `src/stores/lessonReview.ts`: lesson, playback, selection, and Q&A state. Evidence indexes and concept intervals are cached per lesson.
- `src/utils/lessonReview.ts`: source contracts, time and citation validation, evidence indexes, and Markdown exports.
- `src/api/reviewService.ts`: same-origin API client. Model secrets and configuration remain in the backend.

## Development and production

The Vite development server proxies `/api/review` unchanged to `http://127.0.0.1:8000`. Media must use same-origin `/api/review/media/...` paths; video from other lessons is not substituted.

Run `npm run build` for type checks and production output. `dist` contains application scripts, styles, and brand icons only. The review backend serves media per lesson. Production reverse proxies must preserve the same-origin `/api/review` contract and support media Range requests.

The workspace preserves the lesson and feature view in the `lesson` and `view` query parameters. Feature navigation retains the same video element, playback time, and form state.

## Verification

```powershell
npm run test:review
npm run test:activity
npm run build

# Actual backend, media, and local model checks require the corresponding services.
python scripts/verify-proofread-students.py
python scripts/verify-unified-workbench.py
```

Contract tests cover clocks without snapshots, empty lessons, provenance completeness, gaps between repeated concepts, cross-lesson and mistimed citations, model failures without fabricated answers, and retry context. Browser scripts cover transcript-to-video/courseware synchronization, concept evidence navigation, model provenance and citations, exports, and a 390 px layout. Historical `browser-verification.json` records distinguish mocked contracts from actual API/model verification; those checks do not establish human-rated ASR/OCR accuracy or automatic relationship accuracy. A script requires the corresponding data and interface version before its result can be treated as current verification.
