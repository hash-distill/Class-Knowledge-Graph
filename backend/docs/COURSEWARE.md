# Independent courseware: import, alignment, and terminology suggestions

The workbench supports PDF, PPTX, and legacy binary PPT. It preserves original ASR, audio/video, and recorded-screen OCR. Original courseware is stored by SHA-256; every terminology change is a separate, traceable suggestion. No model training is required.

## Workbench workflow

1. Select a lesson and choose a file in independent courseware alignment. Limits: 64 MiB and 200 pages.
2. A background job renders original pages, extracts text or OCR, and matches them to observed screen intervals. The original lesson stays visible during processing; failed jobs do not publish partial results.
3. Select pages/time intervals to replay the recording. Automatic, ambiguous, unmatched, confirmed, and no-page states remain separate. A filter shows items needing confirmation.
4. Compare with the source video and assign a page, or no corresponding page/teacher shot, to an existing observation interval. Saving recomputes citations and terminology suggestions. Repeated visits and skipped pages remain separate rather than merging all appearances of one page.
5. Compare original/suggested wording and page provenance in terminology suggestions; click a time to replay audio. The assistant reads suggestions and aligned original pages, with prompts explicitly stating these are not a human transcript.
6. JSON includes original speech, courseware, mappings, revisions, and provenance. Markdown includes the file SHA, mapping states, and original/suggested text. Removing the association restores pre-import lesson evidence while retaining historical originals.

On the recorded Windows machine, PPT/PPTX conversion used installed PowerPoint in a hidden read-only export session, including legacy PPT. Macros are disabled. If PowerPoint is already running, the converter refuses to take it over and reports why. Close your own PowerPoint session and retry, or export to PDF. PDF remains available without PowerPoint; no LibreOffice fallback is provided.

## Alignment and unknown states

Independent courseware has no intrinsic lesson timestamps. The system preserves page text, images, and file digests, then compares them with S-numbered observations from the same video:

- Literal text similarity accommodates some animations and missing OCR text; near-tied candidates remain undecided.
- ORB keypoints and RANSAC geometry support image matching despite scale, animation, or annotation differences.
- The first/second candidate margin uses unclipped scores to avoid false certainty at thresholds. Similarity is an engineering heuristic, not a calibrated probability or accuracy.
- Only `auto_matched` / `confirmed` pages become timed citations. Other pages receive no invented times. Citations copy the corresponding S interval; boundary precision depends on the original screen-sampling interval.
- Teacher close-ups, blank pages, near-duplicates, and insufficient text/image evidence may remain unmatched. Human confirmation applies only to existing observations.

PyMuPDF and OpenCV once stalled in the same process on the recorded machine. Image matching now uses a separate process with a timeout. Rendering has page-count and per-page/total pixel budgets. Damaged or encrypted files are not replaced with simulated pages.

## Chinese terminology suggestions

Terms are extracted from actual aligned page text, without a preset wrong-to-correct replacement table. A suggestion requires identical pronunciation ignoring tone, identical length, an unambiguous target, and a limited edit scope. It uses the page's actual spelling; simplified/traditional differences alone are not ASR errors.

Source-language mechanism example: a page containing “正弦定理” (sine rule) and “向量” (vector) can support suggestions for ASR text “鄭賢定理” and “相量” at the corresponding time. The Chinese strings are retained because their pronunciation is the evidence for the example. Suggestions include page and match IDs. Numbers, negation, and formula-like clauses are protected, including full-width symbols, parentheses, and whitespace. No aligned evidence, differing pronunciation, or multiple homophones means no change.

This can reduce homophone errors explained by current courseware. It does not guarantee the spoken truth or recover every omission, spoken formula, or non-homophone error. Q&A uses full B blocks with neighboring context; original counts such as 745/646 segments do not change through revision. Real accuracy gains require human reference labels; no percentage improvement is claimed.

## Installation and CLI

Run from `backend/` in the existing media environment:

```powershell
uv pip install --python .venv-media/Scripts/python.exe -r requirements-courseware.txt
./start_review.ps1 -Restart
```

Import into a registered lesson:

```powershell
& .venv-media/Scripts/python.exe scripts/import_courseware.py --lesson-id lesson1-real --file 'D:\lesson-materials\sine-rule.pptx'
```

The CLI and browser use the same core logic. Avoid editing one lesson concurrently through both. Changed source media invalidates old mappings; realign rather than applying old times to a new recording.

Media preparation supports `--language zh` (default) and `--language en`, followed by the existing `build_review.py` bundle step. Rebuilding verifies and retains valid courseware overlays and explicitly rejects reuse after provenance changes.

## Public acceptance materials

Historical acceptance used the full MIT OCW 6.0002 Fall 2016 Lecture 1 video and official 31-page PDF: [official video](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/resources/lecture-1-introduction-and-optimization-problems/) and [official slides](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/resources/mit6_0002f16_lec1/). These are not the slides for the existing sine-rule recording.

Local `artifacts/public-courseware/mit-6-0002-f2016-lec01/` records URLs, file digests, source terms, and visual-pairing evidence. Official subtitles were reference-only; project ASR was rerun from full audio. This English lesson checked real alignment and Q&A. Separate, explicitly labeled Chinese-homophone mechanism tests must not be combined with it as evidence of Chinese recognition accuracy.

The provenance manifest records MIT materials' CC BY-NC-SA 4.0 terms and third-party illustration exclusions in the PDF. Downloaded materials remain ignored local acceptance inputs and are not packaged as project-owned open-source assets.
