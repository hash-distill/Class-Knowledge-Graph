# Manual vision review and correction

This change added manual correction and sample export while retaining original model outputs. It did not change model weights or engagement rules and was not model-accuracy acceptance.

## Use

Open **Student behavior**, then **Vision review and correction** under the state and engagement section. Interface wording may evolve; these steps describe the review workflow.

1. Seek to a video time and start reviewing the current frame. The editor locks that frame; moving playback does not save edits to another frame.
2. Select a box or its numbered entry to inspect state and role. Changing the state records a human correction. If only role or coordinates change, explicitly confirm the object. A geometric pose does not automatically become a human seven-class behavior label.
3. Enter a reason before excluding a false detection. Draw a box for a missed object, then choose role and state; coordinate inputs also adjust boxes. Fullscreen inspection can be closed with its button or Esc.
4. Save the frame review. Withdrawal preserves history; version history restores older records from the same source. Workspace switches retain the current tab's draft. Save or download a draft backup before reloading or closing.
5. Export lesson samples as a ZIP. Unsaved drafts are excluded. If another window saved the same frame, saving an older version is rejected and the draft is retained. Reload the server version and download the old draft for comparison.

## Export contents

- `manifest.json` records video/image digests, frame time, shot, pixel coordinates, human labels, roles, exclusion reasons, and review versions.
- `frames/` contains JPEG copies of reviewed frames. `original_image_sha256` hashes the original file; a re-encoded JPEG copy may have a different digest.
- `crops/<class>/` includes only explicitly confirmed students with a known behavior. Unknown states, unconfirmed roles, teachers, and excluded objects remain in the manifest without seven-class training crops.
- Unreviewed objects are not negative examples. Frames are partially annotated and must not be used directly as complete person-detection labels. The exporter does not split train/test sets; group later splits by recording to prevent adjacent-frame leakage.
- Frames removed by new analysis are listed in `archived_missing_frames` and `ARCHIVED_FRAMES.txt`, retained in history, and excluded from current samples. Changed images or model records disable the previous review until it is checked again.

## Storage and interpretation

Reviews use the separate `vision_review_versions` table in the workbench's `workspace.sqlite3`. They do not overwrite `bundle.json`, `activity.json`, original images, or weights. Saves bind to the video digest, complete current frame record, and original-image digest; frame versions prevent multi-window overwrites.

Confirmed/excluded counts in the review editor are human records. Detection overlays, lesson-wide temporal charts, and engagement rules continue to use original model results. A partial review does not establish class-wide recall, psychological engagement, or teaching effectiveness, and does not automatically generate an accepted lesson-wide engagement curve.

## Historical validation — 2026-10-29

The full backend suite, including frozen desktop-backend execution, reported 336 passed and four skipped. Fourteen added API tests covered save/restart persistence, conflicts, image changes, source preservation, restoration, coordinate/label validation, role/sample filters, lesson/source boundaries, and export recovery after old-frame deletion.

Frontend type checks, production build, review contracts, behavior analysis, and draft-coordinate tests passed. Browser write tests used a separate preview database; they did not add automated labels to real lessons. Local records and screenshots are in `artifacts/verification/visual-review-20261029/`, excluded from source releases. A new installer was built; its installation wizard was not run.
