# Overall student behavior engagement

The overview and student-behavior detail show overall behavior engagement. The current frame, trend, lesson mean, and exports share `frontend/src/utils/classEngagement.ts`. Scores are derived from validated vision observations; models do not need to run again.

## Weights and calculation

Seven rule weights come from `vision/Class_Detection/configs/pipeline_legacy.yaml`:

| Behavior | Rule score |
| --- | ---: |
| Writing | 85 |
| Reading | 80 |
| Looking up | 85 |
| Turning the head | 30 |
| Raising a hand | 95 |
| Standing | 90 |
| Discussion | 75 |

The calculation retains the original class-level dispersion adjustment. Normalize behavior weights to 0–1, compute the valid objects' mean μ and population standard deviation σ, then use `100 × μ × exp(-0.6 × σ)`. Current data has no actual gaze score, so only the behavior channel is used; no default gaze value is inserted. The method version is `overall_behavior_engagement_v1`.

Within a continuous shot, consecutive observations at most 1.2 seconds apart use `0.3 × previous displayed score + 0.7 × current score`. A shot change, observation gap, or missing previous score resets smoothing. The lesson mean weights displayed scores by the duration of valid sample intervals.

## Coverage and missing data

Only the seven behavior labels with consistent model-classification provenance are scored. Geometric poses, unknown labels, and unrecognized objects are not zero-filled. Score coverage is scored-object count divided by currently detected-object count.

At least one valid behavior object produces an estimate. Fewer than three scored objects or coverage below 60% is labeled a low-support estimate. A frame with no people has no observation; a frame with people but no valid behavior has no score. Missing footage neither lowers the mean to zero nor inherits the preceding shot's score.

The score describes detected objects' visible behavior. Identities may remain unverified, and lesson-specific model accuracy requires human annotation. It supports review and is not an individual student grade.

## Provenance and export

Original `activity_analysis`, legacy `visual_reference`, object predictions, and manual reviews are retained. The legacy rules produced no scores when support was insufficient. The current interface derives a separate score from raw observations without changing detection evidence or invalidating reviews.

JSON exports add `derived_engagement`: source digests, method version, weights, smoothing parameters, frame scores, and lesson mean. Markdown adds the current scoring section while preserving original generation-time evidence. Legacy values and current derived scores remain separately traceable.

## Historical lesson validation — 2026-10-29

The source-language lesson title is “认识氨基酸” (“Introduction to amino acids”). Of 181 sampled frames, 172 had valid behavior scores and nine had no people. The smoothed score at 00:21 was 51.9/100; the duration-weighted mean was 47.8/100.

Browser acceptance independently reproduced the calculation in Python and compared the frame score, trend length, mean, Markdown, and JSON. It also verified that original vision data was unchanged and checked 42 combinations of timestamps, viewport sizes, and zoom. These checks establish agreement with the rules, not independently validated recognition accuracy.
