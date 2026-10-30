# Jingxi Insight vision module

Offline analysis extracts people, poses, and visible behavior from classroom recordings, producing timed observations with provenance for the workbench. Start the full application from [Jingxi Insight](../README.md). Historical root-level vision code now lives under `vision/`.

## Current path

Person localization fuses full-frame and overlapping-tile YOLO26n/YOLO26n-pose outputs in source-image coordinates. A per-person MobileNetV3 Small crop classifier provides seven states: writing, reading, looking up, turning the head, raising a hand, standing, and discussion. Low-confidence/pose-conflicting results stay unknown; geometric clues are labeled separately.

The experimental weights are about 3.15 MB. Source distributions exclude weights, classroom images, and training data. Implementation is in `Class_Detection/src/dense_detection.py`, `dense_inference.py`, and classifier modules. Script arguments/model files must match their provenance configuration.

Historical ST-GCN, gaze, dual-detector, VSAM, and CAS/CTES pipelines remain for comparison. Their presence does not establish a validated current end-to-end workbench pipeline. [Training and inference](TRAINING_AND_INFERENCE.md) is a historical manual, not application setup.

## Environment and execution

Use a separate Python environment, device-appropriate PyTorch/torchvision, and `requirements-inference.txt`. Keep model imports out of the API process and vision dependencies separate from lightweight application dependencies. Training, data configuration, and GPU inspection are optional development workflows.

See [Vision change record](../backend/docs/VISION_IMPROVEMENT_20261029.md) for localization, parameters, and recorded outputs. Workbench data binds to video digests, times, and image provenance; do not splice unrelated lessons. Inspect each script's `--help`, then provide real weights/media before execution.

## Correction and scoring

[Manual review/sample export](../backend/docs/VISUAL_REVIEW_20261029.md) produces classification crops only for explicitly confirmed students with known behavior. Partially labeled frames are not complete detection labels. Group training/test splits by recording or classroom.

The interface uses [overall behavior engagement rules](../backend/docs/CLASS_ENGAGEMENT_20261029.md), retaining original observations and legacy values. Historical fallback-person-as-listening assumptions, default high gaze values, or baseline scores for all unknown objects are not valid current observations.

## Limitations

Existing given-box crop Macro-F1 is 0.81694, with adjacent frames from the same classroom across training/test. It may be optimistic and is not target-classroom, detection, or end-to-end accuracy. Misses, duplicate boxes, occlusion, mirrors, and role errors require independent annotation.

Box counts are not class attendance; tracks are not identity recognition; visible behavior is not psychological engagement/mastery. Images, datasets, and weights retain separate provenance/terms and are excluded from source exports.
