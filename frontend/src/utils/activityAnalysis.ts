import type { ActivityFrame, ActivityParticipant, LessonPackage } from '../types/lessonReview'
import { engagementExport } from './classEngagement'

// Activity is an optional observed overlay of the original recording clock.
export function validateActivityAnalysis(data: any, media: (url: unknown) => string) {
  if (data.activity_analysis === undefined) return
  const fail = (field: string): never => { throw new Error(`Invalid activity analysis data: ${field}`) }
  const obj = (x: any, key: string): Record<string, any> => x && typeof x === 'object' && !Array.isArray(x) ? x : fail(key)
  const array = (x: any, key: string): any[] => Array.isArray(x) ? x : fail(key)
  const text = (x: any, key: string): string => typeof x === 'string' && x.trim() ? x : fail(key)
  const number = (x: any, key: string, max = Infinity): number => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= max ? x : fail(key)
  const count = (x: any, key: string): number => Number.isInteger(number(x, key)) ? x : fail(key)
  const duration = data.lesson.duration_sec
  const timed = (x: any, field: string) => {
    obj(x, field); text(x.id, `${field}.id`)
    number(x.start_sec, `${field}.start_sec`, duration); number(x.end_sec, `${field}.end_sec`, duration)
    if (x.end_sec <= x.start_sec) fail(`${field} Time interval is empty or reversed`)
  }
  const activity = obj(data.activity_analysis, 'activity_analysis')
  if (activity.schema_version !== '1.0' || activity.status !== 'complete') fail('Only fully published analysis is accepted for version 1.0 analysis')
  if (text(activity.source_sha256, 'source_sha256') !== data.lesson.source_video.sha256) fail('Source hash does not match the original video')
  if (!number(activity.sampling_interval_sec, 'sampling_interval_sec')) fail('Sampling interval must be greater than zero')
  const knownIds = new Set<string>([
    ...data.transcript, ...data.slides, ...data.alignment, ...(data.evidence_blocks || []), ...(data.courseware_evidence || []),
    ...(data.transcript_revisions || []), ...(data.revised_evidence_blocks || []).filter((row: any) => row.kind !== 'transcript'),
    ...(data.transcript_proofreading?.segments || []), ...(data.transcript_proofreading?.alternate_evidence || []), ...(data.proofread_evidence_blocks || []),
    ...(data.courseware?.pages || []), ...(data.courseware?.matches || []),
  ].map((row: any) => row.id))
  const unique = (row: any) => { if (knownIds.has(row.id)) fail(`Duplicate evidence ID ${row.id}`); knownIds.add(row.id) }
  const contexts = array(activity.context_segments, 'context_segments')
  for (const row of contexts) {
    timed(row, 'context_segment'); unique(row); text(row.text, 'context.text'); text(row.model_id, 'context.model_id')
    if (row.method !== 'trained_context') fail('Context must declare its trained_context source')
    const labels = new Set<string>()
    for (const label of array(row.labels, 'context.labels')) {
      obj(label, 'context.label'); text(label.label, 'context.label'); text(label.name, 'context.name'); number(label.probability, 'context.probability', 1)
      if (labels.has(label.label)) fail('Duplicate context label'); labels.add(label.label)
    }
  }
  const frames = array(activity.frames, 'frames')
  let previous: any = null
  const sceneIds = new Set<number>()
  for (const frame of frames) {
    timed(frame, 'frame'); unique(frame)
    number(frame.timestamp_sec, 'frame.timestamp_sec', duration)
    if (frame.timestamp_sec < frame.start_sec || frame.timestamp_sec >= frame.end_sec) fail('Frame timestamp must fall within the evidence interval')
    if (frame.end_sec - frame.start_sec > activity.sampling_interval_sec + 1e-6) fail('Frame interval exceeds the sampling interval')
    if (!count(frame.scene_id, 'frame.scene_id') || typeof frame.scene_cut !== 'boolean') fail('Scene marker')
    if (previous && (frame.start_sec < previous.end_sec || frame.scene_id < previous.scene_id || (frame.scene_id !== previous.scene_id) !== frame.scene_cut)) fail('Frames overlap or scene boundaries are inconsistent')
    sceneIds.add(frame.scene_id)
    text(frame.status, 'frame.status')
    if (typeof frame.image_url !== 'string') fail('frame.image_url')
    if (frame.image_url) {
      media(frame.image_url)
      let decoded: string
      try { decoded = decodeURIComponent(frame.image_url) } catch { fail('Invalid frame media encoding') }
      const prefix = `/api/review/media/${encodeURIComponent(data.lesson.id)}/`
      if (!frame.image_url.startsWith(prefix) || /(?:^|\/)\.\.(?:\/|$)|[\\?#]/.test(decoded!)) fail('Frame media must use this lesson’s same-origin directory')
    }
    count(frame.detected_people, 'frame.detected_people'); count(frame.known_people, 'frame.known_people')
    if (frame.known_people > frame.detected_people) fail('People with usable rule scores cannot exceed the detected count')
    if (frame.coverage !== null) {
      number(frame.coverage, 'frame.coverage', 1)
      const ratio = frame.detected_people ? frame.known_people / frame.detected_people : 0
      if (Math.abs(frame.coverage - ratio) > .0001) fail('Coverage does not match the observed person count')
    }
    if (frame.participation_index !== null) number(frame.participation_index, 'frame.participation_index', 100)
    if (!frame.known_people && frame.participation_index !== null) fail('With no usable person rule scores, the rule index must be null')
    for (const [label, value] of Object.entries(obj(frame.behaviors, 'frame.behaviors'))) { text(label, 'behavior.label'); count(value, `behavior.${label}`) }
    const tracks = new Set<string>()
    const participants = array(frame.participants, 'frame.participants')
    if (frame.width !== undefined && !number(frame.width, 'frame.width')) fail('Frame width must be greater than zero')
    if (frame.height !== undefined && !number(frame.height, 'frame.height')) fail('Frame height must be greater than zero')
    if (participants.length > frame.detected_people) fail('Participant list exceeds the detected count')
    for (const participant of participants) {
      obj(participant, 'participant'); text(participant.track_id, 'participant.track_id')
      if (!participant.track_id.startsWith(`S${frame.scene_id}-T`) || tracks.has(participant.track_id)) fail('Tracks must be unique within a scene. Tracks cannot be linked across scenes')
      tracks.add(participant.track_id)
      const box = array(participant.bbox, 'participant.bbox')
      if (box.length !== 4) fail('Bounding boxes must have four coordinates')
      box.forEach(value => number(value, 'participant.bbox'))
      if (box[2] <= box[0] || box[3] <= box[1]) fail('Invalid bounding box extent')
      if ((frame.width && box[2] > frame.width + 1) || (frame.height && box[3] > frame.height + 1)) fail('Bounding box exceeds the source frame')
      text(participant.label, 'participant.label'); text(participant.source, 'participant.source'); number(participant.confidence, 'participant.confidence', 1)
      if (participant.score !== null) number(participant.score, 'participant.score', 1)
      if (participant.student_state !== undefined) validateStudentState(participant.student_state, obj, text, number, array, fail)
      if (participant.posture_tags !== undefined) for (const state of array(participant.posture_tags, 'posture_tags')) validateStudentState(state, obj, text, number, array, fail)
      if (participant.keypoints !== undefined) {
        const points = array(participant.keypoints, 'keypoints')
        if (points.length !== 0 && points.length !== 17) fail('Pose keypoints must be empty or contain17items')
        for (const point of points) { if (array(point, 'keypoint').length !== 3) fail('Keypoints must contain coordinates and confidence'); number(point[0], 'keypoint.x'); number(point[1], 'keypoint.y'); number(point[2], 'keypoint.confidence', 1) }
      }
    }
    if (frame.student_state_counts !== undefined || frame.participation !== undefined) {
      const states = obj(frame.student_state_counts, 'student_state_counts')
      const expectedStates: Record<string, number> = {}
      for (const participant of participants) {
        const state = participant.student_state
        if (!state) fail('State distribution is missing per-person states')
        expectedStates[state.label] = (expectedStates[state.label] || 0) + 1
      }
      if (frame.detected_people > participants.length) expectedStates.unknown = (expectedStates.unknown || 0) + frame.detected_people - participants.length
      for (const key of new Set([...Object.keys(states), ...Object.keys(expectedStates)])) if (count(states[key] ?? 0, `student_state_counts.${key}`) !== (expectedStates[key] || 0)) fail('State distribution does not match per-person records')
      const participation = obj(frame.participation, 'participation')
      const total = frame.detected_people, active = ['read', 'write', 'raise_hand'].reduce((n, key) => n + (states[key] || 0), 0), observed = total - (states.unknown || 0)
      for (const [key, expected] of Object.entries({ total_people: total, observed_people: observed, unknown_people: total - observed, active_people: active, scored_people: active })) if (count(participation[key], `participation.${key}`) !== expected) fail(`participation.${key} does not match the state records`)
      for (const [key, numerator] of Object.entries({ coverage: observed, active_fraction: active, score_coverage: active })) {
        if (!total) { if (participation[key] !== null) fail(`With no people, ${key} must be unknown`) }
        else if (Math.abs(number(participation[key], `participation.${key}`, 1) - numerator / total) > .0001) fail(`participation.${key} does not match the state records`)
      }
      const weights = obj(participation.weights, 'participation.weights')
      for (const [key, value] of Object.entries(weights)) { if (!['read', 'write', 'raise_hand'].includes(key)) fail('Posture cues cannot be assigned engagement scores'); number(value, `participation.weights.${key}`, 1) }
      if (!active) { if (participation.index !== null) fail('Reference engagement must be unknown without explicit participation actions') }
      else {
        const expected = ['read', 'write', 'raise_hand'].reduce((n, key) => n + (states[key] || 0) * number(weights[key], `participation.weights.${key}`, 1), 0) / active * 100
        if (Math.abs(number(participation.index, 'participation.index', 100) - expected) > .05) fail('Reference engagement does not match the action rules')
      }
      if (!['observed', 'low_evidence', 'no_people'].includes(participation.status) || participation.semantics !== 'visible_participation_rule_v2' || participation.population_scope !== 'visible_people_roles_unverified') fail('Invalid visible participation source or interpretation scope')
    }
    if (frame.visual_reference !== undefined) validateVisualReference(frame, previous, fail)
    if (frame.detection_review !== undefined) {
      const review = obj(frame.detection_review, 'detection_review')
      if (!['full_plus_tiles_pose_behavior_fusion', 'full_plus_tiles_person_pose_then_crop_classification'].includes(review.method) || review.population_recall_evaluated !== false) fail('Invalid tiled detection method or evaluation scope')
      if (count(review.retained_people, 'retained_people') !== frame.detected_people) fail('Retained tiled boxes do not match the current detected count')
      if (count(review.proposal_count, 'proposal_count') < frame.detected_people) fail('Detection proposals are fewer than retained people')
      count(review.excluded_detection_proposals, 'excluded_detection_proposals'); count(review.tile_windows, 'tile_windows'); number(review.wall_seconds, 'wall_seconds')
      if (review.previous_detected_people !== undefined) count(review.previous_detected_people, 'previous_detected_people')
    }
    previous = frame
  }
  const summary = obj(activity.summary, 'summary')
  const expected = { sampled_frames: frames.length, scene_count: sceneIds.size, person_observations: frames.reduce((n, f) => n + f.detected_people, 0), known_observations: frames.reduce((n, f) => n + f.known_people, 0) }
  for (const [key, value] of Object.entries(expected)) if (count(summary[key], `summary.${key}`) !== value) fail(`summary.${key} does not match frame observations`)
  obj(activity.models, 'models')
  array(activity.limitations, 'limitations').forEach(line => text(line, 'limitation'))
  for (const row of array(activity.evidence, 'evidence')) {
    timed(row, 'evidence'); unique(row); text(row.text, 'evidence.text')
    if (row.kind !== 'behavior') fail('Activity evidence kind must be behavior')
    // An evidence interval must be covered by actual sampled frames or context records, with no unknown gap.
    const supports = [...frames, ...contexts].filter(s => s.start_sec < row.end_sec && s.end_sec > row.start_sec).sort((a, b) => a.start_sec - b.start_sec)
    let covered = row.start_sec
    for (const support of supports) { if (support.start_sec > covered + 1e-6) break; covered = Math.max(covered, support.end_sec) }
    if (covered < row.end_sec - 1e-6) fail(`Activity evidence ${row.id} extends beyond observed intervals`)
  }
}

function validateStudentState(state: any, obj: any, text: any, number: any, array: any, fail: (field: string) => never) {
  obj(state, 'student_state'); text(state.name, 'student_state.name')
  if (!['read', 'write', 'raise_hand', 'head_up', 'head_lowered', 'hand_near_head', 'standing', 'unknown', 'lookup', 'turn_head', 'stand', 'discuss'].includes(state.label)) fail('Unknown student state label')
  if (['lookup', 'turn_head', 'stand', 'discuss'].includes(state.label) && !['pretrained_detector', 'trained_classifier'].includes(state.source)) fail('Pose rules cannot be used as seven-class model labels')
  if (!['pretrained_detector', 'trained_classifier', 'pose_geometry_rule', 'unavailable'].includes(state.source)) fail('Invalid student state source')
  if (state.confidence !== null) number(state.confidence, 'student_state.confidence', 1)
  array(state.evidence, 'student_state.evidence').forEach((value: unknown) => text(value, 'student_state.evidence'))
  array(state.keypoint_indices, 'student_state.keypoint_indices').forEach((value: unknown) => { if (!Number.isInteger(number(value, 'keypoint_index', 16))) fail('Keypoint indices must be integers') })
}

function validateVisualReference(frame: ActivityFrame, previous: ActivityFrame | undefined, fail: (field: string) => never) {
  const ref = frame.visual_reference!
  const weights: Record<string, number> = { read: .8, write: .85, lookup: .85, raise_hand: .95, discuss: .75 }
  if (!ref || typeof ref !== 'object' || ref.semantics !== 'configured_visible_behavior_reference_v1' || ref.calibrated_engagement_probability !== false || ref.population_scope !== 'directly_detected_people_roles_unverified' || ref.weight_source !== 'legacy_project_action_policy_subset') fail('Invalid visual rule reference source or interpretation scope')
  if (!ref.weights || Object.keys(ref.weights).length !== Object.keys(weights).length || Object.entries(weights).some(([key, value]) => ref.weights[key] !== value)) fail('Visual rule reference weights do not match the declaration')
  const scored = frame.participants.filter(person => ['pretrained_detector', 'trained_classifier'].includes(person.source) && person.student_state?.source === person.source && person.label === person.student_state.label && Object.prototype.hasOwnProperty.call(weights, person.label))
  const total = frame.detected_people, support = total ? scored.length / total : null
  const enough = total >= 3 && support !== null && support >= .6 && scored.length > 0
  const index = enough ? scored.reduce((sum, row) => sum + weights[row.label]!, 0) / scored.length * 100 : null
  const old = previous?.visual_reference
  const smooth = index === null ? null : old?.smoothed_index !== undefined && old.smoothed_index !== null && previous!.scene_id === frame.scene_id && frame.timestamp_sec - previous!.timestamp_sec <= 1.2 ? .3 * old.smoothed_index + .7 * index : index
  const same = (value: number | null, expected: number | null) => expected === null ? value === null : typeof value === 'number' && Number.isFinite(value) && Math.abs(value - expected) <= .0002
  if (ref.scored_people !== scored.length || ref.total_people !== total || !same(ref.support_fraction, support) || ref.minimum_people !== 3 || ref.minimum_support_fraction !== .6 || !same(ref.index, index) || !same(ref.smoothed_index, smooth)) fail('Visual rule reference does not match valid state support or continuous observations')
  if (ref.status !== (!total ? 'no_people' : enough ? 'estimated' : 'insufficient_support') || ref.smoothing?.previous !== .3 || ref.smoothing.current !== .7 || ref.smoothing.maximum_gap_sec !== 1.2) fail('Invalid visual rule reference smoothing rules')
}

export function participationRuns(frames: ActivityFrame[]): ActivityFrame[][] {
  const runs: ActivityFrame[][] = []
  let run: ActivityFrame[] = []
  for (const frame of frames) {
    const previous = run[run.length - 1]
    if (frame.participation_index === null || (previous && (frame.scene_id !== previous.scene_id || frame.start_sec > previous.end_sec + 1e-6))) {
      if (run.length) runs.push(run)
      run = []
    }
    if (frame.participation_index !== null) run.push(frame)
  }
  if (run.length) runs.push(run)
  return runs
}

export function sceneTrackRows(frames: ActivityFrame[], sceneId: number): { track_id: string; cells: (ActivityParticipant | null)[] }[] {
  const sceneFrames = frames.filter(frame => frame.scene_id === sceneId)
  const ids = [...new Set(sceneFrames.flatMap(frame => frame.participants.map(participant => participant.track_id)))]
  return ids.map(track_id => ({ track_id, cells: sceneFrames.map(frame => frame.participants.find(participant => participant.track_id === track_id) || null) }))
}

export function readableModelFields(models: Record<string, unknown>): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = []
  const obj = (value: unknown): Record<string, any> => value && typeof value === 'object' ? value as Record<string, any> : {}
  const add = (label: string, value: unknown) => {
    if (typeof value === 'number' && Number.isFinite(value)) rows.push({ label, value: String(value) })
  }
  const context = obj(models.context), model = obj(context.model), metrics = obj(context.metrics)
  const observation = obj(models.observation)
  if (observation.pipeline === 'dense_person_then_seven_crop_observation_v2') {
    rows.push({ label: 'Behavior recognition in the current frame', value: 'Full-frame and tiled person detection → pose fusion and deduplication → per-person seven-class candidates from MobileNetV3 Small. Not yet reviewed against human annotations for this lesson' })
    const classifier = Array.isArray(observation.models) ? observation.models.find((row: any) => row.task === 'person_crop_classification') : undefined
    if (classifier) {
      rows.push({ label: 'State classifier weights', value: `${classifier.architecture} · ${(classifier.weight_bytes / 1e6).toFixed(2)} MB` })
      add('Given annotated crops / Test person count', obj(classifier.test_metrics).samples)
      add('Given annotated crops / Test Macro-F1', obj(classifier.test_metrics).macro_f1)
      rows.push({ label: 'Test scope', value: 'The test and training images include the same classrooms and adjacent times. These results do not establish independent classroom validation, accuracy for this lesson, or end-to-end detection accuracy' })
      rows.push({ label: 'Roles and engagement', value: 'Teacher and student roles are unconfirmed; Neither classification confidence nor recognizable-state support is a probability of psychological engagement' })
    }
  } else if (models.observation) rows.push({ label: 'Behavior recognition in the current frame', value: 'Pretrained behavior detection and pose rules; Not yet evaluated against human annotations for this lesson' })
  if (models.context) {
    rows.push({ label: 'Teaching context prediction', value: context.status === 'unavailable' ? 'Model temporarily unavailable' : 'Whole-clip classification prediction. Interpreted separately from per-person behavior' })
    add('Context model / Total dataset samples', model.dataset_count)
    add('Context model / Training samples', model.training_count)
    for (const [part, label] of [['train', 'Training'], ['val', 'Validation'], ['test', 'Test']]) add(`Context model / ${label} samples`, obj(metrics.split_counts)[part!])
    for (const [part, label] of [['validation', 'Validation set'], ['test', 'Test set']]) {
      add(`Context model / ${label} Macro-F1`, obj(metrics[part!]).macro_f1)
      add(`Context model / ${label} mean average precision`, obj(metrics[part!]).mean_average_precision)
    }
  }
  if (models.aric) {
    rows.push({ label: 'Auxiliary behavior data training records', value: 'Offline research only. It is not used for per-person predictions in the current frame' })
    const report = obj(obj(models.aric).report), test = obj(report.test)
    add('Offline model / Test Macro-F1', test.macro_f1)
    add('Offline model / Test mean average precision', test.mean_average_precision)
  }
  return rows
}

export function activityMarkdown(data: LessonPackage): string[] {
  const activity = data.activity_analysis
  if (!activity) return ['## Student behavior and participation analysis\n\nNo behavior analysis from the same source is available for this lesson; Unknown intervals are not assigned zero scores.']
  const engagement = engagementExport(activity.frames)
  const scoreByFrame = new Map(engagement.frames.map(row => [row.frame_id, row]))
  const time = (value: number) => `${Math.floor(value / 60).toString().padStart(2, '0')}:${Math.floor(value % 60).toString().padStart(2, '0')}`
  const span = (row: { start_sec: number; end_sec: number }) => `${time(row.start_sec)}–${time(row.end_sec)}`
  const names: Record<string, string> = { read: 'Reading', write: 'Writing', raise_hand: 'Hand raising', lookup: 'Model head-up cue', turn_head: 'Head-turn cue', stand: 'Standing cue', discuss: 'Discussion cue', head_up: 'Head raised', head_lowered: 'Head lowered', hand_near_head: 'Hand near head', standing: 'Standing posture cue', unknown: 'State unclear' }
  const percent = (value: number | null) => value === null ? 'Unknown' : `${(value * 100).toFixed(1)}%`
  const visible = (frame: ActivityFrame) => {
    const current = frame.participation
    if (!current) return ''
    const states = Object.entries(frame.student_state_counts || {}).map(([label, count]) => `${names[label] || label}: ${count}`).join(', ') || 'No visible people'
    const weights = Object.entries(current.weights).map(([label, weight]) => `${names[label] || label} ${weight}`).join(', ')
    const reference = frame.visual_reference
    const score = scoreByFrame.get(frame.id)
    const dense = reference && score ? `; Overall student engagement ${score.smoothed_index === null ? 'No score available' : `${score.smoothed_index.toFixed(1)}/100`}; Valid behaviors ${score.scored_people}/${score.total_people}(${percent(score.support_fraction)}); ${score.status === 'limited_support' ? 'Limited-support estimate' : 'Behavior estimate'}; Seven-class weights and dispersion adjustment; 0.3 × previous value + 0.7 × current value temporal smoothing` : ''
    return `; Current visible states ${states}; Recognizable states ${current.observed_people}/${current.total_people}(${percent(current.coverage)}); Detected participation actions ${current.active_people}/${current.total_people}(${percent(current.active_fraction)}); Behavioral engagement reference ${current.index === null ? 'Unknown' : `${current.index.toFixed(1)}/100`}(Rule average for current explicit actions only; ${weights}; Support ${current.scored_people}/${current.total_people}. ${percent(current.score_coverage)}; Not a probability of psychological engagement)${dense}`
  }
  const participant = (p: ActivityParticipant) => {
    const state = p.student_state
    const current = state ? `Current state: ${names[state.label] || state.name}; State source ${state.source}; State confidence ${state.confidence}; Evidence ${state.evidence.join(', ')}; Pose cues ${(p.posture_tags || []).map(tag => `${names[tag.label] || tag.name}(${tag.evidence.join(', ')}; Keypoints ${tag.keypoint_indices.join(',')})`).join(', ') || 'Unknown'}; ` : ''
    return `  - Visible person ${p.track_id}: ${current}Original behavior label ${p.label}; Confidence ${p.confidence}; Temporal rule score ${p.score === null ? 'Unknown' : p.score}; source ${p.source}`
  }
  return [
    `## Student behavior and participation analysis\n\nSource SHA-256: ${activity.source_sha256}\nSampling interval: ${activity.sampling_interval_sec} s\nVisible person observations: ${activity.summary.person_observations}; Observations with usable rule scores: ${activity.summary.known_observations}; Sampled frames: ${activity.summary.sampled_frames}; Scene: ${activity.summary.scene_count}\n\nThe behavioral engagement reference averages only current explicit reading, writing, and hand-raising cues using the published weights. Raised-head and lowered-head posture cues are displayed separately without participation scores. The detected participation action fraction is the current count of people showing explicit actions divided by the current detected person count. Unrecognized does not mean not participating. The temporal rule participation index additionally requires continuous behavior evidence and ranges from 0–100. Per-person temporal scores range from 0–1. They are not engagement probabilities or knowledge mastery results. Within-scene tracks represent visible people only. Teacher and student roles are not established; Unobserved or unknown values remain unknown. Rule-score coverage is the count of observations with usable temporal rule scores divided by the detected person count. Instantaneous behavior labels are displayed separately. Short clips or scene changes can leave a labeled person without a usable temporal rule score.`,
    ...(activity.frames.some(frame => frame.visual_reference) ? [`### Overall student engagement\n\nLesson average engagement: ${engagement.summary.average === null ? 'No score available' : `${engagement.summary.average.toFixed(1)}/100`}; Valid scores ${engagement.summary.scored_frames}/${engagement.summary.total_frames} frames. Weighted by valid frame duration.\n\nSeven-class weights: ${Object.entries(engagement.weights).map(([label,weight]) => `${names[label] || label} ${weight*100} points`).join(', ')}. First average valid behaviors, then multiply by exp(-0.6 × standard deviation). Consecutive observations in the same scene use 0.3 × previous value + 0.7 × current value smoothing. Low coverage is marked as a limited-support estimate, while existing scores are retained. Unknown states and geometric poses are not filled with zeros. No gaze score is added when gaze data is absent. Method version ${engagement.method}. It is an auxiliary behavior estimate for detected people. Original model records and legacy rule values are preserved.`] : []),
    `### Frame-by-frame behavior observations\n\n${activity.frames.map(frame => `- [${frame.id}] ${span(frame)} Scene ${frame.scene_id}; ${frame.status}; Detected ${frame.detected_people}. People with usable rule scores ${frame.known_people}; Rule-score coverage ${frame.coverage === null || !frame.detected_people ? 'Unknown(No usable detections)' : `${(frame.coverage * 100).toFixed(1)}%`}; Temporal rule participation index ${frame.participation_index === null ? 'Unknown' : frame.participation_index.toFixed(1)}${visible(frame)}; Instantaneous behavior labels ${Object.entries(frame.behaviors).map(([label, count]) => `${label}: ${count}`).join(', ') || 'Unknown'}${frame.image_url ? `; Source frame ${frame.image_url}` : ''}\n${frame.participants.map(participant).join('\n')}`).join('\n') || 'No sampled frames.'}`,
    `### Lesson context model predictions\n\n${activity.context_segments.map(row => `- [${row.id}] ${span(row)} ${row.text}; ${row.labels.map(label => `${label.name} (${label.label}) ${(label.probability * 100).toFixed(1)}%`).join(', ') || 'Probability not provided'}; Model ${row.model_id}; Method ${row.method}`).join('\n') || 'No context prediction records.'}`,
    `### Original behavior evidence(Recorded at generation time)\n\n${activity.evidence.map(row => `- [${row.id}] ${span(row)} ${row.text}`).join('\n') || 'No behavior evidence text.'}`,
    `### Behavior models and training records\n\n${readableModelFields(activity.models).map(row => `- ${row.label}: ${row.value}`).join('\n') || 'No displayable model records were provided.'}`,
    `### Activity analysis limitations\n\n${activity.limitations.map(line => `- ${line}`).join('\n') || 'No additional limitation records were provided.'}`,
  ]
}
