import type { ActivityFrame } from '../types/lessonReview'

// Reuses pipeline_legacy.yaml's seven behavior weights and CTES dispersion penalty.
// Recorded observations and legacy reference scores are never rewritten.
export const engagementWeights: Record<string, number> = Object.freeze({
  write: .85, read: .80, lookup: .85, turn_head: .30, raise_hand: .95, stand: .90, discuss: .75,
})
export const engagementPolicy = Object.freeze({
  id: 'overall_behavior_engagement_v1', variance_penalty: .6,
  previous_weight: .3, current_weight: .7, maximum_gap_sec: 1.2,
  support_warning_fraction: .6, support_warning_people: 3,
})
export interface ClassEngagement {
  frame_id: string; method: string; index: number | null; smoothed_index: number | null
  mean_action_score: number | null; dispersion: number | null; penalty_factor: number | null
  scored_people: number; total_people: number; support_fraction: number | null
  status: 'estimated' | 'limited_support' | 'no_people' | 'no_scored_behavior'
  counts: Record<string, number>; smoothing_applied: boolean; gaze_used: false
}
const round = (value: number) => Math.round(value*10000)/10000

export function overallEngagement(frames: ActivityFrame[]): ClassEngagement[] {
  const rows: ClassEngagement[] = []
  for (const [position, frame] of frames.entries()) {
    const values: number[] = [], counts: Record<string, number> = {}
    for (const person of frame.participants) {
      const state = person.student_state
      if (!state || !['pretrained_detector', 'trained_classifier'].includes(person.source) ||
        state.source !== person.source || state.label !== person.label ||
        !Object.prototype.hasOwnProperty.call(engagementWeights, state.label)) continue
      values.push(engagementWeights[state.label]!); counts[state.label] = (counts[state.label] || 0)+1
    }
    const mean = values.length ? values.reduce((sum, value) => sum+value, 0)/values.length : null
    const deviation = mean === null ? null : Math.sqrt(values.reduce((sum, value) => sum+(value-mean)**2, 0)/values.length)
    const penalty = deviation === null ? null : Math.exp(-engagementPolicy.variance_penalty*deviation)
    const index = mean === null || penalty === null ? null : round(mean*penalty*100)
    const support = frame.detected_people ? values.length/frame.detected_people : null
    const previous = rows[position-1], previousFrame = frames[position-1]
    const continuous = Boolean(previousFrame && previous?.smoothed_index !== null && previous?.smoothed_index !== undefined &&
      frame.scene_id === previousFrame.scene_id && frame.timestamp_sec > previousFrame.timestamp_sec &&
      frame.timestamp_sec-previousFrame.timestamp_sec <= engagementPolicy.maximum_gap_sec &&
      frame.start_sec <= previousFrame.end_sec+1e-6)
    rows.push({ frame_id: frame.id, method: engagementPolicy.id, index,
      smoothed_index: index === null ? null : continuous ? round(
        engagementPolicy.previous_weight*previous!.smoothed_index!+engagementPolicy.current_weight*index) : index,
      mean_action_score: mean === null ? null : round(mean*100), dispersion: deviation === null ? null : round(deviation),
      penalty_factor: penalty === null ? null : round(penalty), scored_people: values.length,
      total_people: frame.detected_people, support_fraction: support === null ? null : round(support),
      status: !frame.detected_people ? 'no_people' : !values.length ? 'no_scored_behavior' :
        values.length < engagementPolicy.support_warning_people || support! < engagementPolicy.support_warning_fraction ? 'limited_support' : 'estimated',
      counts, smoothing_applied: index !== null && continuous, gaze_used: false,
    })
  }
  return rows
}

export function engagementSummary(frames: ActivityFrame[], scores = overallEngagement(frames)) {
  let weighted = 0, duration = 0, sampledDuration = 0, scoredFrames = 0
  for (const [position, frame] of frames.entries()) {
    const seconds = Math.max(0, frame.end_sec-frame.start_sec), value = scores[position]?.smoothed_index
    sampledDuration += seconds
    if (value === undefined || value === null) continue
    weighted += value*seconds; duration += seconds; scoredFrames++
  }
  return { average: duration ? round(weighted/duration) : null, scored_frames: scoredFrames,
    total_frames: frames.length, scored_duration_sec: round(duration), sampled_duration_sec: round(sampledDuration),
    duration_coverage: sampledDuration ? round(duration/sampledDuration) : null }
}

export function engagementExport(frames: ActivityFrame[]) {
  const scores = overallEngagement(frames)
  return { method: engagementPolicy.id, weights: { ...engagementWeights }, policy: { ...engagementPolicy },
    summary: engagementSummary(frames, scores), frames: scores, derived_from: 'recorded_model_states',
    gaze_used: false, calibrated_engagement_probability: false }
}
