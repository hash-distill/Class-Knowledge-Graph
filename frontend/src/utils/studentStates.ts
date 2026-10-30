import type { ActivityFrame, ActivityParticipant } from '../types/lessonReview'

export const studentStateLabels: Record<string, string> = {
  read: 'Reading', reading: 'Reading', write: 'Writing', writing: 'Writing', raise_hand: 'Hand raising', raising_hand: 'Hand raising', hand_raising: 'Hand raising',
  head_up: 'Head raised', head_lowered: 'Head lowered', hand_near_head: 'Hand near head', standing: 'Standing posture cue', unknown: 'Unknown state',
  lookup: 'Head-up cue', turn_head: 'Head-turn cue', stand: 'Standing cue', discuss: 'Discussion cue',
}
export const studentStateColors: Record<string, string> = {
  read: '#4b8ce7', reading: '#4b8ce7', write: '#78b5f1', writing: '#78b5f1', raise_hand: '#2454bc', raising_hand: '#2454bc', hand_raising: '#2454bc',
  head_up: '#6bb8e8', head_lowered: '#a0b4df', hand_near_head: '#4b72b6', standing: '#8aa8d4', unknown: '#aebbd0',
  lookup: '#3f82dd', turn_head: '#8aa0c8', stand: '#597bb6', discuss: '#2768c2',
}
export function studentStateName(label: string) { return studentStateLabels[label] || 'Other visible cues' }
export function studentStateColor(label: string) { return studentStateColors[label] || studentStateColors.unknown }
export function participantState(participant: ActivityParticipant) { return participant.student_state?.label || participant.label || 'unknown' }
export function frameStateCounts(frame: ActivityFrame): [string, number][] {
  const counts = frame.student_state_counts || frame.behaviors
  return Object.entries(counts).filter(([, count]) => count > 0).sort((a, b) => (a[0] === 'unknown' ? 1 : b[0] === 'unknown' ? -1 : b[1] - a[1]))
}
export function visibleParticipation(frame: ActivityFrame) {
  if (frame.participation) return frame.participation
  // Older packages retain their recorded rule score; never infer posture from missing evidence.
  return { index: frame.participation_index, observed_people: frame.known_people, total_people: frame.detected_people, unknown_people: frame.detected_people - frame.known_people, coverage: frame.coverage || 0, active_people: null, active_fraction: null, weights: {}, semantics: 'legacy_rule' }
}
