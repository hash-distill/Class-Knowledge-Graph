import type { ActivityFrame, ActivityParticipant } from '../types/lessonReview'

export const studentStateLabels: Record<string, string> = {
  read: '阅读', reading: '阅读', write: '书写', writing: '书写', raise_hand: '举手', raising_hand: '举手', hand_raising: '举手',
  head_up: '头部抬起', head_lowered: '头部降低', hand_near_head: '手靠近头部', standing: '站姿线索', unknown: '状态未知',
  lookup: '抬头线索', turn_head: '转头线索', stand: '站立线索', discuss: '讨论线索',
}
export const studentStateColors: Record<string, string> = {
  read: '#4b8ce7', reading: '#4b8ce7', write: '#78b5f1', writing: '#78b5f1', raise_hand: '#2454bc', raising_hand: '#2454bc', hand_raising: '#2454bc',
  head_up: '#6bb8e8', head_lowered: '#a0b4df', hand_near_head: '#4b72b6', standing: '#8aa8d4', unknown: '#aebbd0',
  lookup: '#3f82dd', turn_head: '#8aa0c8', stand: '#597bb6', discuss: '#2768c2',
}
export function studentStateName(label: string) { return studentStateLabels[label] || '其他可见线索' }
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
