import type { ActivityFrame } from '../types/lessonReview'
import type { VisualAnnotation, VisualDraft, VisualLabel } from '../types/visionReview'

export const visualLabels: Record<VisualLabel, string> = {
  write: '书写', read: '阅读', lookup: '抬头', turn_head: '转头', raise_hand: '举手', stand: '站立', discuss: '讨论', unknown: '状态未知',
}
export function reviewRows(frame: ActivityFrame, annotations: VisualAnnotation[]): VisualDraft[] {
  const changes = new Map(annotations.map(row => [row.id, row]))
  return [...frame.participants.map(person => {
    const label = person.student_state?.label || person.label
    return changes.get(person.track_id) || {
      id: person.track_id, origin: 'model' as const, status: 'unreviewed' as const,
      label: (label in visualLabels ? label : 'unknown') as VisualLabel,
      role: 'unknown' as const, bbox: [...person.bbox] as VisualDraft['bbox'], reason: '',
    }
  }), ...annotations.filter(row => row.origin === 'manual')].map(row => ({ ...row, bbox: [...row.bbox] }))
}
export function savedAnnotations(rows: VisualDraft[]): VisualAnnotation[] {
  return rows.filter((row): row is VisualAnnotation => row.status !== 'unreviewed').map(row => ({ ...row, bbox: [...row.bbox] }))
}
export function drawBox(start: number[], end: number[], width: number, height: number): VisualDraft['bbox'] | null {
  const x1 = Math.round(Math.max(0, Math.min(width, Math.min(start[0]!, end[0]!))))
  const y1 = Math.round(Math.max(0, Math.min(height, Math.min(start[1]!, end[1]!))))
  const x2 = Math.round(Math.max(0, Math.min(width, Math.max(start[0]!, end[0]!))))
  const y2 = Math.round(Math.max(0, Math.min(height, Math.max(start[1]!, end[1]!))))
  return x2-x1 >= 2 && y2-y1 >= 2 ? [x1, y1, x2, y2] : null
}
