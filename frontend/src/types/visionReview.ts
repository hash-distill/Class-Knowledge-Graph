import type { ActivityFrame } from './lessonReview'
export type VisualLabel = 'write' | 'read' | 'lookup' | 'turn_head' | 'raise_hand' | 'stand' | 'discuss' | 'unknown'
export interface VisualAnnotation {
  id: string; origin: 'model' | 'manual'; status: 'confirmed' | 'ignored'; label: VisualLabel
  role: 'student' | 'teacher' | 'unknown'; bbox: [number, number, number, number]; reason: string
}
export interface VisualDraft extends Omit<VisualAnnotation, 'status'> { status: VisualAnnotation['status'] | 'unreviewed' }
export interface VisualReviewDocument {
  lesson_id: string; frame_id: string; version: number; basis_fingerprint: string; source_changed: boolean
  updated_at: string | null; annotations: VisualAnnotation[]; width: number; height: number; image_sha256: string; frame: ActivityFrame
}
export interface VisualReviewHistory { version: number; basis_fingerprint: string; annotations: VisualAnnotation[]; updated_at: string }
