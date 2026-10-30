import type { TimedEvidence } from './lessonReview'

export interface TeacherCorrection { id: string; transcript_id: string; original_text: string; replacement: string; reason: string; created_at: string; revoked_at: string | null; human_verified: true }
export interface TeacherAction { id: string; title: string; status: 'planned' | 'done'; outcome: string; evidence_ids: string[]; evidence_snapshot: TimedEvidence[]; created_at: string; updated_at: string }
export interface TeacherAssessment { id: string; question: string; reference_answer: string; evidence_ids: string[]; evidence_snapshot: TimedEvidence[]; teacher_confirmed: true; created_at: string }
export interface TeacherAttempt { id: string; assessment_id: string; answer: string; feedback: string; score: number | null; created_at: string; source: 'teacher_recorded' }
export interface TeacherWorkspace {
  lesson_id: string; source_revision: string; stored_source_revision: string; source_changed: boolean; version: number; goals: string
  corrections: TeacherCorrection[]; correction_history: TeacherCorrection[]; actions: TeacherAction[]; assessments: TeacherAssessment[]; attempts: TeacherAttempt[]; history: unknown[]
}
export interface TeacherTranscript extends TimedEvidence {
  transcript_id: string; source_refs: string[]; source: 'teacher_revision' | 'machine_context'; human_verified: boolean; correction_id: string | null
  uncertainties?: unknown[]; proofreading_ids?: string[]; uncertain_spans?: unknown[]
}
export interface TeacherEvidenceBlock extends TimedEvidence { kind: 'transcript_teacher'; source_refs: string[]; quality: { human_verified: boolean }; workspace_version: number; uncertainties?: unknown[]; proofreading_ids?: string[] }
export type ReviewTaskStatus = 'queued' | 'running' | 'complete' | 'failed' | 'cancelled'
export interface ReviewTask {
  job_id: string; lesson_id: string; kind: string; filename: string; status: ReviewTaskStatus; progress: number; cancel_requested: boolean; attempt: number
  error?: string | { message?: string }; retry_of?: string | null
}
