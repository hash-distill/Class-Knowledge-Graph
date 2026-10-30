import type { LessonPackage } from '../types/lessonReview'
import type { ReviewTask, ReviewTaskStatus, TeacherWorkspace } from '../types/teacherWorkspace'

type Row = Record<string, any>
const fail = (message: string): never => { throw new Error(message) }
const object = (value: unknown, label: string): Row => value && typeof value === 'object' && !Array.isArray(value) ? value as Row : fail(`${label} has an invalid format.`)
const array = (value: unknown, label: string): any[] => Array.isArray(value) ? value : fail(`${label} is not a valid list.`)
const text = (value: unknown, label: string, empty = false): string => typeof value === 'string' && (empty || !!value.trim()) ? value : fail(`${label} has invalid text.`)
const ids = (value: unknown, label: string): string[] => array(value, label).map(id => text(id, label))
const integer = (value: unknown, label: string, minimum = 0): number => Number.isInteger(value) && (value as number) >= minimum ? value as number : fail(`${label} is invalid.`)
const taskStatuses: ReviewTaskStatus[] = ['queued', 'running', 'complete', 'failed', 'cancelled']
export function parseOptionalScore(value: unknown): number | null {
  if (value === '' || value === null || value === undefined) return null
  if (typeof value !== 'number' && typeof value !== 'string') fail('Score must be a number from 0 to 1  .')
  if (typeof value === 'string' && !value.trim()) return null
  const score = Number(value)
  if (!Number.isFinite(score) || score < 0 || score > 1) fail('Score must be a number from 0 to 1  .')
  return score
}
export function acceptWorkspaceSnapshot(current: Pick<TeacherWorkspace, 'lesson_id' | 'version'> | null, incoming: Pick<TeacherWorkspace, 'lesson_id' | 'version'>, lessonId: string): boolean {
  return incoming.lesson_id === lessonId && (!current || current.lesson_id !== lessonId || incoming.version >= current.version)
}
function snapshots(value: unknown, references: string[]) {
  for (const raw of array(value, 'Lesson evidence snapshot')) {
    const row = object(raw, 'Lesson evidence snapshot')
    text(row.id, 'Lesson evidence snapshot'); text(row.text, 'Lesson evidence snapshot', true)
    if (!references.includes(row.id) || typeof row.start_sec !== 'number' || !Number.isFinite(row.start_sec) || row.start_sec < 0 || typeof row.end_sec !== 'number' || !Number.isFinite(row.end_sec) || row.end_sec < row.start_sec) fail('Invalid lesson evidence snapshot source or times.')
  }
}

export function parseTeacherWorkspace(value: unknown, lessonId: string): TeacherWorkspace {
  const data = object(value, 'Teacher workspace record')
  if (data.lesson_id !== lessonId) fail('Teacher workspace record does not belong to the current lesson.')
  text(data.source_revision, 'Current source'); text(data.stored_source_revision, 'Stored source')
  if (typeof data.source_changed !== 'boolean') fail('Invalid teacher workspace source status.')
  integer(data.version, 'Teacher workspace version'); text(data.goals, 'Teaching goals', true)
  const allIds = new Set<string>()
  const base = (value: unknown, label: string) => {
    const row = object(value, label); text(row.id, label); text(row.created_at, 'Record timestamp')
    if (allIds.has(row.id)) fail(`${label} ID Duplicate.`)
    allIds.add(row.id); return row
  }
  for (const raw of array(data.corrections, 'Teacher correction')) {
    const row = base(raw, 'Teacher correction')
    for (const field of ['transcript_id', 'original_text', 'replacement']) text(row[field], 'Teacher correction')
    text(row.reason, 'Teacher correction notes', true)
    if (row.human_verified !== true || row.revoked_at !== null) fail('Invalid current teacher correction verification status.')
  }
  for (const raw of array(data.correction_history, 'Teacher correction history')) {
    const row = object(raw, 'Teacher correction history')
    for (const field of ['id', 'transcript_id', 'original_text', 'replacement', 'created_at']) text(row[field], 'Teacher correction history')
    text(row.reason, 'Teacher correction history notes', true)
    if (row.human_verified !== true || (row.revoked_at !== null && typeof row.revoked_at !== 'string')) fail('Invalid teacher correction history status.')
  }
  for (const raw of array(data.actions, 'Improvement actions')) {
    const row = base(raw, 'Improvement actions'); text(row.title, 'Action title'); text(row.outcome, 'Actual outcome', true); text(row.updated_at, 'Updated at')
    if (!['planned', 'done'].includes(row.status)) fail('Invalid improvement action status.')
    if (row.status === 'done' && !row.outcome.trim()) fail('Completed actions must record their actual outcomes.')
    snapshots(row.evidence_snapshot, ids(row.evidence_ids, 'Action evidence'))
  }
  const assessmentIds = new Set<string>()
  for (const raw of array(data.assessments, 'Teacher assessment')) {
    const row = base(raw, 'Teacher assessment'); text(row.question, 'Assessment question'); text(row.reference_answer, 'Reference answer')
    if (row.teacher_confirmed !== true) fail('Assessments must be confirmed by the teacher.')
    if (!ids(row.evidence_ids, 'Assessment evidence').length) fail('Teacher assessment is missing actual evidence.')
    snapshots(row.evidence_snapshot, row.evidence_ids); assessmentIds.add(row.id)
  }
  for (const raw of array(data.attempts, 'Actual response')) {
    const row = base(raw, 'Actual response'); text(row.answer, 'Actual response'); text(row.feedback, 'Feedback', true)
    if (!assessmentIds.has(row.assessment_id)) fail('Actual response references an unknown assessment.')
    if (row.source !== 'teacher_recorded') fail('Actual responses must retain the teacher-recorded source.')
    if (row.score !== null && (typeof row.score !== 'number' || !Number.isFinite(row.score) || row.score < 0 || row.score > 1)) fail('Invalid actual response score(Expected 0 to 1).')
  }
  array(data.history, 'Historical source records')
  return data as TeacherWorkspace
}

export function parseReviewJobState(value: unknown): { status: ReviewTaskStatus; progress?: number; error?: string | { message?: string } } {
  const row = object(value, 'Processing task')
  if (!taskStatuses.includes(row.status)) fail('Invalid processing task status.')
  if (row.progress !== undefined && (typeof row.progress !== 'number' || !Number.isFinite(row.progress) || row.progress < 0 || row.progress > 100)) fail('Invalid processing task progress.')
  return row as ReturnType<typeof parseReviewJobState>
}
export function parseReviewTasks(value: unknown, lessonId: string): ReviewTask[] {
  const seen = new Set<string>()
  return array(object(value, 'Task').tasks, 'Task').map(raw => {
    const row = object(raw, 'Task'); parseReviewJobState(row)
    for (const field of ['job_id', 'kind']) text(row[field], 'Task')
    text(row.filename, 'Task filename', true)
    if (row.lesson_id !== lessonId) fail('Processing task does not belong to the current lesson.')
    if (seen.has(row.job_id)) fail('Processing task ID Duplicate.')
    seen.add(row.job_id)
    if (typeof row.cancel_requested !== 'boolean' || row.progress === undefined) fail('Invalid processing task progress or cancellation state.')
    integer(row.attempt, 'Task attempt number', 1)
    return row as ReviewTask
  })
}
export function taskIsPending(task: Pick<ReviewTask, 'status'>): boolean { return task.status === 'queued' || task.status === 'running' }
export function hasNewlyCompletedTasks(previous: ReviewTask[], current: ReviewTask[]): boolean {
  return current.some(task => task.status === 'complete' && !previous.some(old => old.job_id === task.job_id && old.status === 'complete'))
}

export function validateTeacherEvidence(data: Row, knownIds: Set<string>) {
  if (data.teacher_transcript === undefined) {
    if (data.teacher_evidence_blocks?.length || data.teacher_review) fail('Teacher evidence is missing its teacher transcript source.')
    return
  }
  const transcript = new Map<string, Row>(data.transcript.map((row: Row) => [row.id, row]))
  const proofs = new Map<string, Row>((data.transcript_proofreading?.segments || []).map((row: Row) => [row.transcript_id, row]))
  const teachers = new Map<string, Row>()
  const sameValues = (a: unknown[], b: unknown[]) => JSON.stringify(a) === JSON.stringify(b)
  const metadata = (row: Row) => {
    if (row.uncertainties !== undefined) array(row.uncertainties, 'Teacher evidence uncertainties')
    const refs = ids(row.proofreading_ids || [], 'Teacher evidence machine correction sources')
    if (refs.some(id => !knownIds.has(id))) fail('Teacher evidence references unknown machine correction sources.')
  }
  const sameTime = (a: Row, b: Row) => a.start_sec === b.start_sec && a.end_sec === b.end_sec
  const add = (row: Row) => { text(row.id, 'Teacher evidence'); if (knownIds.has(row.id)) fail('Teacher evidence ID Duplicate.'); knownIds.add(row.id) }
  for (const raw of array(data.teacher_transcript, 'Teacher transcript')) {
    const row = object(raw, 'Teacher transcript'), source = transcript.get(row.transcript_id)
    add(row); text(row.text, 'Teacher transcript', true)
    if (!source || teachers.has(row.transcript_id) || !sameTime(row, source)) fail('Teacher transcript times do not match the original transcript.')
    const refs = ids(row.source_refs, 'Teacher transcript source')
    if (!refs.includes(source!.id) || refs.some(id => !transcript.has(id))) fail('Teacher transcript references a source outside the original transcript.')
    if (!['teacher_revision', 'machine_context'].includes(row.source) || typeof row.human_verified !== 'boolean' || row.human_verified !== (row.source === 'teacher_revision')) fail('Invalid teacher transcript verification status.')
    if (row.human_verified ? typeof row.correction_id !== 'string' || !row.correction_id.trim() : row.correction_id !== null) fail('Invalid teacher transcript correction source.')
    metadata(row)
    if (!row.human_verified) {
      const baseline = proofs.get(row.transcript_id) || source!
      if (row.text !== baseline.text || !sameValues(row.uncertainties || [], baseline.uncertainties || [])) fail('Teacher transcript machine context text or uncertainties do not match the source.')
    }
    teachers.set(row.transcript_id, row)
  }
  for (const raw of array(data.teacher_evidence_blocks || [], 'Teacher retrieval evidence')) {
    const row = object(raw, 'Teacher retrieval evidence'); add(row); text(row.text, 'Teacher retrieval evidence')
    if (row.kind !== 'transcript_teacher') fail('Invalid teacher retrieval evidence kind.')
    const refs = ids(row.source_refs, 'Teacher retrieval evidence source'), members = refs.map(id => teachers.get(id))
    if (!refs.length || members.some(member => !member)) fail('Invalid teacher retrieval evidence source.')
    const sources = members as Row[]
    if (row.start_sec !== Math.min(...sources.map(s => s.start_sec)) || row.end_sec !== Math.max(...sources.map(s => s.end_sec)) || row.text !== sources.map(s => s.text).join(' ')) fail('Teacher retrieval evidence text or times do not match the source.')
    if (object(row.quality, 'Teacher retrieval evidence quality').human_verified !== sources.every(s => s.human_verified)) fail('Teacher retrieval evidence must not mark machine context as verified.')
    integer(row.workspace_version, 'Teacher retrieval evidence version')
    metadata(row)
    const uncertainties = [...new Set(sources.flatMap(source => source.uncertainties || []))]
    const proofreadingIds = [...new Set(sources.flatMap(source => source.proofreading_ids || []))]
    if (!sameValues(row.uncertainties || [], uncertainties) || !sameValues(row.proofreading_ids || [], proofreadingIds)) fail('Teacher retrieval evidence does not preserve machine context sources or uncertainties.')
  }
  if (data.teacher_review !== undefined) {
    const review = object(data.teacher_review, 'Teacher review summary')
    integer(review.version, 'Teacher review version'); integer(review.corrected_segment_count, 'Teacher correction count'); text(review.source_revision, 'Teacher review source')
    if (review.corrected_segment_count !== [...teachers.values()].filter(row => row.human_verified).length || (data.teacher_evidence_blocks || []).some((row: Row) => row.workspace_version !== review.version)) fail('Teacher review summary does not match the correction record version.')
  }
}

export function teacherDisplayTranscripts(data: LessonPackage) {
  const teachers = new Map((data.teacher_transcript || []).map(row => [row.transcript_id, row]))
  const proofs = new Map((data.transcript_proofreading?.segments || []).map(row => [row.transcript_id, row]))
  return data.transcript.map(row => ({ ...row, text: teachers.get(row.id)?.text ?? proofs.get(row.id)?.text ?? row.text, teacher: teachers.get(row.id), proofreading: proofs.get(row.id), original_text: row.text }))
}
