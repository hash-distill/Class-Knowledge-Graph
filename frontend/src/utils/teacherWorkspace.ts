import type { LessonPackage } from '../types/lessonReview'
import type { ReviewTask, ReviewTaskStatus, TeacherWorkspace } from '../types/teacherWorkspace'

type Row = Record<string, any>
const fail = (message: string): never => { throw new Error(message) }
const object = (value: unknown, label: string): Row => value && typeof value === 'object' && !Array.isArray(value) ? value as Row : fail(`${label}格式无效。`)
const array = (value: unknown, label: string): any[] => Array.isArray(value) ? value : fail(`${label}列表无效。`)
const text = (value: unknown, label: string, empty = false): string => typeof value === 'string' && (empty || !!value.trim()) ? value : fail(`${label}文字无效。`)
const ids = (value: unknown, label: string): string[] => array(value, label).map(id => text(id, label))
const integer = (value: unknown, label: string, minimum = 0): number => Number.isInteger(value) && (value as number) >= minimum ? value as number : fail(`${label}无效。`)
const taskStatuses: ReviewTaskStatus[] = ['queued', 'running', 'complete', 'failed', 'cancelled']
export function parseOptionalScore(value: unknown): number | null {
  if (value === '' || value === null || value === undefined) return null
  if (typeof value !== 'number' && typeof value !== 'string') fail('得分需为 0 至 1 的数字。')
  if (typeof value === 'string' && !value.trim()) return null
  const score = Number(value)
  if (!Number.isFinite(score) || score < 0 || score > 1) fail('得分需为 0 至 1 的数字。')
  return score
}
export function acceptWorkspaceSnapshot(current: Pick<TeacherWorkspace, 'lesson_id' | 'version'> | null, incoming: Pick<TeacherWorkspace, 'lesson_id' | 'version'>, lessonId: string): boolean {
  return incoming.lesson_id === lessonId && (!current || current.lesson_id !== lessonId || incoming.version >= current.version)
}
function snapshots(value: unknown, references: string[]) {
  for (const raw of array(value, '课堂证据快照')) {
    const row = object(raw, '课堂证据快照')
    text(row.id, '课堂证据快照'); text(row.text, '课堂证据快照', true)
    if (!references.includes(row.id) || typeof row.start_sec !== 'number' || !Number.isFinite(row.start_sec) || row.start_sec < 0 || typeof row.end_sec !== 'number' || !Number.isFinite(row.end_sec) || row.end_sec < row.start_sec) fail('课堂证据快照来源或时间无效。')
  }
}

export function parseTeacherWorkspace(value: unknown, lessonId: string): TeacherWorkspace {
  const data = object(value, '教师工作记录')
  if (data.lesson_id !== lessonId) fail('教师工作记录不属于当前课程。')
  text(data.source_revision, '当前来源'); text(data.stored_source_revision, '已存来源')
  if (typeof data.source_changed !== 'boolean') fail('教师工作记录来源状态无效。')
  integer(data.version, '教师工作记录版本'); text(data.goals, '教学目标', true)
  const allIds = new Set<string>()
  const base = (value: unknown, label: string) => {
    const row = object(value, label); text(row.id, label); text(row.created_at, '记录时间')
    if (allIds.has(row.id)) fail(`${label} ID 重复。`)
    allIds.add(row.id); return row
  }
  for (const raw of array(data.corrections, '教师修订')) {
    const row = base(raw, '教师修订')
    for (const field of ['transcript_id', 'original_text', 'replacement']) text(row[field], '教师修订')
    text(row.reason, '教师修订说明', true)
    if (row.human_verified !== true || row.revoked_at !== null) fail('当前教师修订核验状态无效。')
  }
  for (const raw of array(data.correction_history, '教师修订历史')) {
    const row = object(raw, '教师修订历史')
    for (const field of ['id', 'transcript_id', 'original_text', 'replacement', 'created_at']) text(row[field], '教师修订历史')
    text(row.reason, '教师修订历史说明', true)
    if (row.human_verified !== true || (row.revoked_at !== null && typeof row.revoked_at !== 'string')) fail('教师修订历史状态无效。')
  }
  for (const raw of array(data.actions, '改进行动')) {
    const row = base(raw, '改进行动'); text(row.title, '行动标题'); text(row.outcome, '实际结果', true); text(row.updated_at, '更新时间')
    if (!['planned', 'done'].includes(row.status)) fail('改进行动状态无效。')
    if (row.status === 'done' && !row.outcome.trim()) fail('完成行动必须记录实际结果。')
    snapshots(row.evidence_snapshot, ids(row.evidence_ids, '行动证据'))
  }
  const assessmentIds = new Set<string>()
  for (const raw of array(data.assessments, '教师练习')) {
    const row = base(raw, '教师练习'); text(row.question, '练习问题'); text(row.reference_answer, '参考答案')
    if (row.teacher_confirmed !== true) fail('练习必须由教师确认。')
    if (!ids(row.evidence_ids, '练习证据').length) fail('教师练习缺少实际证据。')
    snapshots(row.evidence_snapshot, row.evidence_ids); assessmentIds.add(row.id)
  }
  for (const raw of array(data.attempts, '实际作答')) {
    const row = base(raw, '实际作答'); text(row.answer, '实际作答'); text(row.feedback, '反馈', true)
    if (!assessmentIds.has(row.assessment_id)) fail('实际作答引用了未知练习。')
    if (row.source !== 'teacher_recorded') fail('实际作答必须保留教师记录来源。')
    if (row.score !== null && (typeof row.score !== 'number' || !Number.isFinite(row.score) || row.score < 0 || row.score > 1)) fail('实际作答分数无效（应为 0 至 1）。')
  }
  array(data.history, '历史来源记录')
  return data as TeacherWorkspace
}

export function parseReviewJobState(value: unknown): { status: ReviewTaskStatus; progress?: number; error?: string | { message?: string } } {
  const row = object(value, '处理任务')
  if (!taskStatuses.includes(row.status)) fail('处理任务状态无效。')
  if (row.progress !== undefined && (typeof row.progress !== 'number' || !Number.isFinite(row.progress) || row.progress < 0 || row.progress > 100)) fail('处理任务进度无效。')
  return row as ReturnType<typeof parseReviewJobState>
}
export function parseReviewTasks(value: unknown, lessonId: string): ReviewTask[] {
  const seen = new Set<string>()
  return array(object(value, '任务').tasks, '任务').map(raw => {
    const row = object(raw, '任务'); parseReviewJobState(row)
    for (const field of ['job_id', 'kind']) text(row[field], '任务')
    text(row.filename, '任务文件名', true)
    if (row.lesson_id !== lessonId) fail('处理任务不属于当前课程。')
    if (seen.has(row.job_id)) fail('处理任务 ID 重复。')
    seen.add(row.job_id)
    if (typeof row.cancel_requested !== 'boolean' || row.progress === undefined) fail('处理任务进度或取消状态无效。')
    integer(row.attempt, '任务尝试次数', 1)
    return row as ReviewTask
  })
}
export function taskIsPending(task: Pick<ReviewTask, 'status'>): boolean { return task.status === 'queued' || task.status === 'running' }
export function hasNewlyCompletedTasks(previous: ReviewTask[], current: ReviewTask[]): boolean {
  return current.some(task => task.status === 'complete' && !previous.some(old => old.job_id === task.job_id && old.status === 'complete'))
}

export function validateTeacherEvidence(data: Row, knownIds: Set<string>) {
  if (data.teacher_transcript === undefined) {
    if (data.teacher_evidence_blocks?.length || data.teacher_review) fail('教师证据缺少教师转写来源。')
    return
  }
  const transcript = new Map<string, Row>(data.transcript.map((row: Row) => [row.id, row]))
  const proofs = new Map<string, Row>((data.transcript_proofreading?.segments || []).map((row: Row) => [row.transcript_id, row]))
  const teachers = new Map<string, Row>()
  const sameValues = (a: unknown[], b: unknown[]) => JSON.stringify(a) === JSON.stringify(b)
  const metadata = (row: Row) => {
    if (row.uncertainties !== undefined) array(row.uncertainties, '教师证据不确定性')
    const refs = ids(row.proofreading_ids || [], '教师证据机器校对来源')
    if (refs.some(id => !knownIds.has(id))) fail('教师证据引用了未知机器校对来源。')
  }
  const sameTime = (a: Row, b: Row) => a.start_sec === b.start_sec && a.end_sec === b.end_sec
  const add = (row: Row) => { text(row.id, '教师证据'); if (knownIds.has(row.id)) fail('教师证据 ID 重复。'); knownIds.add(row.id) }
  for (const raw of array(data.teacher_transcript, '教师转写')) {
    const row = object(raw, '教师转写'), source = transcript.get(row.transcript_id)
    add(row); text(row.text, '教师转写', true)
    if (!source || teachers.has(row.transcript_id) || !sameTime(row, source)) fail('教师转写时间与原始转写不符。')
    const refs = ids(row.source_refs, '教师转写来源')
    if (!refs.includes(source!.id) || refs.some(id => !transcript.has(id))) fail('教师转写引用了非原始转写来源。')
    if (!['teacher_revision', 'machine_context'].includes(row.source) || typeof row.human_verified !== 'boolean' || row.human_verified !== (row.source === 'teacher_revision')) fail('教师转写核验状态无效。')
    if (row.human_verified ? typeof row.correction_id !== 'string' || !row.correction_id.trim() : row.correction_id !== null) fail('教师转写修订来源无效。')
    metadata(row)
    if (!row.human_verified) {
      const baseline = proofs.get(row.transcript_id) || source!
      if (row.text !== baseline.text || !sameValues(row.uncertainties || [], baseline.uncertainties || [])) fail('教师转写的机器上下文文字或不确定性与来源不符。')
    }
    teachers.set(row.transcript_id, row)
  }
  for (const raw of array(data.teacher_evidence_blocks || [], '教师检索证据')) {
    const row = object(raw, '教师检索证据'); add(row); text(row.text, '教师检索证据')
    if (row.kind !== 'transcript_teacher') fail('教师检索证据类型无效。')
    const refs = ids(row.source_refs, '教师检索证据来源'), members = refs.map(id => teachers.get(id))
    if (!refs.length || members.some(member => !member)) fail('教师检索证据来源无效。')
    const sources = members as Row[]
    if (row.start_sec !== Math.min(...sources.map(s => s.start_sec)) || row.end_sec !== Math.max(...sources.map(s => s.end_sec)) || row.text !== sources.map(s => s.text).join(' ')) fail('教师检索证据文字或时间与来源不符。')
    if (object(row.quality, '教师检索证据质量').human_verified !== sources.every(s => s.human_verified)) fail('教师检索证据不得将机器上下文标为已核验。')
    integer(row.workspace_version, '教师检索证据版本')
    metadata(row)
    const uncertainties = [...new Set(sources.flatMap(source => source.uncertainties || []))]
    const proofreadingIds = [...new Set(sources.flatMap(source => source.proofreading_ids || []))]
    if (!sameValues(row.uncertainties || [], uncertainties) || !sameValues(row.proofreading_ids || [], proofreadingIds)) fail('教师检索证据未保留机器上下文来源或不确定性。')
  }
  if (data.teacher_review !== undefined) {
    const review = object(data.teacher_review, '教师复盘摘要')
    integer(review.version, '教师复盘版本'); integer(review.corrected_segment_count, '教师修订数量'); text(review.source_revision, '教师复盘来源')
    if (review.corrected_segment_count !== [...teachers.values()].filter(row => row.human_verified).length || (data.teacher_evidence_blocks || []).some((row: Row) => row.workspace_version !== review.version)) fail('教师复盘摘要与核对记录版本不符。')
  }
}

export function teacherDisplayTranscripts(data: LessonPackage) {
  const teachers = new Map((data.teacher_transcript || []).map(row => [row.transcript_id, row]))
  const proofs = new Map((data.transcript_proofreading?.segments || []).map(row => [row.transcript_id, row]))
  return data.transcript.map(row => ({ ...row, text: teachers.get(row.id)?.text ?? proofs.get(row.id)?.text ?? row.text, teacher: teachers.get(row.id), proofreading: proofs.get(row.id), original_text: row.text }))
}
