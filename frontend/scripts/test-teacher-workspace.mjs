import assert from 'node:assert/strict'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { join } from 'node:path'
import ts from 'typescript'

// These synthetic records exercise contracts only and are never production data.
const root = fileURLToPath(new URL('../', import.meta.url))
const cache = join(root, 'node_modules/.cache/teacher-workspace-tests')
mkdirSync(cache, { recursive: true })
const file = join(root, 'src/utils/teacherWorkspace.ts')
const output = ts.transpileModule(existsSync(file) ? readFileSync(file, 'utf8') : 'export {}', { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText
writeFileSync(join(cache, 'teacherWorkspace.mjs'), output)
const api = await import(pathToFileURL(join(cache, 'teacherWorkspace.mjs')).href)
assert.equal(typeof api.parseTeacherWorkspace, 'function', 'A workspace must be validated before editable teacher records are shown')
assert.equal(typeof api.parseOptionalScore, 'function', 'Numeric browser inputs must preserve zero and missing scores')
assert.equal(api.parseOptionalScore(0), 0)
assert.equal(api.parseOptionalScore('0.5'), .5)
assert.equal(api.parseOptionalScore(''), null)
assert.throws(() => api.parseOptionalScore(true), /Score/i)
assert.throws(() => api.parseOptionalScore(1.1), /Score/i)
assert.equal(typeof api.acceptWorkspaceSnapshot, 'function')
assert.equal(api.acceptWorkspaceSnapshot({lesson_id:'a',version:2},{lesson_id:'a',version:1},'a'), false, 'A delayed GET must not roll back a successful mutation')
assert.equal(api.acceptWorkspaceSnapshot({lesson_id:'a',version:1},{lesson_id:'a',version:2},'a'), true)
const workspace = { lesson_id: 'fixture', source_revision: 'source-a', stored_source_revision: 'source-a', source_changed: false, version: 2, goals: '', corrections: [], correction_history: [], actions: [], assessments: [], attempts: [], history: [] }
assert.equal(api.parseTeacherWorkspace(workspace, 'fixture').version, 2)
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, lesson_id: 'other' }, 'fixture'), /Lesson/i)
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, version: -1 }, 'fixture'), /Version/i)
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, source_changed: 'false' }, 'fixture'), /source/i)
const correctionRecord = { id: 'C1', transcript_id: 'T1', original_text: '原文', replacement: '核对稿', reason: '', created_at: 'now', revoked_at: null, human_verified: true }
assert.equal(api.parseTeacherWorkspace({ ...workspace, corrections: [correctionRecord], correction_history: [correctionRecord] }, 'fixture').corrections[0].reason, '', 'An optional correction note may be empty')
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, actions: [{ id: 'A1', title: 'Assessment', status: 'done', outcome: '', evidence_ids: [], evidence_snapshot: [], created_at: 'now', updated_at: 'now' }] }, 'fixture'), /Actual outcome/i)
const assessment = { id: 'Q1', question: '实际问题', reference_answer: '教师参考答案', evidence_ids: ['T1'], evidence_snapshot: [], teacher_confirmed: true, created_at: 'now' }
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, assessments: [{ ...assessment, teacher_confirmed: false }] }, 'fixture'), /confirmed by the teacher/i)
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, assessments: [{ ...assessment, evidence_ids: [] }] }, 'fixture'), /Evidence/i)
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, assessments: [{ ...assessment, evidence_snapshot: [{ id: 'T1', text: '证据文字', start_sec: 10, end_sec: 2 }] }] }, 'fixture'), /snapshot/i)
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, assessments: [assessment], attempts: [{ id: 'R1', assessment_id: 'unknown', answer: 'Actual response', feedback: '', score: null, source: 'teacher_recorded', created_at: 'now' }] }, 'fixture'), /Assessment/i)
const validAttempt = { id: 'R1', assessment_id: 'Q1', answer: '实际作答', feedback: '', score: 0, source: 'teacher_recorded', created_at: 'now' }
assert.equal(api.parseTeacherWorkspace({ ...workspace, assessments: [assessment], attempts: [validAttempt] }, 'fixture').attempts[0].score, 0, 'A recorded zero is not missing')
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, assessments: [assessment], attempts: [{ ...validAttempt, score: NaN }] }, 'fixture'), /score/i)
assert.throws(() => api.parseTeacherWorkspace({ ...workspace, assessments: [assessment], attempts: [{ ...validAttempt, score: 1.1 }] }, 'fixture'), /score/i)
const task = { job_id: 'J1', lesson_id: 'fixture', kind: 'courseware', filename: '实际文件.pdf', status: 'queued', progress: 0, cancel_requested: false, attempt: 1 }
for (const status of ['queued', 'running', 'complete', 'failed', 'cancelled']) assert.equal(api.parseReviewTasks({ tasks: [{ ...task, status }] }, 'fixture')[0].status, status)
assert.throws(() => api.parseReviewTasks({ tasks: [{ ...task, lesson_id: 'other' }] }, 'fixture'), /Lesson/i)
assert.throws(() => api.parseReviewTasks({ tasks: [{ ...task, progress: 101 }] }, 'fixture'), /progress/i)
assert.equal(api.taskIsPending({ ...task, status: 'running' }), true)
assert.equal(api.taskIsPending({ ...task, status: 'cancelled' }), false)
assert.equal(typeof api.hasNewlyCompletedTasks, 'function', 'A task finished while the upload panel was closed must refresh the classroom')
assert.equal(api.hasNewlyCompletedTasks([], [{ ...task, status: 'complete' }]), true)
assert.equal(api.hasNewlyCompletedTasks([{ ...task, status: 'complete' }], [{ ...task, status: 'complete' }]), false)
assert.equal(api.hasNewlyCompletedTasks([task], [{ ...task, status: 'cancelled' }]), false)
assert.equal(api.parseReviewJobState({ status: 'queued', progress: 0 }).status, 'queued')
assert.equal(api.parseReviewJobState({ status: 'cancelled' }).status, 'cancelled')
assert.throws(() => api.parseReviewJobState({ status: 'imaginary', progress: 55 }), /status/i)

const raw = [{ id: 'T1', text: '原始机器文字', start_sec: 0, end_sec: 1, source: 'machine' }, { id: 'T2', text: '第二段', start_sec: 1, end_sec: 2, source: 'machine' }]
const teacherRows = [{ id: 'HT1', transcript_id: 'T1', text: '教师核准文字', start_sec: 0, end_sec: 1, source_refs: ['T1'], source: 'teacher_revision', human_verified: true, correction_id: 'C1' }, { id: 'HT2', transcript_id: 'T2', text: '第二段', start_sec: 1, end_sec: 2, source_refs: ['T2'], source: 'machine_context', human_verified: false, correction_id: null }]
const block = { id: 'HB1', kind: 'transcript_teacher', text: '教师核准文字 第二段', start_sec: 0, end_sec: 2, source_refs: ['T1', 'T2'], quality: { human_verified: false }, workspace_version: 2 }
const lesson = { transcript: raw, teacher_transcript: teacherRows, teacher_evidence_blocks: [block], lesson: { duration_sec: 2 }, transcript_proofreading: { segments: [] } }
api.validateTeacherEvidence(lesson, new Set(['T1', 'T2']))
const shown = api.teacherDisplayTranscripts(lesson)
assert.equal(shown[0].id, 'T1', 'Display preserves the raw segment ID and player clock')
assert.equal(shown[0].text, '教师核准文字')
assert.equal(shown[0].teacher.human_verified, true)
assert.equal(shown[1].teacher.human_verified, false, 'Machine context must never inherit teacher verification')
assert.equal(raw[0].text, '原始机器文字', 'Teacher display never mutates original ASR')
for (const change of [
  { teacher_transcript: [{ ...teacherRows[0], start_sec: .5 }] },
  { teacher_transcript: [{ ...teacherRows[0], source_refs: ['OTHER'] }] },
  { teacher_transcript: [{ ...teacherRows[1], human_verified: true }] },
  { teacher_evidence_blocks: [{ ...block, quality: { human_verified: true } }] },
  { teacher_evidence_blocks: [{ ...block, text: '模型自行改写' }] },
  { teacher_transcript: [teacherRows[0], { ...teacherRows[1], text: '机器上下文被随意改写' }] },
]) assert.throws(() => api.validateTeacherEvidence({ ...lesson, ...change }, new Set(['T1', 'T2'])), /Teacher/i)
assert.throws(() => api.validateTeacherEvidence(lesson, new Set(['T1', 'T2', 'HB1'])), /Duplicate/i)
const uncertainLesson = structuredClone(lesson)
uncertainLesson.transcript_proofreading.segments = [{ id: 'P2', transcript_id: 'T2', text: '第二段', uncertainties: ['原声不清楚'] }]
uncertainLesson.teacher_transcript[1].proofreading_ids = ['P2']
uncertainLesson.teacher_transcript[1].uncertainties = ['原声不清楚']
assert.throws(() => api.validateTeacherEvidence(uncertainLesson, new Set(['T1', 'T2', 'P2'])), /Teacher/i, 'Teacher blocks must retain machine-context uncertainty')
uncertainLesson.teacher_evidence_blocks[0].uncertainties = ['原声不清楚']
uncertainLesson.teacher_evidence_blocks[0].proofreading_ids = ['P2']
api.validateTeacherEvidence(uncertainLesson, new Set(['T1', 'T2', 'P2']))
console.log('PASS: teacher workspace versions, real outcomes, assessment evidence, null/zero scores, task contracts, teacher source/time/verification boundaries.')
