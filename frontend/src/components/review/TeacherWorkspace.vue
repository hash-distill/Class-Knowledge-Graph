<script setup lang="ts">
import { headingText } from '@/utils/reviewPresentation'
import { computed, onActivated, onDeactivated, onUnmounted, reactive, ref, watch } from 'vue'
import { useLessonReviewStore } from '@/stores/lessonReview'
import { reviewRequest, reviewService, ReviewServiceError } from '@/api/reviewService'
import { formatReviewTime } from '@/utils/lessonReview'
import { acceptWorkspaceSnapshot, hasNewlyCompletedTasks, parseOptionalScore, parseTeacherWorkspace, taskIsPending } from '@/utils/teacherWorkspace'
import type { ReviewTask, TeacherWorkspace } from '@/types/teacherWorkspace'

type CorrectionDraft = { replacement: string; reason: string }
type AttemptDraft = { answer: string; feedback: string; score: string | number }
type Draft = { goals: string | null; transcriptId: string; corrections: Record<string, CorrectionDraft>; actionTitle: string; actionEvidence: string[]; outcomes: Record<string, string>; question: string; referenceAnswer: string; assessmentEvidence: string[]; attempts: Record<string, AttemptDraft> }
const blankDraft = (): Draft => ({ goals: null, transcriptId: '', corrections: {}, actionTitle: '', actionEvidence: [], outcomes: {}, question: '', referenceAnswer: '', assessmentEvidence: [], attempts: {} })
const store = useLessonReviewStore()
const teachingTab = ref('review')
const teachingTabs = [
  { id: 'review', label: 'Goals and verification', number: '01' },
  { id: 'actions', label: 'Improvement actions', number: '02' },
  { id: 'practice', label: 'Assessments and feedback', number: '03' },
  { id: 'records', label: 'History and tasks', number: '04' },
]
const lessonId = computed(() => store.data?.lesson.id || '')
const drafts = reactive<Record<string, Draft>>({})
const draft = computed(() => drafts[lessonId.value] ||= blankDraft())
const workspace = ref<TeacherWorkspace | null>(null)
const loading = ref(false), busy = ref(false), error = ref(''), notice = ref(''), conflict = ref(''), reconcileConfirmed = ref(false)
const tasks = ref<ReviewTask[]>([]), tasksLoading = ref(false), taskBusy = ref(''), taskError = ref('')
const archived = ref<{ archived_at: string; workspace: TeacherWorkspace }[]>([]), historyLoading = ref(false)
let active = false, epoch = 0, loadRequest = 0, taskRequest = 0, timer: ReturnType<typeof setTimeout> | undefined
const locked = computed(() => busy.value || loading.value || !workspace.value || workspace.value.source_changed || !!conflict.value)
const selectedTranscript = computed(() => store.data?.transcript.find(row => row.id === draft.value.transcriptId))
const correction = computed(() => {
  const id = draft.value.transcriptId
  if (!id) return null
  const saved = workspace.value?.corrections.find(row => row.transcript_id === id)
  return draft.value.corrections[id] ||= { replacement: saved?.replacement ?? store.displayTranscripts.find(row => row.id === id)?.text ?? '', reason: saved?.reason || '' }
})
const currentCorrection = computed(() => workspace.value?.corrections.find(row => row.transcript_id === draft.value.transcriptId))
// Every selectable ID is a timed current classroom source accepted by the service.
const evidenceOptions = computed(() => store.data ? [
  ...store.data.transcript, ...store.observedSlides, ...(store.data.evidence_blocks || []),
  ...(store.data.proofread_evidence_blocks || []), ...(store.data.teacher_transcript || []),
  ...(store.data.teacher_evidence_blocks || []), ...(store.data.revised_evidence_blocks || []),
  ...(store.data.courseware_evidence || []), ...(store.data.activity_analysis?.evidence || []),
].filter((row, index, rows) => rows.findIndex(other => other.id === row.id) === index) : [])
function evidenceLabel(id: string) { const row = evidenceOptions.value.find(row => row.id === id); return row ? `${id} · ${formatReviewTime(row.start_sec)} · ${row.text.slice(0, 70)}` : `${id} · Saved evidence snapshot` }
function outcome(id: string, saved: string) { return draft.value.outcomes[id] ?? saved }
function attempt(id: string): AttemptDraft { return draft.value.attempts[id] ||= { answer: '', feedback: '', score: '' } }
function message(reason: unknown) { return reason instanceof Error ? reason.message : 'Save incomplete. Please try again.' }
function validContext(id: string, version: number) { return epoch === version && lessonId.value === id }
function stopPolling() { if (timer) clearTimeout(timer); timer = undefined }
function scheduleTasks() { stopPolling(); if (active && tasks.value.some(taskIsPending)) timer = setTimeout(() => { void loadTasks() }, 2000) }
async function loadWorkspace() {
  const id = lessonId.value, version = epoch, request = ++loadRequest
  if (!id) return
  loading.value = true; error.value = ''
  try {
    const result = await reviewService.workspace(id)
    if (!validContext(id, version) || request !== loadRequest) return
    if (!acceptWorkspaceSnapshot(workspace.value, result, id)) return
    workspace.value = result; conflict.value = ''; reconcileConfirmed.value = false
    if (draft.value.goals === null) draft.value.goals = result.goals
    if (!draft.value.transcriptId) draft.value.transcriptId = store.data?.transcript[0]?.id || ''
  } catch (reason) { if (validContext(id, version) && request === loadRequest) error.value = message(reason) }
  finally { if (validContext(id, version) && request === loadRequest) loading.value = false }
}
async function mutate(path: string, body: Record<string, unknown>, after?: () => void, method = 'POST', refreshLesson = false) {
  const id = lessonId.value, version = epoch, current = workspace.value
  if (!id || !current || busy.value || (path !== '/reconcile' && locked.value)) return
  busy.value = true; error.value = ''; notice.value = ''
  try {
    const result = path === '/revert'
      ? await reviewService.revertCorrection(id, String(body.transcript_id), current.version)
      : await reviewService.saveWorkspace(id, path, { ...body, expected_version: current.version }, method)
    if (!validContext(id, version)) return
    loadRequest++; loading.value = false
    if (acceptWorkspaceSnapshot(workspace.value, result, id)) {
      workspace.value = result; conflict.value = ''; reconcileConfirmed.value = false
    }
    after?.()
    notice.value = 'Saved teacher records.'
    if (refreshLesson && store.data?.lesson.id === id) await store.refreshLesson(id)
  } catch (reason) {
    if (!validContext(id, version)) return
    error.value = message(reason)
    if (reason instanceof ReviewServiceError && reason.status === 409 && ['workspace_changed', 'workspace_source_changed'].includes(reason.code)) conflict.value = reason.code
  } finally { if (validContext(id, version)) busy.value = false }
}
async function saveCorrection() {
  if (!selectedTranscript.value || !correction.value?.replacement.trim()) return
  await mutate('/corrections', { transcript_id: selectedTranscript.value.id, ...correction.value }, undefined, 'POST', true)
}
async function revert() {
  const id = draft.value.transcriptId
  const baseline = store.proofreadByTranscript.get(id)?.text ?? store.data?.transcript.find(row => row.id === id)?.text ?? ''
  await mutate('/revert', { transcript_id: id }, () => { draft.value.corrections[id] = { replacement: baseline, reason: '' } }, 'DELETE', true)
}
async function addAction() { await mutate('/actions', { title: draft.value.actionTitle, evidence_ids: draft.value.actionEvidence }, () => { draft.value.actionTitle = ''; draft.value.actionEvidence = [] }) }
async function addAssessment() { await mutate('/assessments', { question: draft.value.question, reference_answer: draft.value.referenceAnswer, evidence_ids: draft.value.assessmentEvidence }, () => { draft.value.question = ''; draft.value.referenceAnswer = ''; draft.value.assessmentEvidence = [] }) }
async function addAttempt(id: string) {
  const value = attempt(id)
  let score: number | null
  try { score = parseOptionalScore(value.score) } catch (reason) { error.value = message(reason); return }
  await mutate(`/assessments/${encodeURIComponent(id)}/attempts`, { answer: value.answer, feedback: value.feedback, score }, () => { delete draft.value.attempts[id] })
}
async function reconcile() {
  if (!reconcileConfirmed.value) return
  await mutate('/reconcile', {}, () => {
    // Old-source saved records live in the audit archive. Unsaved drafts remain available.
    archived.value = []
  }, 'POST', true)
}
async function loadTasks() {
  const id = lessonId.value, version = epoch, request = ++taskRequest
  if (!id || !active) return
  stopPolling(); tasksLoading.value = true; taskError.value = ''
  try {
    const result = await reviewService.tasks(id)
    if (!validContext(id, version) || request !== taskRequest || !active) return
    const completed = hasNewlyCompletedTasks(tasks.value, result)
    tasks.value = result
    if (completed && store.data?.lesson.id === id) { const refreshed = await store.refreshLesson(id); if (refreshed && validContext(id, version)) await loadWorkspace() }
  } catch (reason) { if (validContext(id, version) && request === taskRequest) taskError.value = message(reason) }
  finally { if (validContext(id, version) && request === taskRequest) { tasksLoading.value = false; scheduleTasks() } }
}
async function changeTask(task: ReviewTask, action: 'cancel' | 'retry') {
  if (taskBusy.value) return
  const id = lessonId.value, version = epoch
  taskBusy.value = task.job_id; taskError.value = ''
  try {
    // A cancellation request is not completion. Re-read the service's actual state.
    await reviewService.task(task.job_id, action)
    if (validContext(id, version)) await loadTasks()
  } catch (reason) { if (validContext(id, version)) taskError.value = message(reason) }
  finally { if (validContext(id, version)) taskBusy.value = '' }
}
async function loadHistory() {
  const id = lessonId.value, version = epoch
  historyLoading.value = true
  try {
    const result = await reviewRequest(`/lessons/${encodeURIComponent(id)}/workspace/history`) as { history: { archived_at: string; snapshot: unknown }[] }
    if (!validContext(id, version)) return
    if (!Array.isArray(result.history)) throw new Error('Invalid historical source record format.')
    archived.value = result.history.map(row => ({ archived_at: row.archived_at, workspace: parseTeacherWorkspace(row.snapshot, id) }))
  } catch (reason) { if (validContext(id, version)) error.value = message(reason) }
  finally { if (validContext(id, version)) historyLoading.value = false }
}
const taskLabels = { queued: 'Queued', running: 'Processing', complete: 'Completed', failed: 'Failed', cancelled: 'Cancelled' }
function taskErrorText(task: ReviewTask) { return typeof task.error === 'string' ? task.error : task.error?.message || '' }
watch(lessonId, () => {
  epoch++; loadRequest++; taskRequest++; stopPolling(); workspace.value = null; tasks.value = []; archived.value = []
  loading.value = false; busy.value = false; tasksLoading.value = false; taskBusy.value = ''; historyLoading.value = false
  error.value = ''; notice.value = ''; conflict.value = ''; taskError.value = ''; reconcileConfirmed.value = false
  if (active) { void loadWorkspace(); void loadTasks() }
})
onActivated(() => { active = true; void loadWorkspace(); void loadTasks() })
onDeactivated(() => { active = false; stopPolling(); taskRequest++; tasksLoading.value = false })
onUnmounted(() => { active = false; epoch++; stopPolling() })
</script>

<template>
  <section class="teacher-workspace" data-testid="teacher-workspace" aria-label="Teaching improvement records">
    <div class="panel teacher-intro"><div><span class="eyebrow">Replay · Verify · Action · Feedback</span><h2>From lesson evidence to the next action</h2><p>Verify transcript. Record improvements. Review actual responses</p></div><button class="btn" data-testid="workspace-reload" :disabled="loading || busy" @click="loadWorkspace">Reload records</button></div>
    <nav class="teacher-tabs" aria-label="Teaching improvement sections"><button v-for="tab in teachingTabs" :key="tab.id" :data-teaching-tab="tab.id" :class="{ active: teachingTab === tab.id }" :aria-pressed="teachingTab === tab.id" @click="teachingTab = tab.id"><span>{{ tab.number }}</span>{{ tab.label }}</button></nav>
    <p v-if="loading" role="status" class="teacher-note">Loading teacher workspace records…</p>
    <div v-if="error || conflict" class="teacher-alert" role="alert" data-testid="workspace-error"><p>{{ error }}</p><p v-if="conflict">Unsaved input has been retained. Reload the records. Check the latest content before saving again.</p><button v-if="conflict" class="btn" :disabled="loading || busy" @click="loadWorkspace">Load the latest records and retain your input</button></div>
    <p v-if="notice" class="teacher-success" role="status">{{ notice }}</p>
    <div v-if="workspace?.source_changed || conflict === 'workspace_source_changed'" class="teacher-alert" data-testid="workspace-source-conflict"><h3>The lesson source has changed</h3><p>Existing teacher records refer to an earlier source. Confirming archives these records. Then start records for the current source. The original transcript and correction history are preserved; Unsaved input remains on this page. Check the evidence again before submitting.</p><label><input v-model="reconcileConfirmed" type="checkbox" data-testid="workspace-reconcile-confirm" :disabled="busy || loading" />I have checked the current lesson. Confirm archiving the old source records and starting a new version</label><button class="btn" data-testid="workspace-reconcile" :disabled="!reconcileConfirmed || busy || loading || !workspace" @click="reconcile">Confirm archive and start a new version</button></div>

    <template v-if="workspace">
      <p class="teacher-note">{{ workspace.source_changed ? 'The saved records below refer to an earlier lesson source. Check the source first.' : 'Teacher records for this lesson' }} · Record version {{ workspace.version }} · {{ workspace.corrections.length }} segments verified by the teacher</p>
      <section v-show="teachingTab === 'review'" class="panel teacher-section" data-teaching-section="review"><header><h2>Teaching goals</h2><span class="tag">Teacher record</span></header><form @submit.prevent="mutate('/goals', { text: draft.goals || '' })"><label for="teacher-goals">Lesson goals and review priorities</label><textarea id="teacher-goals" v-model="draft.goals" data-testid="teacher-goals" rows="3" maxlength="4000" :disabled="busy" placeholder="Enter the actual teaching goals for this lesson" /><p v-if="!workspace.goals" class="teacher-note">No teaching goals saved yet.</p><p v-else class="saved-text">Saved: {{ workspace.goals }}</p><button class="btn" type="submit" data-testid="save-teacher-goals" :disabled="locked">Save goals</button></form></section>

      <section v-show="teachingTab === 'review'" class="panel teacher-section" data-teaching-section="review"><header><h2>Verify lesson transcript</h2><span class="tag">Confirm after replay</span></header><p class="teacher-note">Choose original segment. Check against the shared video player. Verified text is saved separately. The original transcript and machine correction records are preserved.</p><label for="teacher-transcript">Original transcript segment</label><select id="teacher-transcript" v-model="draft.transcriptId" data-testid="teacher-transcript-select" :disabled="busy"><option value="">Choose segment</option><option v-for="row in store.data?.transcript" :key="row.id" :value="row.id">{{ row.id }} · {{ formatReviewTime(row.start_sec) }} · {{ row.text.slice(0, 70) }}</option></select><p v-if="!store.data?.transcript.length" class="teacher-note">No original transcript segments are available to verify for this lesson.</p>
        <form v-if="selectedTranscript && correction" @submit.prevent="saveCorrection"><div class="teacher-source"><button class="time-link" type="button" @click="store.seek(selectedTranscript.start_sec)">{{ formatReviewTime(selectedTranscript.start_sec) }}–{{ formatReviewTime(selectedTranscript.end_sec) }} Replay original audio ↗</button><p><small>Original machine transcript · Unverified</small>{{ selectedTranscript.text }}</p><p v-if="store.proofreadByTranscript.get(selectedTranscript.id)"><small>Machine correction suggestion · Unverified</small>{{ store.proofreadByTranscript.get(selectedTranscript.id)?.text }}</p></div><label for="teacher-replacement">Teacher-verified transcript</label><textarea id="teacher-replacement" v-model="correction.replacement" data-testid="teacher-correction-text" rows="3" maxlength="4000" :disabled="busy" required /><label for="teacher-correction-reason">Verification notes(Optional)</label><input id="teacher-correction-reason" v-model="correction.reason" maxlength="1000" :disabled="busy" /><p class="teacher-note">{{ currentCorrection ? 'This segment already has a teacher-verified transcript. Saving again preserves the previous correction history.' : 'This segment has not yet been approved by a teacher.' }}</p><div class="teacher-buttons"><button class="btn" type="submit" data-testid="save-teacher-correction" :disabled="locked || !correction.replacement.trim()">Confirm and save verified transcript</button><button v-if="currentCorrection" class="btn" type="button" data-testid="revert-teacher-correction" :disabled="locked" @click="revert">Revoke current verification</button></div></form>
        <details v-if="workspace.correction_history.length"><summary>View verification history({{ workspace.correction_history.length }} items)</summary><article v-for="row in workspace.correction_history" :key="row.id" class="teacher-record"><strong>{{ row.transcript_id }} · {{ row.revoked_at ? 'Revoked / Superseded by a later verification' : 'Currently active · Teacher verified' }}</strong><p>Original: {{ row.original_text }}</p><p>Verify: {{ row.replacement }}</p><p v-if="row.reason">Notes: {{ row.reason }}</p><small>{{ row.created_at }}</small></article></details>
      </section>

      <section v-show="teachingTab === 'actions'" class="panel teacher-section" data-teaching-section="actions"><header><h2>Improvement actions</h2><span class="tag">Plan → Actual outcome</span></header><form @submit.prevent="addAction"><label for="teacher-action-title">Next action</label><input id="teacher-action-title" v-model="draft.actionTitle" data-testid="teacher-action-title" maxlength="500" :disabled="busy" required placeholder="Enter a specific action you plan to carry out" /><label for="teacher-action-evidence">Supporting evidence(Optional. Select multiple)</label><select id="teacher-action-evidence" v-model="draft.actionEvidence" multiple size="4" :disabled="busy" data-testid="teacher-action-evidence"><option v-for="row in evidenceOptions" :key="row.id" :value="row.id">{{ evidenceLabel(row.id) }}</option></select><button class="btn" type="submit" data-testid="save-teacher-action" :disabled="locked || !draft.actionTitle.trim()">Add action plan</button></form><p v-if="!workspace.actions.length" class="teacher-note">No improvement actions yet. After completing the actual teaching activity, record its outcome.</p><article v-for="row in workspace.actions" :key="row.id" class="teacher-record" :data-testid="`teacher-action-${row.id}`"><h3>{{ headingText(row.title) }} <span class="tag">{{ row.status === 'done' ? 'Completed' : 'Planned' }}</span></h3><div class="evidence-links"><button v-for="id in row.evidence_ids" :key="id" class="time-link" @click="store.jumpToEvidence(id)">{{ id }} Review evidence ↗</button><span v-if="!row.evidence_ids.length" class="teacher-note">No linked evidence</span></div><details v-if="row.evidence_snapshot.length"><summary>Evidence snapshot at save time</summary><p v-for="source in row.evidence_snapshot" :key="source.id">{{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p></details><label :for="`outcome-${row.id}`">Actual action outcome</label><textarea :id="`outcome-${row.id}`" :value="outcome(row.id, row.outcome)" :data-testid="`teacher-action-outcome-${row.id}`" rows="2" maxlength="4000" :disabled="busy" @input="draft.outcomes[row.id] = ($event.target as HTMLTextAreaElement).value" /><p v-if="!row.outcome" class="teacher-note">No actual outcome recorded yet.</p><p v-else class="saved-text">Saved outcome: {{ row.outcome }}</p><div class="teacher-buttons"><button class="btn" :disabled="locked || !outcome(row.id, row.outcome).trim()" :data-testid="`complete-teacher-action-${row.id}`" @click="mutate(`/actions/${encodeURIComponent(row.id)}`, { status: 'done', outcome: outcome(row.id, row.outcome) }, undefined, 'PATCH')">{{ row.status === 'done' ? 'Update actual outcome' : 'Record outcome and complete' }}</button><button v-if="row.status === 'done'" class="btn" :disabled="locked" @click="mutate(`/actions/${encodeURIComponent(row.id)}`, { status: 'planned', outcome: outcome(row.id, row.outcome) }, undefined, 'PATCH')">Mark as planned again</button></div></article></section>

      <section v-show="teachingTab === 'practice'" class="panel teacher-section" data-teaching-section="practice"><header><h2>Follow-up assessments and actual responses</h2><span class="tag">Teacher confirmation</span></header><p class="teacher-note">Assessments must link to actual evidence from this lesson. The teacher must confirm the question and reference answer. Responses and scores are recorded by the teacher. They do not establish student identity or knowledge mastery.</p><form @submit.prevent="addAssessment"><label for="teacher-question">Assessment question</label><textarea id="teacher-question" v-model="draft.question" rows="2" maxlength="4000" data-testid="teacher-assessment-question" :disabled="busy" required /><label for="teacher-reference-answer">Teacher-confirmed reference answer</label><textarea id="teacher-reference-answer" v-model="draft.referenceAnswer" rows="2" maxlength="4000" data-testid="teacher-assessment-reference" :disabled="busy" required /><label for="teacher-assessment-evidence">Link lesson evidence(Required. Select multiple)</label><select id="teacher-assessment-evidence" v-model="draft.assessmentEvidence" multiple size="5" :disabled="busy" required data-testid="teacher-assessment-evidence"><option v-for="row in evidenceOptions" :key="row.id" :value="row.id">{{ evidenceLabel(row.id) }}</option></select><p v-if="!evidenceOptions.length" class="teacher-note">No lesson evidence is currently available to link. Cannot save assessment yet.</p><button class="btn" type="submit" data-testid="save-teacher-assessment" :disabled="locked || !draft.question.trim() || !draft.referenceAnswer.trim() || !draft.assessmentEvidence.length">Confirm and save assessment</button></form><p v-if="!workspace.assessments.length" class="teacher-note">No teacher-confirmed follow-up assessments yet.</p>
        <article v-for="row in workspace.assessments" :key="row.id" class="teacher-record" :data-testid="`teacher-assessment-${row.id}`"><h3>{{ headingText(row.question) }}</h3><p class="saved-text">Teacher-confirmed reference answer: {{ row.reference_answer }}</p><div class="evidence-links"><button v-for="id in row.evidence_ids" :key="id" class="time-link" @click="store.jumpToEvidence(id)">{{ id }} Review evidence ↗</button></div><details v-if="row.evidence_snapshot.length"><summary>Evidence snapshot at save time</summary><p v-for="source in row.evidence_snapshot" :key="source.id">{{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p></details><p v-if="!workspace.attempts.some(item => item.assessment_id === row.id)" class="teacher-note">No actual response recorded yet; Score unknown.</p><div v-for="answer in workspace.attempts.filter(item => item.assessment_id === row.id)" :key="answer.id" class="teacher-answer" data-testid="teacher-recorded-attempt"><p>Actual response: {{ answer.answer }}</p><p>Teacher feedback: {{ answer.feedback || 'Not entered' }}</p><p>{{ answer.score === null ? 'Unscored' : `Teacher-recorded score: ${answer.score}(0–1)` }}</p></div><form @submit.prevent="addAttempt(row.id)"><label :for="`answer-${row.id}`">Record actual response</label><textarea :id="`answer-${row.id}`" v-model="attempt(row.id).answer" :data-testid="`teacher-attempt-answer-${row.id}`" rows="2" maxlength="4000" :disabled="busy" required /><label :for="`feedback-${row.id}`">Teacher feedback(Optional)</label><textarea :id="`feedback-${row.id}`" v-model="attempt(row.id).feedback" rows="2" maxlength="4000" :disabled="busy" /><label :for="`score-${row.id}`">Teacher-recorded score(0–1. Leave blank for unscored)</label><input :id="`score-${row.id}`" v-model="attempt(row.id).score" :data-testid="`teacher-attempt-score-${row.id}`" type="number" min="0" max="1" step="any" :disabled="busy" /><button class="btn" type="submit" :data-testid="`save-teacher-attempt-${row.id}`" :disabled="locked || !attempt(row.id).answer.trim()">Save actual response</button></form></article>
      </section>
      <section v-show="teachingTab === 'records'" class="panel teacher-section" data-teaching-section="records"><header><h2>Historical source records</h2><button class="btn" :disabled="historyLoading" data-testid="workspace-history-load" @click="loadHistory">{{ historyLoading ? 'Loading…' : 'View archived records' }}</button></header><p class="teacher-note">{{ workspace.history.length ? `${workspace.history.length} previous source archives. ` : 'No previous source archives.' }}Archived records preserve the goals, revisions, actions, evidence snapshots, and actual responses at that time.</p><details v-for="(entry, index) in archived" :key="index" class="teacher-record"><summary>Archive {{ entry.archived_at }} · Record version {{ entry.workspace.version }}</summary><p>Goals: {{ entry.workspace.goals || 'Not entered' }}</p><p v-for="row in entry.workspace.correction_history" :key="row.id">{{ row.transcript_id }}: {{ row.original_text }} → {{ row.replacement }}({{ row.revoked_at ? 'Revoked' : 'Effective at that time' }})</p><div v-for="row in entry.workspace.actions" :key="row.id"><p>Action: {{ row.title }} · {{ row.status === 'done' ? 'Completed' : 'Planned' }} · {{ row.outcome || 'No actual outcome recorded' }}</p><p v-for="source in row.evidence_snapshot" :key="source.id">Evidence snapshot {{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p></div><div v-for="row in entry.workspace.assessments" :key="row.id"><p>Assessment: {{ row.question }}</p><p>Reference answer: {{ row.reference_answer }}</p><p v-for="source in row.evidence_snapshot" :key="source.id">Evidence snapshot {{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p><p v-for="answer in entry.workspace.attempts.filter(item => item.assessment_id === row.id)" :key="answer.id">Actual response: {{ answer.answer }} · Feedback: {{ answer.feedback || 'Not entered' }} · {{ answer.score === null ? 'Unscored' : `Score ${answer.score}` }}</p></div></details></section>
    </template>

    <section v-show="teachingTab === 'records'" class="panel teacher-section" data-teaching-section="records" data-testid="teacher-task-list"><header><h2>Courseware processing task</h2><button class="btn" data-testid="refresh-review-tasks" :disabled="tasksLoading || !!taskBusy" @click="loadTasks">Refresh tasks</button></header><p class="teacher-note">Shows the actual status and progress returned by the service. Cancellation requires service confirmation; The service continues processing after this page is closed.</p><p v-if="taskError" role="alert" class="teacher-alert">{{ taskError }}</p><p v-if="!tasks.length && !tasksLoading" class="teacher-note">No processing tasks for this lesson.</p><p v-if="tasksLoading" role="status" class="teacher-note">Loading task status…</p><article v-for="task in tasks" :key="task.job_id" class="teacher-record" :data-testid="`review-task-${task.job_id}`"><h3>{{ headingText(task.filename || 'Courseware processing') }} <span class="tag" :data-status="task.status">{{ taskLabels[task.status] }}</span></h3><p>{{ task.progress }}% · Attempt {{ task.attempt }}<span v-if="task.cancel_requested && taskIsPending(task)"> · Cancellation requested. Waiting for service confirmation</span></p><progress :value="task.progress" max="100" :aria-label="`${task.filename || 'Courseware task'} processing progress`" /><p v-if="taskErrorText(task)" class="teacher-note">{{ taskErrorText(task) }}</p><div class="teacher-buttons"><button v-if="taskIsPending(task)" class="btn" :disabled="!!taskBusy || task.cancel_requested" :data-testid="`cancel-review-task-${task.job_id}`" @click="changeTask(task, 'cancel')">{{ task.cancel_requested ? 'Waiting for cancellation confirmation' : 'Cancel task' }}</button><button v-if="task.status === 'failed' || task.status === 'cancelled'" class="btn" :disabled="!!taskBusy || tasks.some(taskIsPending)" :data-testid="`retry-review-task-${task.job_id}`" @click="changeTask(task, 'retry')">Retry task</button></div></article></section>
  </section>
</template>

<style scoped>
.teacher-workspace{display:grid;gap:18px;min-width:0}.teacher-intro{padding:22px;display:flex;align-items:start;justify-content:space-between;gap:16px}.teacher-intro h2,.teacher-section h2{font-size:16px;font-weight:500;color:var(--navy)}.teacher-intro p{font-size:11px;line-height:1.9;color:var(--muted);margin-top:9px}.teacher-section{padding:22px;min-width:0}.teacher-section header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:17px}.teacher-section form{display:grid;gap:10px}.teacher-section label{display:block;font-size:11px;color:#435c7e;line-height:1.8}.teacher-section input:not([type=checkbox]),.teacher-section textarea,.teacher-section select{width:100%;min-width:0;box-sizing:border-box;padding:10px 12px;border:1px solid #d0dbea;background:#f8fbfe;color:#324a6b;border-radius:6px;font:inherit;font-size:12px;line-height:1.8}.teacher-section textarea{resize:vertical}.teacher-section select[multiple]{min-height:104px;padding:6px}.teacher-section select[multiple] option{padding:6px;white-space:normal;overflow-wrap:anywhere}.teacher-section .btn{justify-self:start;font-size:11px}.teacher-section input:disabled,.teacher-section textarea:disabled,.teacher-section select:disabled{opacity:.65}.teacher-note{font-size:11px;line-height:1.9;color:#617898;margin:7px 0}.teacher-alert{padding:17px 19px;border:1px solid #b4cdf0;border-radius:8px;color:#2459a2;background:#e2ecfa;font-size:12px;line-height:1.9;overflow-wrap:anywhere}.teacher-alert h3{font-size:14px;margin-bottom:8px}.teacher-alert label{display:flex;align-items:start;gap:9px;margin:14px 0}.teacher-alert .btn{margin-top:9px}.teacher-success{font-size:12px;color:#385b8c;padding:12px 16px;background:#e0eaf9;border-radius:7px}.teacher-source{border-left:3px solid #a1b5d1;padding:12px 15px;margin:8px 0;background:#e9f1fb;font-size:12px;line-height:1.9}.teacher-source small{display:block;color:#647b9c;font-size:10px;margin-top:10px}.time-link{background:none;border:0;color:#4c6992;text-align:left;font-size:11px;line-height:1.8;padding:3px 0;cursor:pointer}.teacher-buttons,.evidence-links{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.evidence-links{margin:9px 0}.teacher-record{padding:18px 0 3px;border-top:1px solid var(--line);margin-top:18px;overflow-wrap:anywhere}.teacher-record h3,.teacher-record strong{font-size:13px;font-weight:500;line-height:1.9}.teacher-record h3 .tag{margin-left:8px}.teacher-record p,.saved-text{font-size:12px;line-height:1.9;color:#506685;white-space:pre-wrap;margin:8px 0}.teacher-record>small{font-size:10px;color:#677d9e}.teacher-record>label{margin-top:12px}.teacher-record>textarea{margin:7px 0 10px}.teacher-record>form{margin-top:15px}.teacher-record progress{accent-color:var(--blue);height:8px;width:100%;margin:8px 0 13px}.teacher-section details{margin-top:18px}.teacher-section summary{cursor:pointer;font-size:11px;color:#546d8f;line-height:1.9}.teacher-answer{background:#e5eefa;border-radius:6px;padding:10px 14px;margin-top:13px}.teacher-answer p{font-size:11px}.teacher-workspace button:focus-visible,.teacher-workspace input:focus-visible,.teacher-workspace textarea:focus-visible,.teacher-workspace select:focus-visible{outline:2px solid #5a7eb0;outline-offset:3px}@media(max-width:900px){.teacher-intro{flex-direction:column}.teacher-section{padding:18px}.teacher-section header{align-items:start}.teacher-section select[multiple]{font-size:11px}}@media(max-width:550px){.teacher-intro h2,.teacher-section h2{font-size:14px}.teacher-intro,.teacher-section{padding:16px}.teacher-section header{flex-wrap:wrap}.teacher-buttons .btn{font-size:10px}.teacher-alert{padding:14px}}
.teacher-workspace{padding:20px;gap:16px}.teacher-intro{padding:19px;background:linear-gradient(120deg,#eaf2ff,#f7faff);border-color:#d8e6fb;align-items:center}.teacher-intro .eyebrow{font-size:8px;color:#7293bf;display:block;margin-bottom:9px}.teacher-intro h2{font-size:17px;font-weight:600;color:#285292}.teacher-intro p{font-size:11px;color:#7690b4}.teacher-intro .btn{flex-shrink:0;background:#fff;font-size:10px}.teacher-tabs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:5px;padding:5px;background:#eff5ff;border:1px solid #dde8f9;border-radius:10px;position:sticky;top:0;z-index:4}.teacher-tabs button{display:flex;justify-content:center;align-items:center;gap:7px;padding:11px 4px;background:transparent;color:#6e89ad;font-size:10px;white-space:nowrap;border-radius:7px}.teacher-tabs button>span{font-size:8px;color:#9bb2d1}.teacher-tabs button:hover{color:#2864ca;background:#e4eeff}.teacher-tabs button.active{background:#fff;box-shadow:0 2px 8px #1f55a610;color:#245edb}.teacher-tabs button.active>span{color:#7ea6e8}.teacher-section{padding:20px;border-radius:12px;background:#fff;box-shadow:0 2px 7px #214a8104}.teacher-section h2{font-size:14px;font-weight:600}.teacher-section input:not([type=checkbox]),.teacher-section textarea,.teacher-section select{background:#f9fbff;border-color:#d9e5f5;color:#3c5c89;border-radius:8px}.teacher-section button[type=submit]{color:#fff;background:#2d6ce3;border-color:#2d6ce3;box-shadow:0 3px 10px #245edb14}.teacher-section button[type=submit]:hover:not(:disabled){background:#205bc9}.teacher-success{color:#356dc3;background:#edf4ff;border:1px solid #dbe8fc}.teacher-source{border-left-color:#82adef;background:#f2f7ff;border-radius:0 8px 8px 0}.teacher-record{margin-top:20px;padding:19px 0 3px}.teacher-answer{border:1px solid #d7e6fc;background:#f0f6ff;border-radius:8px}.teacher-note{color:#7489a7}.teacher-section details{padding:10px 12px;background:#f8faff;border:1px solid #e2ebf8;border-radius:7px}.teacher-tabs~.teacher-note{margin:0}.teacher-tabs button:focus-visible{outline:2px solid var(--blue);outline-offset:2px}@media(max-width:1280px){.teacher-tabs button{gap:4px;font-size:9px}.teacher-tabs button>span{display:none}}@media(max-width:600px){.teacher-workspace{padding:14px;gap:13px}.teacher-intro{padding:16px;align-items:start}.teacher-intro h2{font-size:15px}.teacher-tabs{grid-template-columns:repeat(2,minmax(0,1fr));position:static}.teacher-tabs button{font-size:10px}.teacher-section{padding:16px}}
</style>
