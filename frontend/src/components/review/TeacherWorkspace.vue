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
  { id: 'review', label: '目标与核对', number: '01' },
  { id: 'actions', label: '改进行动', number: '02' },
  { id: 'practice', label: '练习与反馈', number: '03' },
  { id: 'records', label: '历史与任务', number: '04' },
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
function evidenceLabel(id: string) { const row = evidenceOptions.value.find(row => row.id === id); return row ? `${id} · ${formatReviewTime(row.start_sec)} · ${row.text.slice(0, 70)}` : `${id} · 已存证据快照` }
function outcome(id: string, saved: string) { return draft.value.outcomes[id] ?? saved }
function attempt(id: string): AttemptDraft { return draft.value.attempts[id] ||= { answer: '', feedback: '', score: '' } }
function message(reason: unknown) { return reason instanceof Error ? reason.message : '保存未完成，请重试。' }
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
    notice.value = '已保存教师记录。'
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
    if (!Array.isArray(result.history)) throw new Error('历史来源记录格式无效。')
    archived.value = result.history.map(row => ({ archived_at: row.archived_at, workspace: parseTeacherWorkspace(row.snapshot, id) }))
  } catch (reason) { if (validContext(id, version)) error.value = message(reason) }
  finally { if (validContext(id, version)) historyLoading.value = false }
}
const taskLabels = { queued: '排队中', running: '处理中', complete: '已完成', failed: '失败', cancelled: '已取消' }
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
  <section class="teacher-workspace" data-testid="teacher-workspace" aria-label="教学改进工作记录">
    <div class="panel teacher-intro"><div><span class="eyebrow">回看 · 核对 · 行动 · 反馈</span><h2>从课堂证据到下一步行动</h2><p>核对转写，记录改进，回顾真实作答</p></div><button class="btn" data-testid="workspace-reload" :disabled="loading || busy" @click="loadWorkspace">重新加载记录</button></div>
    <nav class="teacher-tabs" aria-label="教学改进分区"><button v-for="tab in teachingTabs" :key="tab.id" :data-teaching-tab="tab.id" :class="{ active: teachingTab === tab.id }" :aria-pressed="teachingTab === tab.id" @click="teachingTab = tab.id"><span>{{ tab.number }}</span>{{ tab.label }}</button></nav>
    <p v-if="loading" role="status" class="teacher-note">正在读取教师工作记录…</p>
    <div v-if="error || conflict" class="teacher-alert" role="alert" data-testid="workspace-error"><p>{{ error }}</p><p v-if="conflict">未保存输入已保留。请重新加载记录，核对最新内容后再次保存。</p><button v-if="conflict" class="btn" :disabled="loading || busy" @click="loadWorkspace">加载最新记录并保留输入</button></div>
    <p v-if="notice" class="teacher-success" role="status">{{ notice }}</p>
    <div v-if="workspace?.source_changed || conflict === 'workspace_source_changed'" class="teacher-alert" data-testid="workspace-source-conflict"><h3>课堂来源已更新</h3><p>已有教师记录属于旧来源。确认后会将这些记录归档，再开始当前来源的记录。原始转写和历史核对记录会保留；未保存输入仍留在此页，提交前请重新核对证据。</p><label><input v-model="reconcileConfirmed" type="checkbox" data-testid="workspace-reconcile-confirm" :disabled="busy || loading" />我已核对当前课堂，确认归档旧来源记录并开始新版本</label><button class="btn" data-testid="workspace-reconcile" :disabled="!reconcileConfirmed || busy || loading || !workspace" @click="reconcile">确认归档并开始新版本</button></div>

    <template v-if="workspace">
      <p class="teacher-note">{{ workspace.source_changed ? '下方已保存记录来自旧课堂来源，请先核对来源。' : '本课教师记录' }} · 记录版本 {{ workspace.version }} · {{ workspace.corrections.length }} 段教师已核对</p>
      <section v-show="teachingTab === 'review'" class="panel teacher-section" data-teaching-section="review"><header><h2>教学目标</h2><span class="tag">教师记录</span></header><form @submit.prevent="mutate('/goals', { text: draft.goals || '' })"><label for="teacher-goals">本课目标与复盘重点</label><textarea id="teacher-goals" v-model="draft.goals" data-testid="teacher-goals" rows="3" maxlength="4000" :disabled="busy" placeholder="填写本课的实际教学目标" /><p v-if="!workspace.goals" class="teacher-note">尚未保存教学目标。</p><p v-else class="saved-text">已保存：{{ workspace.goals }}</p><button class="btn" type="submit" data-testid="save-teacher-goals" :disabled="locked">保存目标</button></form></section>

      <section v-show="teachingTab === 'review'" class="panel teacher-section" data-teaching-section="review"><header><h2>核对课堂转写</h2><span class="tag">回听后确认</span></header><p class="teacher-note">选择原始片段，对照共用播放器回听。核对稿单独保存，原始转写与机器校对记录保留。</p><label for="teacher-transcript">原始转写片段</label><select id="teacher-transcript" v-model="draft.transcriptId" data-testid="teacher-transcript-select" :disabled="busy"><option value="">选择片段</option><option v-for="row in store.data?.transcript" :key="row.id" :value="row.id">{{ row.id }} · {{ formatReviewTime(row.start_sec) }} · {{ row.text.slice(0, 70) }}</option></select><p v-if="!store.data?.transcript.length" class="teacher-note">本课没有可核对的原始转写片段。</p>
        <form v-if="selectedTranscript && correction" @submit.prevent="saveCorrection"><div class="teacher-source"><button class="time-link" type="button" @click="store.seek(selectedTranscript.start_sec)">{{ formatReviewTime(selectedTranscript.start_sec) }}–{{ formatReviewTime(selectedTranscript.end_sec) }} 回听原声 ↗</button><p><small>原始机器转写 · 未核验</small>{{ selectedTranscript.text }}</p><p v-if="store.proofreadByTranscript.get(selectedTranscript.id)"><small>机器校对建议 · 未核验</small>{{ store.proofreadByTranscript.get(selectedTranscript.id)?.text }}</p></div><label for="teacher-replacement">教师核对稿</label><textarea id="teacher-replacement" v-model="correction.replacement" data-testid="teacher-correction-text" rows="3" maxlength="4000" :disabled="busy" required /><label for="teacher-correction-reason">核对说明（可选）</label><input id="teacher-correction-reason" v-model="correction.reason" maxlength="1000" :disabled="busy" /><p class="teacher-note">{{ currentCorrection ? '当前片段已有教师核对稿。再次保存将保留前次修订历史。' : '当前片段尚未经教师核准。' }}</p><div class="teacher-buttons"><button class="btn" type="submit" data-testid="save-teacher-correction" :disabled="locked || !correction.replacement.trim()">确认并保存核对稿</button><button v-if="currentCorrection" class="btn" type="button" data-testid="revert-teacher-correction" :disabled="locked" @click="revert">撤销当前核对稿</button></div></form>
        <details v-if="workspace.correction_history.length"><summary>查看核对历史（{{ workspace.correction_history.length }} 条）</summary><article v-for="row in workspace.correction_history" :key="row.id" class="teacher-record"><strong>{{ row.transcript_id }} · {{ row.revoked_at ? '已撤销 / 已被后续核对替代' : '当前生效 · 教师已核对' }}</strong><p>原始：{{ row.original_text }}</p><p>核对：{{ row.replacement }}</p><p v-if="row.reason">说明：{{ row.reason }}</p><small>{{ row.created_at }}</small></article></details>
      </section>

      <section v-show="teachingTab === 'actions'" class="panel teacher-section" data-teaching-section="actions"><header><h2>改进行动</h2><span class="tag">计划 → 实际结果</span></header><form @submit.prevent="addAction"><label for="teacher-action-title">下一步行动</label><input id="teacher-action-title" v-model="draft.actionTitle" data-testid="teacher-action-title" maxlength="500" :disabled="busy" required placeholder="填写准备实施的具体行动" /><label for="teacher-action-evidence">支撑证据（可选，可多选）</label><select id="teacher-action-evidence" v-model="draft.actionEvidence" multiple size="4" :disabled="busy" data-testid="teacher-action-evidence"><option v-for="row in evidenceOptions" :key="row.id" :value="row.id">{{ evidenceLabel(row.id) }}</option></select><button class="btn" type="submit" data-testid="save-teacher-action" :disabled="locked || !draft.actionTitle.trim()">添加行动计划</button></form><p v-if="!workspace.actions.length" class="teacher-note">尚无改进行动。完成实际教学后，再记录执行结果。</p><article v-for="row in workspace.actions" :key="row.id" class="teacher-record" :data-testid="`teacher-action-${row.id}`"><h3>{{ headingText(row.title) }} <span class="tag">{{ row.status === 'done' ? '已完成' : '待实施' }}</span></h3><div class="evidence-links"><button v-for="id in row.evidence_ids" :key="id" class="time-link" @click="store.jumpToEvidence(id)">{{ id }} 回看证据 ↗</button><span v-if="!row.evidence_ids.length" class="teacher-note">未关联证据</span></div><details v-if="row.evidence_snapshot.length"><summary>保存时的证据快照</summary><p v-for="source in row.evidence_snapshot" :key="source.id">{{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p></details><label :for="`outcome-${row.id}`">实际执行结果</label><textarea :id="`outcome-${row.id}`" :value="outcome(row.id, row.outcome)" :data-testid="`teacher-action-outcome-${row.id}`" rows="2" maxlength="4000" :disabled="busy" @input="draft.outcomes[row.id] = ($event.target as HTMLTextAreaElement).value" /><p v-if="!row.outcome" class="teacher-note">尚未记录实际结果。</p><p v-else class="saved-text">已保存结果：{{ row.outcome }}</p><div class="teacher-buttons"><button class="btn" :disabled="locked || !outcome(row.id, row.outcome).trim()" :data-testid="`complete-teacher-action-${row.id}`" @click="mutate(`/actions/${encodeURIComponent(row.id)}`, { status: 'done', outcome: outcome(row.id, row.outcome) }, undefined, 'PATCH')">{{ row.status === 'done' ? '更新实际结果' : '记录结果并完成' }}</button><button v-if="row.status === 'done'" class="btn" :disabled="locked" @click="mutate(`/actions/${encodeURIComponent(row.id)}`, { status: 'planned', outcome: outcome(row.id, row.outcome) }, undefined, 'PATCH')">重新列为待实施</button></div></article></section>

      <section v-show="teachingTab === 'practice'" class="panel teacher-section" data-teaching-section="practice"><header><h2>课后练习与实际作答</h2><span class="tag">教师确认</span></header><p class="teacher-note">练习需关联本课实际证据，并由教师确认问题和参考答案。作答与得分由教师记录，不据此推断学生身份或知识掌握。</p><form @submit.prevent="addAssessment"><label for="teacher-question">练习问题</label><textarea id="teacher-question" v-model="draft.question" rows="2" maxlength="4000" data-testid="teacher-assessment-question" :disabled="busy" required /><label for="teacher-reference-answer">教师确认的参考答案</label><textarea id="teacher-reference-answer" v-model="draft.referenceAnswer" rows="2" maxlength="4000" data-testid="teacher-assessment-reference" :disabled="busy" required /><label for="teacher-assessment-evidence">关联本课证据（必选，可多选）</label><select id="teacher-assessment-evidence" v-model="draft.assessmentEvidence" multiple size="5" :disabled="busy" required data-testid="teacher-assessment-evidence"><option v-for="row in evidenceOptions" :key="row.id" :value="row.id">{{ evidenceLabel(row.id) }}</option></select><p v-if="!evidenceOptions.length" class="teacher-note">当前没有可关联的课堂证据，暂不能保存练习。</p><button class="btn" type="submit" data-testid="save-teacher-assessment" :disabled="locked || !draft.question.trim() || !draft.referenceAnswer.trim() || !draft.assessmentEvidence.length">教师确认并保存练习</button></form><p v-if="!workspace.assessments.length" class="teacher-note">尚无教师确认的课后练习。</p>
        <article v-for="row in workspace.assessments" :key="row.id" class="teacher-record" :data-testid="`teacher-assessment-${row.id}`"><h3>{{ headingText(row.question) }}</h3><p class="saved-text">教师确认的参考答案：{{ row.reference_answer }}</p><div class="evidence-links"><button v-for="id in row.evidence_ids" :key="id" class="time-link" @click="store.jumpToEvidence(id)">{{ id }} 回看证据 ↗</button></div><details v-if="row.evidence_snapshot.length"><summary>保存时的证据快照</summary><p v-for="source in row.evidence_snapshot" :key="source.id">{{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p></details><p v-if="!workspace.attempts.some(item => item.assessment_id === row.id)" class="teacher-note">尚未记录实际作答；得分未知。</p><div v-for="answer in workspace.attempts.filter(item => item.assessment_id === row.id)" :key="answer.id" class="teacher-answer" data-testid="teacher-recorded-attempt"><p>实际作答：{{ answer.answer }}</p><p>教师反馈：{{ answer.feedback || '未填写' }}</p><p>{{ answer.score === null ? '未评分' : `教师记录得分：${answer.score}（0–1）` }}</p></div><form @submit.prevent="addAttempt(row.id)"><label :for="`answer-${row.id}`">记录实际作答</label><textarea :id="`answer-${row.id}`" v-model="attempt(row.id).answer" :data-testid="`teacher-attempt-answer-${row.id}`" rows="2" maxlength="4000" :disabled="busy" required /><label :for="`feedback-${row.id}`">教师反馈（可选）</label><textarea :id="`feedback-${row.id}`" v-model="attempt(row.id).feedback" rows="2" maxlength="4000" :disabled="busy" /><label :for="`score-${row.id}`">教师记录得分（0–1，留空表示未评分）</label><input :id="`score-${row.id}`" v-model="attempt(row.id).score" :data-testid="`teacher-attempt-score-${row.id}`" type="number" min="0" max="1" step="any" :disabled="busy" /><button class="btn" type="submit" :data-testid="`save-teacher-attempt-${row.id}`" :disabled="locked || !attempt(row.id).answer.trim()">保存实际作答</button></form></article>
      </section>
      <section v-show="teachingTab === 'records'" class="panel teacher-section" data-teaching-section="records"><header><h2>历史来源记录</h2><button class="btn" :disabled="historyLoading" data-testid="workspace-history-load" @click="loadHistory">{{ historyLoading ? '正在读取…' : '查看归档记录' }}</button></header><p class="teacher-note">{{ workspace.history.length ? `已有 ${workspace.history.length} 次旧来源归档。` : '尚无旧来源归档。' }}归档记录保留当时的目标、修订、行动、证据快照和实际作答。</p><details v-for="(entry, index) in archived" :key="index" class="teacher-record"><summary>归档 {{ entry.archived_at }} · 记录版本 {{ entry.workspace.version }}</summary><p>目标：{{ entry.workspace.goals || '未填写' }}</p><p v-for="row in entry.workspace.correction_history" :key="row.id">{{ row.transcript_id }}：{{ row.original_text }} → {{ row.replacement }}（{{ row.revoked_at ? '已撤销' : '当时生效' }}）</p><div v-for="row in entry.workspace.actions" :key="row.id"><p>行动：{{ row.title }} · {{ row.status === 'done' ? '已完成' : '待实施' }} · {{ row.outcome || '未记录实际结果' }}</p><p v-for="source in row.evidence_snapshot" :key="source.id">证据快照 {{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p></div><div v-for="row in entry.workspace.assessments" :key="row.id"><p>练习：{{ row.question }}</p><p>参考答案：{{ row.reference_answer }}</p><p v-for="source in row.evidence_snapshot" :key="source.id">证据快照 {{ source.id }} · {{ formatReviewTime(source.start_sec) }}–{{ formatReviewTime(source.end_sec) }} · {{ source.text }}</p><p v-for="answer in entry.workspace.attempts.filter(item => item.assessment_id === row.id)" :key="answer.id">实际作答：{{ answer.answer }} · 反馈：{{ answer.feedback || '未填写' }} · {{ answer.score === null ? '未评分' : `得分 ${answer.score}` }}</p></div></details></section>
    </template>

    <section v-show="teachingTab === 'records'" class="panel teacher-section" data-teaching-section="records" data-testid="teacher-task-list"><header><h2>课件处理任务</h2><button class="btn" data-testid="refresh-review-tasks" :disabled="tasksLoading || !!taskBusy" @click="loadTasks">刷新任务</button></header><p class="teacher-note">展示服务返回的实际状态与进度。取消需等待服务确认；关闭此页后任务仍由服务处理。</p><p v-if="taskError" role="alert" class="teacher-alert">{{ taskError }}</p><p v-if="!tasks.length && !tasksLoading" class="teacher-note">本课尚无处理任务。</p><p v-if="tasksLoading" role="status" class="teacher-note">正在读取任务状态…</p><article v-for="task in tasks" :key="task.job_id" class="teacher-record" :data-testid="`review-task-${task.job_id}`"><h3>{{ headingText(task.filename || '课件处理') }} <span class="tag" :data-status="task.status">{{ taskLabels[task.status] }}</span></h3><p>{{ task.progress }}% · 第 {{ task.attempt }} 次尝试<span v-if="task.cancel_requested && taskIsPending(task)"> · 已请求取消，等待服务确认</span></p><progress :value="task.progress" max="100" :aria-label="`${task.filename || '课件任务'}实际处理进度`" /><p v-if="taskErrorText(task)" class="teacher-note">{{ taskErrorText(task) }}</p><div class="teacher-buttons"><button v-if="taskIsPending(task)" class="btn" :disabled="!!taskBusy || task.cancel_requested" :data-testid="`cancel-review-task-${task.job_id}`" @click="changeTask(task, 'cancel')">{{ task.cancel_requested ? '等待取消确认' : '取消任务' }}</button><button v-if="task.status === 'failed' || task.status === 'cancelled'" class="btn" :disabled="!!taskBusy || tasks.some(taskIsPending)" :data-testid="`retry-review-task-${task.job_id}`" @click="changeTask(task, 'retry')">重试任务</button></div></article></section>
  </section>
</template>

<style scoped>
.teacher-workspace{display:grid;gap:18px;min-width:0}.teacher-intro{padding:22px;display:flex;align-items:start;justify-content:space-between;gap:16px}.teacher-intro h2,.teacher-section h2{font-size:16px;font-weight:500;color:var(--navy)}.teacher-intro p{font-size:11px;line-height:1.9;color:var(--muted);margin-top:9px}.teacher-section{padding:22px;min-width:0}.teacher-section header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:17px}.teacher-section form{display:grid;gap:10px}.teacher-section label{display:block;font-size:11px;color:#435c7e;line-height:1.8}.teacher-section input:not([type=checkbox]),.teacher-section textarea,.teacher-section select{width:100%;min-width:0;box-sizing:border-box;padding:10px 12px;border:1px solid #d0dbea;background:#f8fbfe;color:#324a6b;border-radius:6px;font:inherit;font-size:12px;line-height:1.8}.teacher-section textarea{resize:vertical}.teacher-section select[multiple]{min-height:104px;padding:6px}.teacher-section select[multiple] option{padding:6px;white-space:normal;overflow-wrap:anywhere}.teacher-section .btn{justify-self:start;font-size:11px}.teacher-section input:disabled,.teacher-section textarea:disabled,.teacher-section select:disabled{opacity:.65}.teacher-note{font-size:11px;line-height:1.9;color:#617898;margin:7px 0}.teacher-alert{padding:17px 19px;border:1px solid #b4cdf0;border-radius:8px;color:#2459a2;background:#e2ecfa;font-size:12px;line-height:1.9;overflow-wrap:anywhere}.teacher-alert h3{font-size:14px;margin-bottom:8px}.teacher-alert label{display:flex;align-items:start;gap:9px;margin:14px 0}.teacher-alert .btn{margin-top:9px}.teacher-success{font-size:12px;color:#385b8c;padding:12px 16px;background:#e0eaf9;border-radius:7px}.teacher-source{border-left:3px solid #a1b5d1;padding:12px 15px;margin:8px 0;background:#e9f1fb;font-size:12px;line-height:1.9}.teacher-source small{display:block;color:#647b9c;font-size:10px;margin-top:10px}.time-link{background:none;border:0;color:#4c6992;text-align:left;font-size:11px;line-height:1.8;padding:3px 0;cursor:pointer}.teacher-buttons,.evidence-links{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.evidence-links{margin:9px 0}.teacher-record{padding:18px 0 3px;border-top:1px solid var(--line);margin-top:18px;overflow-wrap:anywhere}.teacher-record h3,.teacher-record strong{font-size:13px;font-weight:500;line-height:1.9}.teacher-record h3 .tag{margin-left:8px}.teacher-record p,.saved-text{font-size:12px;line-height:1.9;color:#506685;white-space:pre-wrap;margin:8px 0}.teacher-record>small{font-size:10px;color:#677d9e}.teacher-record>label{margin-top:12px}.teacher-record>textarea{margin:7px 0 10px}.teacher-record>form{margin-top:15px}.teacher-record progress{accent-color:var(--blue);height:8px;width:100%;margin:8px 0 13px}.teacher-section details{margin-top:18px}.teacher-section summary{cursor:pointer;font-size:11px;color:#546d8f;line-height:1.9}.teacher-answer{background:#e5eefa;border-radius:6px;padding:10px 14px;margin-top:13px}.teacher-answer p{font-size:11px}.teacher-workspace button:focus-visible,.teacher-workspace input:focus-visible,.teacher-workspace textarea:focus-visible,.teacher-workspace select:focus-visible{outline:2px solid #5a7eb0;outline-offset:3px}@media(max-width:900px){.teacher-intro{flex-direction:column}.teacher-section{padding:18px}.teacher-section header{align-items:start}.teacher-section select[multiple]{font-size:11px}}@media(max-width:550px){.teacher-intro h2,.teacher-section h2{font-size:14px}.teacher-intro,.teacher-section{padding:16px}.teacher-section header{flex-wrap:wrap}.teacher-buttons .btn{font-size:10px}.teacher-alert{padding:14px}}
.teacher-workspace{padding:20px;gap:16px}.teacher-intro{padding:19px;background:linear-gradient(120deg,#eaf2ff,#f7faff);border-color:#d8e6fb;align-items:center}.teacher-intro .eyebrow{font-size:8px;color:#7293bf;display:block;margin-bottom:9px}.teacher-intro h2{font-size:17px;font-weight:600;color:#285292}.teacher-intro p{font-size:11px;color:#7690b4}.teacher-intro .btn{flex-shrink:0;background:#fff;font-size:10px}.teacher-tabs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:5px;padding:5px;background:#eff5ff;border:1px solid #dde8f9;border-radius:10px;position:sticky;top:0;z-index:4}.teacher-tabs button{display:flex;justify-content:center;align-items:center;gap:7px;padding:11px 4px;background:transparent;color:#6e89ad;font-size:10px;white-space:nowrap;border-radius:7px}.teacher-tabs button>span{font-size:8px;color:#9bb2d1}.teacher-tabs button:hover{color:#2864ca;background:#e4eeff}.teacher-tabs button.active{background:#fff;box-shadow:0 2px 8px #1f55a610;color:#245edb}.teacher-tabs button.active>span{color:#7ea6e8}.teacher-section{padding:20px;border-radius:12px;background:#fff;box-shadow:0 2px 7px #214a8104}.teacher-section h2{font-size:14px;font-weight:600}.teacher-section input:not([type=checkbox]),.teacher-section textarea,.teacher-section select{background:#f9fbff;border-color:#d9e5f5;color:#3c5c89;border-radius:8px}.teacher-section button[type=submit]{color:#fff;background:#2d6ce3;border-color:#2d6ce3;box-shadow:0 3px 10px #245edb14}.teacher-section button[type=submit]:hover:not(:disabled){background:#205bc9}.teacher-success{color:#356dc3;background:#edf4ff;border:1px solid #dbe8fc}.teacher-source{border-left-color:#82adef;background:#f2f7ff;border-radius:0 8px 8px 0}.teacher-record{margin-top:20px;padding:19px 0 3px}.teacher-answer{border:1px solid #d7e6fc;background:#f0f6ff;border-radius:8px}.teacher-note{color:#7489a7}.teacher-section details{padding:10px 12px;background:#f8faff;border:1px solid #e2ebf8;border-radius:7px}.teacher-tabs~.teacher-note{margin:0}.teacher-tabs button:focus-visible{outline:2px solid var(--blue);outline-offset:2px}@media(max-width:1280px){.teacher-tabs button{gap:4px;font-size:9px}.teacher-tabs button>span{display:none}}@media(max-width:600px){.teacher-workspace{padding:14px;gap:13px}.teacher-intro{padding:16px;align-items:start}.teacher-intro h2{font-size:15px}.teacher-tabs{grid-template-columns:repeat(2,minmax(0,1fr));position:static}.teacher-tabs button{font-size:10px}.teacher-section{padding:16px}}
</style>
