import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { join } from 'node:path'
import ts from 'typescript'
import { parse, compileScript } from '@vue/compiler-sfc'
import { createSSRApp } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import { renderToString } from '@vue/server-renderer'

// Synthetic fixtures test UI state and source boundaries; they are never application data.
const root = fileURLToPath(new URL('../', import.meta.url))
const cache = join(root, 'node_modules/.cache/workbench-contract-tests')
mkdirSync(cache, { recursive: true })
const replacements = {
  '@/api/reviewService': './reviewService.mjs', '@/stores/lessonReview': './store.mjs',
  '@/utils/lessonReview': './lessonReview.mjs', '../utils/lessonReview': './lessonReview.mjs',
  '@/utils/workbenchNavigation': './workbenchNavigation.mjs',
  '@/utils/teacherWorkspace': './teacherWorkspaceUtils.mjs', '../utils/teacherWorkspace': './teacherWorkspaceUtils.mjs', './teacherWorkspace': './teacherWorkspaceUtils.mjs',
  '@/utils/studentStates': './studentStates.mjs', '@/utils/activityAnalysis': './activityAnalysis.mjs', '@/utils/reviewPresentation': './reviewPresentation.mjs',
  './coursewareValidation': './coursewareValidation.mjs', './activityAnalysis': './activityAnalysis.mjs',
  './modelUpgrade': './modelUpgrade.mjs', '@/utils/modelUpgrade': './modelUpgrade.mjs',
  '@/components/common/AppIcon.vue': './AppIcon.mjs',
  '@/utils/classEngagement': './classEngagement.mjs', './classEngagement': './classEngagement.mjs',
  '@/api/visionReview': './visionReviewApi.mjs', '@/utils/visionReview': './visionReview.mjs', './reviewService': './reviewService.mjs',
}
const components = ['VisionFrameReview', 'LessonStudentState', 'LessonOverview', 'LessonPlayer', 'LessonCorrectionComparison', 'LessonModelUpgrade', 'LessonEvidence', 'LessonCourseware', 'LessonEvidenceWorkspace', 'LessonBehavior', 'LessonGraph', 'LessonReport', 'LessonAssistant', 'TeacherWorkspace', 'LessonReviewWorkspace']
for (const name of components) replacements[`./${name}.vue`] = `./${name}.mjs`
function compile(source, name, component = false) {
  let content = readFileSync(join(root, source), 'utf8')
  if (component) {
    const { descriptor, errors } = parse(content, { filename: source })
    assert.deepEqual(errors, [])
    content = compileScript(descriptor, { id: name, inlineTemplate: true }).content
  }
  let output = ts.transpileModule(content, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText
  // Long names first avoid replacing './activityAnalysis' inside a generated extension twice.
  for (const [from, to] of Object.entries(replacements)) output = output.replaceAll(`'${from}'`, `'${to}'`).replaceAll(`"${from}"`, `"${to}"`)
  writeFileSync(join(cache, `${name}.mjs`), output)
  return pathToFileURL(join(cache, `${name}.mjs`)).href
}
for (const name of ['classEngagement', 'studentStates', 'coursewareValidation', 'activityAnalysis', 'reviewPresentation', 'workbenchNavigation', 'modelUpgrade', 'lessonReview']) compile(`src/utils/${name}.ts`, name)
compile('src/utils/teacherWorkspace.ts', 'teacherWorkspaceUtils')
compile('src/api/reviewService.ts', 'reviewService')
compile('src/utils/visionReview.ts', 'visionReview')
compile('src/api/visionReview.ts', 'visionReviewApi')
compile('src/stores/lessonReview.ts', 'store')
compile('src/components/common/AppIcon.vue', 'AppIcon', true)
for (const name of components) compile(`src/components/review/${name}.vue`, name, true)
const { parseWorkbenchRoute, workbenchUrl } = await import(pathToFileURL(join(cache, 'workbenchNavigation.mjs')).href)
const { proofreadingModelName, lessonAsrModelName } = await import(pathToFileURL(join(cache, 'reviewPresentation.mjs')).href)
assert.equal(proofreadingModelName('classroom-corrector'), 'ChineseErrorCorrector4-4B (CEC4)')
assert.equal(proofreadingModelName('classroom-local'), 'Qwen3-4B')
assert.equal(proofreadingModelName('classroom-corrector-qwen35'), 'Qwen3.5-4B')
assert.equal(proofreadingModelName('classroom-corrector-qwen35', { model_label: 'Qwen3.5-4B verified file label' }), 'Qwen3.5-4B verified file label')
assert.equal(proofreadingModelName('classroom-corrector-qwen35', { model_label: {} }), 'Qwen3.5-4B')
assert.equal(lessonAsrModelName({ asr: { config: { model_id: 'Qwen/Qwen3-ASR-1.7B' } } }), 'Qwen3-ASR-1.7B')
assert.equal(lessonAsrModelName({ asr: { config: { model_id: 'Systran/faster-whisper-small' } } }), 'faster-whisper-small')
assert.equal(lessonAsrModelName({}), 'ASR model not provided')
assert.equal(proofreadingModelName('explicit-model-label'), 'explicit-model-label', 'Unknown model labels are preserved rather than relabelled')
assert.deepEqual(parseWorkbenchRoute('?lesson=中文课程&view=behavior'), { view: 'behavior', lesson: '中文课程' })
assert.equal(parseWorkbenchRoute('?view=assistant').view, 'review', 'Assistant is an on-demand panel, never a replacement view')
assert.equal(parseWorkbenchRoute('?view=unknown').view, 'review')
assert.equal(parseWorkbenchRoute('?view=teaching').view, 'teaching', 'Teaching improvement has a stable sixth deep link')
assert.match(readFileSync(join(root, 'src/components/review/LessonReviewWorkspace.vue'), 'utf8'), /title: 'Teaching improvement'/, 'The teacher workspace is an accessible sixth navigation item')
const url = workbenchUrl('http://localhost:5173/?lesson=old&import=preserved#old-anchor', 'knowledge', '中文课程')
assert.equal(url.searchParams.get('import'), 'preserved')
assert.equal(url.searchParams.get('lesson'), '中文课程')
assert.equal(url.searchParams.get('view'), 'knowledge')
assert.equal(url.hash, '')
for (const flag of ['snapshots', 'data', 'demo']) assert.equal(workbenchUrl(`http://localhost/?${flag}=1`, 'review').searchParams.has(flag), false)
assert.doesNotMatch(readFileSync(join(root, 'src/App.vue'), 'utf8'), /Legacy|query|snapshots|demo/, 'Removed flags cannot load a legacy app')
assert.doesNotMatch(readFileSync(join(root, 'src/components/review/LessonReviewWorkspace.vue'), 'utf8'), /Behavior snapshot workspace|snapshots=1/)

// The store must also construct without browser globals during SSR.
const previousWindow = globalThis.window
delete globalThis.window
const { useLessonReviewStore } = await import(pathToFileURL(join(cache, 'store.mjs')).href)
const { default: Workspace } = await import(pathToFileURL(join(cache, 'LessonReviewWorkspace.mjs')).href)
const pinia = createPinia(); setActivePinia(pinia)
const store = useLessonReviewStore()
assert.equal(store.activeView, 'review')
store.data = {
  schema_version: '3.0', data_source: 'lesson_review',
  lesson: { id: 'workbench-fixture', title: '同课合同夹具', duration_sec: 10, status: 'test_fixture', video_url: '/api/review/media/workbench-fixture/source.mp4', source_video: { filename: 'fixture.mp4', sha256: 'same-source' }, modalities: { audio: true, slides: true, behavior: true }, provenance: { fixture: true } },
  transcript: [{ id: 'T1', start_sec: 0, end_sec: 1, text: '当前原声测试片段', source: 'test_fixture', quality: { status: 'machine_transcript_ocr_vocabulary_prompted_unreviewed', note: 'fixture provenance retained' } }],
  slides: [{ id: 'S1', start_sec: 0, end_sec: 1, title: '当前课件测试标题', text: '当前课件正文', image_url: '/api/review/media/workbench-fixture/slide.jpg', method: 'test_fixture', uncertainty_sec: 0 },
    { id: 'S2', start_sec: 1, end_sec: 2, title: '缺失投影占位', text: '不应当作课件展示的全班画面', image_url: '/api/review/media/workbench-fixture/classroom.jpg', method: 'no_slide_evidence', uncertainty_sec: 0 },
    { id: 'S3', start_sec: 2, end_sec: 3, title: '质量字段缺失投影占位', text: '不应当作课件展示的教师画面', image_url: '/api/review/media/workbench-fixture/teacher.jpg', method: 'video_ocr', quality: { screen_status: 'no_slide' }, uncertainty_sec: 0 }],
  alignment: [], graph: { nodes: [], edges: [] }, snapshots: [{ original: 'legacy retained' }],
  report: { title: '报告测试标题', summary: '报告来源保留', sections: [], limitations: ['课堂使用边界'], generated_by: 'fixture' },
  activity_analysis: { schema_version: '1.0', status: 'complete', source_sha256: 'same-source', sampling_interval_sec: 1,
    frames: [{ id: 'F1', start_sec: 0, end_sec: 1, timestamp_sec: 0, scene_id: 1, scene_cut: false, image_url: '', status: 'observed', detected_people: 1, known_people: 1, coverage: 1, participation_index: 0, behaviors: { reading: 1 }, participants: [] }],
    context_segments: [], evidence: [], models: {}, limitations: ['不得推断真实积极度'], summary: { sampled_frames: 1, scene_count: 1, person_observations: 1, known_observations: 1 } },
}
store.lessons = [{ id: store.data.lesson.id, title: store.data.lesson.title, duration_sec: 10, status: 'test_fixture' }]
store.data.transcript_proofreading = { schema_version: '1.0', status: 'complete', method: 'source_constrained_llm', model: 'fixture-model', human_verified: false, source_asr_sha256: 'fixture-asr', source_video_sha256: 'same-source', segments: [{ id: 'P0001', transcript_id: 'T1', start_sec: 0, end_sec: 1, original_text: store.data.transcript[0].text, text: '机器校对后的测试片段', source_refs: ['T1', 'S1'], edits: [], uncertainties: [], quality: { status: 'automatically_proofread_unreviewed' } }], counts: {}, provenance: { fixture: true } }
const originalPackage = JSON.stringify(store.data)
async function render(view, time = .5) {
  store.activeView = view; store.updateTime(time)
  const app = createSSRApp(Workspace); app.use(pinia)
  return renderToString(app)
}
const overview = await render('review')
for (const text of ['机器校对后的测试片段', 'overview-speech', 'overview-behavior', 'Jingxi Insight']) assert.ok(overview.includes(text), text)
assert.doesNotMatch(overview, /overview-courseware|Lesson courseware image|Three synchronized sources|Former teaching demo/)
assert.match(overview, /data-testid="visible-participation">0\.0/, 'Overview retains a valid observed zero')
assert.match(overview, /Machine correction · Unverified/)
assert.doesNotMatch(overview, /当前原声测试片段/, 'Default overview displays corrected text while preserving raw in data')
assert.match(overview, /transcript-raw-toggle/)
assert.doesNotMatch(overview, /lesson-behavior|lesson-evidence|lesson-graph|lesson-report|review-assistant"/, 'Overview renders only its own task')
const gap = await render('review', 1)
assert.equal(store.observedSlides.length, 1, 'Missing-screen placeholders remain in source data and never become observed slides')
assert.match(gap, /No transcript is recorded at this time/)
assert.doesNotMatch(gap, /overview-courseware|No aligned slide image at this time|Locate visible courseware for this lesson/)
assert.doesNotMatch(gap, /classroom\.jpg|teacher\.jpg/, 'Classroom/teacher placeholder pictures must not be presented as courseware')
assert.match(gap, /No behavior sample interval at this time/)
assert.doesNotMatch(gap.match(/<section class="lesson-overview"[\s\S]*?<\/section>/)?.[0] || '', /当前原声测试片段|当前课件测试标题/, 'Evidence at the previous frame cannot leak into a gap summary')
for (const [view, className] of [['evidence', 'evidence-workspace'], ['behavior', 'lesson-behavior'], ['knowledge', 'lesson-graph'], ['report', 'lesson-report'], ['teaching', 'teacher-workspace']]) {
  const html = await render(view)
  assert.match(html, new RegExp(`data-view-panel="${view}"`))
  assert.ok(html.includes(className))
  if (view === 'evidence') { assert.match(html, /机器校对后的测试片段/); assert.match(html, /transcript-proofread-default/); assert.match(html, /transcript-raw-toggle/) }
  assert.doesNotMatch(html, /data-testid="correction-comparison"/, 'No comparison panel or demo numbers without actual run records')
  assert.doesNotMatch(html, /data-testid="model-upgrade-experiment"/, 'No upgrade panel or invented experiment without actual records')
  assert.equal((html.match(/<video /g) || []).length, 1, `${view} keeps one original player`)
  assert.doesNotMatch(html, /data-testid="lesson-overview"/)
}
store.data.teacher_transcript = [{ id: 'HT1', transcript_id: 'T1', text: '教师已核准的夹具', start_sec: 0, end_sec: 1, source_refs: ['T1'], source: 'teacher_revision', human_verified: true, correction_id: 'C1' }]
const teacherOverview = await render('review')
assert.match(teacherOverview, /教师已核准的夹具/)
assert.match(teacherOverview, /Teacher verified/)
assert.match(await render('evidence'), /Teacher verified/)
store.data.teacher_transcript[0] = { ...store.data.teacher_transcript[0], source: 'machine_context', human_verified: false, correction_id: null }
assert.match(await render('evidence'), /Machine context · Unverified/)
delete store.data.teacher_transcript
const comparisonFixture = {
  schema_version: '1.0', source_video_sha256: 'same-source', evaluated_at: '2026-10-28T00:00:00Z', scope: 'same_lesson_controlled_model_comparison', reference_status: 'no_human_verbatim_reference', limitations: ['合同测试夹具，非实际模型结果'],
  runs: [
    { id: 'qwen', model_label: 'Qwen3-4B', model_sha256: 'fixture-qwen', accepted_edit_count: 0, model_proposal_count: 2, guard_rejected_count: 2, term_presence_count: 1, terminology_group_count: 4, memory: { unit: 'MiB', scope: 'whole_gpu_sampled', baseline_mib: 111, loaded_mib: 222, peak_mib: 333, sample_interval_ms: 100 }, segments: [{ transcript_id: 'T1', start_sec: 0, end_sec: 1, original_text: store.data.transcript[0].text, text: store.data.transcript[0].text, uncertainties: [] }] },
    { id: 'cec4', model_label: 'ChineseErrorCorrector4-4B', model_sha256: 'fixture-cec4', accepted_edit_count: 1, segments: [{ transcript_id: 'T1', start_sec: 0, end_sec: 1, original_text: store.data.transcript[0].text, text: '对比模型测试输出', uncertainties: ['需要回听原声'] }] },
  ],
}
store.data.correction_comparison = comparisonFixture
for (const run of comparisonFixture.runs) {
  run.status = 'partial'; run.native_status = 'complete'
  run.model_accepted_edit_count = 0; run.source_rule_accepted_count = run.accepted_edit_count
  run.additional_changed_spans_vs_rules = run.id === 'qwen' ? 1 : 0
  run.segments[0].uncertainties = ['U2_0030']
  run.native_segments = run.segments.map(row => ({ ...row, text: '仅用于模型直接输出的合同夹具', uncertainties: ['未经规则校验'] }))
}
store.data.transcript_proofreading.model = 'classroom-corrector'
const withComparison = await render('evidence')
assert.match(withComparison, /data-testid="active-correction-model">ChineseErrorCorrector4-4B \(CEC4\)/)
assert.match(withComparison, /<details class="correction-comparison"[^>]*data-testid="correction-comparison"/)
assert.doesNotMatch(withComparison, /<details[^>]*\bopen(?:=|>| )/, 'Model comparison starts collapsed')
for (const text of ['correction-comparison-toggle', 'correction-comparison-run-qwen', 'correction-comparison-run-cec4', 'comparison-segment-T1', 'Qwen3-4B', 'ChineseErrorCorrector4-4B', 'Sampled whole-GPU memory', '333 MiB', 'Unknown', '当前原声测试片段', '对比模型测试输出', 'This lesson has no human verbatim transcript', 'It is not used in the knowledge graph or Q&amp;A citations']) assert.ok(withComparison.includes(text), text)
for (const text of ['Accepted model candidates', 'Source-rule revisions', 'Total accepted edits', 'Controlled correction: Partially completed', 'Direct output: Completed', 'correction-comparison-lane', 'Guarded results', 'Direct model output']) assert.ok(withComparison.includes(text), text)
assert.doesNotMatch(withComparison, /仅用于模型直接输出的合同夹具/, 'Raw native hypotheses are not the default guarded results')
assert.match(withComparison, /Additional changes beyond rule-only results: 1 edits \(unverified\)/)
assert.match(withComparison, /Additional changes beyond rule-only results: 0 edits \(unverified\)/)
assert.match(withComparison, /This segment still contains unconfirmed content. Replay the original audio/)
assert.doesNotMatch(withComparison, /U2_0030|Explicit model contribution|Source-rule contribution/, 'Display uses auditable counts without claiming unique model benefits or showing an unexplained uncertainty ID')
assert.equal((withComparison.match(/<video /g) || []).length, 1)
assert.doesNotMatch(withComparison, /CER|Erroneous correction rate[: :]|Accuracy[: :]/, 'No unsupported accuracy score')
assert.deepEqual(JSON.parse(JSON.stringify(store.data)).correction_comparison, comparisonFixture, 'Full JSON export preserves both run records')
delete store.data.correction_comparison
store.data.transcript_proofreading.model = 'fixture-model'
// Upgrade trial is an optional compact disclosure, only inside speech/courseware.
const upgradeData = JSON.parse(JSON.stringify(store.data))
store.data.lesson.provenance.asr = { config: { model_id: 'Qwen/Qwen3-ASR-1.7B' }, audio_sha256: 'fixture-audio', source_asr_sha256: 'fixture-asr' }
store.data.transcript_proofreading.model = 'classroom-corrector-qwen35'
store.data.transcript_proofreading.provenance.model_label = 'Qwen3.5-4B'
store.data.model_upgrade_experiment = {
  schema_version: '1.0', source_video_sha256: 'same-source', source_audio_sha256: 'fixture-audio', evaluated_at: '2026-10-28T00:00:00Z', reference_status: 'no_human_verbatim_reference',
  asr_runs: [
    { id: 'baseline', model_label: 'faster-whisper-small', source_asr_sha256: 'fixture-old-asr', segments: [{ id: 'BASE1', start_sec: .123, end_sec: .7, text: '原模型自身时间内的夹具' }, { id: 'BASE2', start_sec: 2, end_sec: 3, text: '原模型其他时刻的夹具' }] },
    { id: 'qwen3_asr', model_label: 'Qwen3-ASR-1.7B', source_asr_sha256: 'fixture-asr', segments: JSON.parse(JSON.stringify(store.data.transcript)) },
  ],
  correction: { model_label: 'Qwen3.5-4B', model_sha256: 'fixture-model-hash', source_asr_sha256: 'fixture-asr', accepted_edit_count: 0, model_accepted_edit_count: 0, source_rule_accepted_count: 0, status: 'complete', segments: store.data.transcript_proofreading.segments.map(({ transcript_id, start_sec, end_sec, original_text, text, uncertainties }) => ({ transcript_id, start_sec, end_sec, original_text, text, uncertainties })) },
  limitations: ['合同夹具边界说明'], external_summary: { opaque: 'preserved' },
}
const upgradedEvidence = await render('evidence')
for (const value of ['Model upgrade experiment', 'Baseline ASR', 'New ASR', 'Conservative correction', 'faster-whisper-small', 'Qwen3-ASR-1.7B', 'Qwen3.5-4B', 'Not evaluated', '0.123 s–0.7 s', 'model-upgrade-baseline-BASE1', 'model-upgrade-qwen3_asr-T1', 'model-upgrade-correction-T1']) assert.ok(upgradedEvidence.includes(value), value)
assert.match(upgradedEvidence, /data-testid="active-correction-model">Qwen3.5-4B/)
assert.match(upgradedEvidence, /data-testid="active-asr-model">Speech recognition: Qwen3-ASR-1.7B/)
assert.doesNotMatch(upgradedEvidence, /model-upgrade-baseline-BASE2|原模型其他时刻的夹具/, 'Current-time view independently excludes off-clock baseline segments')
assert.match(upgradedEvidence, /<details class="model-upgrade"[^>]*data-testid="model-upgrade-experiment"/)
assert.doesNotMatch(upgradedEvidence, /<details[^>]*\bopen(?:=|>| )/, 'Upgrade trial starts collapsed')
assert.equal((upgradedEvidence.match(/<video /g) || []).length, 1)
assert.match(await render('evidence', 1.5), /No New ASR segment at the current time/)
for (const view of ['review', 'behavior', 'knowledge', 'report']) assert.doesNotMatch(await render(view), /data-testid="model-upgrade-experiment"/, 'Upgrade trial never piles onto other workbench views')
assert.deepEqual(JSON.parse(JSON.stringify(store.data)).model_upgrade_experiment.external_summary, { opaque: 'preserved' })
store.data = upgradeData
store.assistantOpen = true
store.scope = 'current'
store.messages = [{ id: 1, query: '保留的问题', at_sec: .5, scope: 'current', status: 'error', error: '来源模型未返回' }]
assert.match(await render('knowledge'), /保留的问题/)
store.assistantOpen = false
assert.doesNotMatch(await render('report'), /id="review-assistant-drawer"/)
store.assistantOpen = true
assert.match(await render('behavior'), /保留的问题/)
assert.equal(store.scope, 'current')
assert.equal(store.messages.length, 1)
assert.equal(JSON.stringify(store.data), originalPackage, 'Navigation and helper visibility preserve the complete source package')

// Simulate real URL transitions: no time/playback mutation and no duplicate history entries.
const historyEntries = []
globalThis.window = { location: { href: 'http://localhost:5173/?view=behavior&lesson=workbench-fixture', search: '?view=behavior&lesson=workbench-fixture' }, history: { pushState: (_state, _title, route) => historyEntries.push(new URL(route)) } }
store.activeView = 'behavior'; store.currentTime = 7.25; store.isPlaying = true
const seekVersion = store.seekVersion
store.setView('knowledge'); store.setView('knowledge')
assert.equal(historyEntries.length, 1)
assert.equal(historyEntries[0].searchParams.get('lesson'), 'workbench-fixture')
assert.equal(store.currentTime, 7.25)
assert.equal(store.isPlaying, true)
assert.equal(store.seekVersion, seekVersion, 'Feature navigation never sends a seek to the mounted video')
window.location.search = '?view=evidence&lesson=workbench-fixture'
await store.restoreRoute()
assert.equal(store.activeView, 'evidence', 'Back/forward restores the query view')
assert.equal(store.currentTime, 7.25, 'Back/forward within the same lesson preserves playback time')
assert.equal(historyEntries.length, 1, 'History restoration must not push a new entry')
if (previousWindow === undefined) delete globalThis.window
else globalThis.window = previousWindow
const serviceModule = await import(pathToFileURL(join(cache, 'reviewService.mjs')).href)
const originalLessonRequest = serviceModule.reviewService.lesson
const lessonA = JSON.parse(JSON.stringify(store.data))
const lessonB = { ...lessonA, lesson: { ...lessonA.lesson, id: 'second-lesson' } }
assert.equal(typeof store.refreshLesson, 'function', 'Background refresh must not create a navigation intent')
try {
  const calls = []; let finishB
  serviceModule.reviewService.lesson = id => { calls.push(id); return new Promise(resolve => { finishB = resolve }) }
  const navigation = store.selectLesson('second-lesson')
  assert.equal(await store.refreshLesson(lessonA.lesson.id), false, 'A pending navigation takes precedence over background work')
  assert.deepEqual(calls, ['second-lesson'])
  finishB(lessonB); await navigation
  assert.equal(store.data.lesson.id, 'second-lesson')
  store.data = lessonA
  let finishA
  serviceModule.reviewService.lesson = id => new Promise(resolve => { if (id === lessonA.lesson.id) finishA = resolve; else finishB = resolve })
  const refresh = store.refreshLesson(lessonA.lesson.id)
  const nextNavigation = store.selectLesson('second-lesson')
  finishB(lessonB); await nextNavigation
  finishA(lessonA)
  assert.equal(await refresh, false, 'A late old-lesson refresh must not overwrite a newer navigation')
  assert.equal(store.data.lesson.id, 'second-lesson')
} finally { serviceModule.reviewService.lesson = originalLessonRequest }
console.log('PASS: deep links, query preservation, SSR active views, same-time triad gaps/zero, assistant state, complete source retention and history without playback mutation.')
