import assert from 'node:assert/strict'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { join } from 'node:path'
import ts from 'typescript'
import { parse, compileScript } from '@vue/compiler-sfc'
import { createSSRApp } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import { renderToString } from '@vue/server-renderer'

// Synthetic fixtures exercise the contract only; never used by the application.
const root = fileURLToPath(new URL('../', import.meta.url))
const cache = join(root, 'node_modules', '.cache', 'activity-contract-tests')
mkdirSync(cache, { recursive: true })
function compile(source, name, replacements = {}) {
  let output = ts.transpileModule(readFileSync(join(root, source), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText
  for (const [from, to] of Object.entries(replacements)) output = output.replaceAll(from, to)
  writeFileSync(join(cache, name), output)
  return pathToFileURL(join(cache, name)).href
}
compile('src/utils/classEngagement.ts', 'classEngagement.mjs')
compile('src/utils/coursewareValidation.ts', 'coursewareValidation.mjs')
compile('src/utils/modelUpgrade.ts', 'modelUpgrade.mjs')
if (existsSync(join(root, 'src/utils/activityAnalysis.ts'))) compile('src/utils/activityAnalysis.ts', 'activityAnalysis.mjs', { './classEngagement': './classEngagement.mjs' })
compile('src/utils/workbenchNavigation.ts', 'workbenchNavigation.mjs')
compile('src/utils/teacherWorkspace.ts', 'teacherWorkspace.mjs')
const util = await import(compile('src/utils/lessonReview.ts', 'lessonReview.mjs', { './coursewareValidation': './coursewareValidation.mjs', './activityAnalysis': './activityAnalysis.mjs', './modelUpgrade': './modelUpgrade.mjs', './teacherWorkspace': './teacherWorkspace.mjs' }))
const fixture = {
  schema_version: '3.0', data_source: 'lesson_review',
  lesson: { id: 'activity-fixture', title: '行为合同夹具', duration_sec: 10, status: 'test_fixture', video_url: '/api/review/media/activity-fixture/source.mp4', source_video: { filename: 'fixture.mp4', sha256: 'same-source' }, modalities: { audio: true, slides: true, behavior: false }, provenance: { fixture: true } },
  transcript: [{ id: 'T1', start_sec: 0, end_sec: 2, text: '夹具转写', source: 'test_fixture' }], slides: [], alignment: [], graph: { nodes: [], edges: [] }, snapshots: [],
  report: { title: '合同报告', summary: '仅用于测试', sections: [], limitations: ['夹具不是实际课堂'], generated_by: { fixture: true } },
  activity_analysis: {
    schema_version: '1.0', status: 'complete', source_sha256: 'same-source', sampling_interval_sec: 1,
    context_segments: [{ id: 'X1', start_sec: 0, end_sec: 2, text: '课堂情境预测', labels: [{ label: 'lecture', name: '讲授', probability: .8 }], model_id: 'fixture-context', method: 'trained_context' }],
    frames: [
      { id: 'V1', start_sec: 0, end_sec: 1, timestamp_sec: 0, scene_id: 1, scene_cut: false, image_url: '/api/review/media/activity-fixture/activity/frames/1.jpg', status: 'observed', detected_people: 2, known_people: 1, coverage: .5, participation_index: 0, behaviors: { reading: 1 }, participants: [{ track_id: 'S1-T1', bbox: [0, 0, 50, 80], label: 'reading', confidence: .8, score: 0, source: 'pretrained_detector' }] },
      { id: 'V2', start_sec: 1, end_sec: 2, timestamp_sec: 1, scene_id: 1, scene_cut: false, image_url: '', status: 'observed', detected_people: 0, known_people: 0, coverage: 0, participation_index: null, behaviors: {}, participants: [] },
      { id: 'V3', start_sec: 4, end_sec: 5, timestamp_sec: 4, scene_id: 2, scene_cut: true, image_url: '/api/review/media/activity-fixture/activity/frames/3.jpg', status: 'observed', detected_people: 1, known_people: 1, coverage: 1, participation_index: 75, behaviors: { writing: 1 }, participants: [{ track_id: 'S2-T1', bbox: [2, 3, 50, 80], label: 'writing', confidence: .9, score: .75, source: 'rule_observation' }] },
    ],
    summary: { sampled_frames: 3, scene_count: 2, person_observations: 3, known_observations: 2 },
    models: { context: { id: 'fixture-context', method: 'frozen_encoder', metrics: { test: { macro_f1: .42, samples: 7 } }, artifact_path: 'C:\\private\\head.pt', remote_path: '/mnt/Data4/private/head.pt' } }, limitations: ['仅为可见对象，不证明知识掌握'],
    evidence: [{ id: 'E1', kind: 'behavior', start_sec: 0, end_sec: 1, text: '检测到2个可见对象，其中1个规则分数可用；规则分数覆盖率50%' }],
  },
}
const bad = mutation => { const value = structuredClone(fixture); mutation(value); return value }
// RED first: old parser silently accepts activity from an unrelated recording.
assert.throws(() => util.parseLessonPackage(bad(f => { f.activity_analysis.source_sha256 = 'other-source' })), /Invalid activity analysis data.*source|same-origin/i)
const parsed = util.parseLessonPackage(fixture)
assert.equal(parsed.activity_analysis.frames[0].participation_index, 0, 'Measured zero is retained')
assert.equal(parsed.activity_analysis.frames[1].participation_index, null, 'Unknown is retained as null')
assert.equal(JSON.stringify(parsed), JSON.stringify(fixture), 'Parsing preserves source records')
const dense = structuredClone(fixture)
const denseFrame = dense.activity_analysis.frames[0]
denseFrame.participants = [0, 1, 2].map(i => ({ track_id: `S1-T${i}`, bbox: [i * 12, 2, i * 12 + 11, 50], label: 'lookup', confidence: .9, score: null, source: 'pretrained_detector', keypoints: [], posture_tags: [], student_state: { label: 'lookup', name: '抬头线索', source: 'pretrained_detector', method: 'scbehavior_seven_yolo26n', confidence: .9, keypoint_indices: [], metrics: {}, evidence: ['七类检测测试夹具，不是注意力判断'] } }))
Object.assign(denseFrame, { detected_people: 3, known_people: 0, coverage: 0, participation_index: null, status: 'low_evidence', behaviors: { lookup: 3 }, student_state_counts: { lookup: 3 }, participation: { semantics: 'visible_participation_rule_v2', population_scope: 'visible_people_roles_unverified', index: null, status: 'observed', observed_people: 3, total_people: 3, unknown_people: 0, coverage: 1, active_people: 0, active_fraction: 0, scored_people: 0, score_coverage: 0, weights: { read: .8, write: .85, raise_hand: .95 } }, visual_reference: { semantics: 'configured_visible_behavior_reference_v1', status: 'estimated', index: 85, smoothed_index: 85, scored_people: 3, total_people: 3, support_fraction: 1, weights: { read: .8, write: .85, lookup: .85, raise_hand: .95, discuss: .75 }, weight_source: 'legacy_project_action_policy_subset', minimum_support_fraction: .6, minimum_people: 3, smoothing: { previous: .3, current: .7, maximum_gap_sec: 1.2 }, calibrated_engagement_probability: false, population_scope: 'directly_detected_people_roles_unverified' } })
dense.activity_analysis.summary.person_observations = 4
dense.activity_analysis.summary.known_observations = 1
assert.equal(util.parseLessonPackage(dense).activity_analysis.frames[0].visual_reference.index, 85, 'A trained seven-state class has a separate declared visual rule, never a geometry attention score')
const changedReference = structuredClone(dense)
changedReference.activity_analysis.frames[0].visual_reference.index = 99
assert.throws(() => util.parseLessonPackage(changedReference), /Invalid activity analysis data|Visual rule reference/i)
const fakeGeometry = structuredClone(dense)
fakeGeometry.activity_analysis.frames[0].participants[0].student_state.source = 'pose_geometry_rule'
assert.throws(() => util.parseLessonPackage(fakeGeometry), /Invalid activity analysis data|Visual rule reference/i)
const old = structuredClone(fixture); delete old.activity_analysis
assert.equal(util.parseLessonPackage(old).activity_analysis, undefined, 'Existing v3 lessons remain valid')
for (const mutate of [
  f => { f.activity_analysis.status = 'running' },
  f => { f.activity_analysis.sampling_interval_sec = 0 },
  f => { f.activity_analysis.frames[0].timestamp_sec = 20 },
  f => { f.activity_analysis.frames[0].end_sec = 10.5 },
  f => { f.activity_analysis.frames[0].start_sec = -1 },
  f => { f.activity_analysis.frames[0].end_sec = 1.2 },
  f => { f.activity_analysis.frames[0].detected_people = 1.5 },
  f => { f.activity_analysis.frames[0].known_people = 3 },
  f => { f.activity_analysis.frames[0].coverage = .8 },
  f => { f.activity_analysis.frames[0].participation_index = 101 },
  f => { f.activity_analysis.frames[1].participation_index = 0 },
  f => { f.activity_analysis.frames[0].participants[0].score = 20 },
  f => { f.activity_analysis.frames[0].participants[0].confidence = NaN },
  f => { f.activity_analysis.frames[0].participants[0].bbox = [10, 10, 0, 0] },
  f => { f.activity_analysis.frames[0].participants.push(f.activity_analysis.frames[0].participants[0]) },
  f => { f.activity_analysis.frames[0].image_url = '/api/review/media/other-lesson/1.jpg' },
  f => { f.activity_analysis.frames[0].image_url = '/api/review/media/activity-fixture/%2e%2e/secrets' },
  f => { f.activity_analysis.frames[2].scene_id = 1 },
  f => { f.activity_analysis.frames[2].participants[0].track_id = 'S1-T1' },
  f => { f.activity_analysis.context_segments[0].labels[0].probability = 1.1 },
  f => { f.activity_analysis.context_segments[0].method = 'invented' },
  f => { f.activity_analysis.evidence[0].id = 'T1' },
  f => { f.activity_analysis.evidence[0].start_sec = 2; f.activity_analysis.evidence[0].end_sec = 4 },
  f => { f.activity_analysis.evidence[0].end_sec = 10.01 },
  f => { f.activity_analysis.summary.sampled_frames = 99 },
]) assert.throws(() => util.parseLessonPackage(bad(mutate)), /Invalid activity analysis data|same-origin/i, String(mutate))
const empty = bad(f => { Object.assign(f.activity_analysis, { frames: [], context_segments: [], evidence: [], summary: { sampled_frames: 0, scene_count: 0, person_observations: 0, known_observations: 0 } }) })
assert.equal(util.parseLessonPackage(empty).activity_analysis.frames.length, 0, 'Empty lists never trigger demo data')
assert.equal(util.evidenceIndex(parsed).get('E1').text, fixture.activity_analysis.evidence[0].text)
const answer = { answer: '根据观测复盘。', source: 'local_llm', model: 'test-fixture', request_id: 'test-fixture', evidence_ids: ['E1'], citations: [fixture.activity_analysis.evidence[0]] }
assert.equal(util.parseChatResponse(answer, parsed).citations[0].kind, 'behavior')
assert.throws(() => util.parseChatResponse({ ...answer, citations: [{ ...answer.citations[0], text: '改写后的行为结论' }] }, parsed), /text or kind/i)
assert.throws(() => util.parseChatResponse({ ...answer, citations: [{ ...answer.citations[0], end_sec: 1.02 }] }, parsed), /times.*do not match/i)
assert.throws(() => util.parseChatResponse({ ...answer, citations: [{ ...answer.citations[0], kind: 'transcript' }] }, parsed), /text or kind/i)
const markdown = util.buildReviewMarkdown(parsed)
for (const expected of [/Student behavior/, /same-source/, /Temporal rule participation index/, /coverage/, /E1/, /Lesson context model predictions/, /Macro-F1/, /0\.42/, /仅为可见对象/]) assert.match(markdown, expected)
assert.doesNotMatch(markdown, /C:\\private|\/mnt\/Data4\/private/, 'Readable model provenance excludes private absolute paths')
const activity = await import(pathToFileURL(join(cache, 'activityAnalysis.mjs')).href)
assert.deepEqual(activity.participationRuns(parsed.activity_analysis.frames).map(run => run.map(frame => frame.id)), [['V1'], ['V3']], 'Null, gaps and scene cuts break the curve')
const sameSceneWithGap = [fixture.activity_analysis.frames[0], { ...fixture.activity_analysis.frames[0], id: 'V4', start_sec: 3, timestamp_sec: 3, end_sec: 4 }]
assert.equal(activity.participationRuns(sameSceneWithGap).length, 2, 'No interpolation across missing samples')
const heatmap = activity.sceneTrackRows(parsed.activity_analysis.frames, 2)
assert.deepEqual(heatmap.map(row => row.track_id), ['S2-T1'])
assert.equal(heatmap[0].cells[0].score, .75)
assert.equal(activity.sceneTrackRows(parsed.activity_analysis.frames, 1)[0].cells[1], null, 'Absent object is unknown, not a zero observation')
const modelFields = activity.readableModelFields(parsed.activity_analysis.models)
assert.ok(modelFields.some(item => item.label.includes('Macro-F1') && item.value === '0.42'))
assert.ok(modelFields.every(item => !item.value.includes('private')))
const cropFields = activity.readableModelFields({ observation: { pipeline: 'dense_person_then_seven_crop_observation_v2', models: [{ task: 'person_crop_classification', architecture: 'mobilenet_v3_small', weight_bytes: 3145239, test_metrics: { samples: 833, macro_f1: .81694 }, weight_file: 'C:\\private\\model.pt' }] } })
assert.ok(cropFields.some(row => row.value.includes('Not yet reviewed against human annotations for this lesson')))
assert.ok(cropFields.some(row => row.value.includes('same classrooms and adjacent times')))
assert.ok(cropFields.every(row => !row.value.includes('private')))
const cropFixture = structuredClone(dense)
for (const person of cropFixture.activity_analysis.frames[0].participants) person.source = person.student_state.source = 'trained_classifier'
assert.equal(util.parseLessonPackage(cropFixture).activity_analysis.frames[0].visual_reference.index, 85)
assert.match(util.buildReviewMarkdown(cropFixture), /Overall student engagement 85/)
const workspace = readFileSync(join(root, 'src/components/review/LessonReviewWorkspace.vue'), 'utf8')
assert.match(workspace, /title: 'Student behavior'/)
assert.doesNotMatch(workspace, /\?snapshots=1|Behavior snapshot workspace/, 'Removed legacy behavior entry never returns')
const component = readFileSync(join(root, 'src/components/review/LessonBehavior.vue'), 'utf8')
assert.match(component, /Detected participation action fraction/)
assert.match(component, /Visible people/)
assert.match(component, /Unknown/)
assert.match(component, /store\.seek\(/, 'Source frame actions seek original recording')
assert.doesNotMatch(component, /Math\.random|synthetic_demo|demoData/)
// Render the actual Vue component, using the real store and source fixture.
compile('src/api/reviewService.ts', 'reviewService.mjs', { '../utils/lessonReview': './lessonReview.mjs', '../utils/teacherWorkspace': './teacherWorkspace.mjs' })
compile('src/stores/lessonReview.ts', 'store.mjs', { '@/api/reviewService': './reviewService.mjs', '@/utils/lessonReview': './lessonReview.mjs', '@/utils/workbenchNavigation': './workbenchNavigation.mjs', '@/utils/teacherWorkspace': './teacherWorkspace.mjs' })
function compileComponent(source, filename, replacements = {}) {
  const { descriptor, errors } = parse(readFileSync(join(root, source), 'utf8'), { filename: source })
  assert.deepEqual(errors, [])
  const script = compileScript(descriptor, { id: filename, inlineTemplate: true })
  let output = ts.transpileModule(script.content, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText
  for (const [from, to] of Object.entries(replacements)) output = output.replaceAll(from, to)
  writeFileSync(join(cache, filename), output)
  return pathToFileURL(join(cache, filename)).href
}
const presentationUrl = compile('src/utils/reviewPresentation.ts', 'reviewPresentation.mjs')
const presentation = await import(presentationUrl)
assert.equal(presentation.reviewTerm('behavior'), 'Activity evidence')
assert.equal(presentation.reviewTerm('video_ocr'), 'Video screen OCR')
assert.equal(presentation.readableChineseText('英文噪声 noise\n北国的槐树与秋天\nPR XX'), '北国的槐树与秋天')
compileComponent('src/components/common/AppIcon.vue', 'AppIcon.mjs')
compile('src/utils/studentStates.ts', 'studentStates.mjs')
compile('src/utils/visionReview.ts', 'visionReview.mjs')
compile('src/api/visionReview.ts', 'visionReviewApi.mjs', { './reviewService': './reviewService.mjs' })
compileComponent('src/components/review/VisionFrameReview.vue', 'VisionFrameReview.mjs', { '@/api/visionReview': './visionReviewApi.mjs', '@/utils/visionReview': './visionReview.mjs', '@/utils/lessonReview': './lessonReview.mjs' })
compileComponent('src/components/review/LessonStudentState.vue', 'LessonStudentState.mjs', { '@/utils/classEngagement': './classEngagement.mjs', '@/stores/lessonReview': './store.mjs', '@/utils/lessonReview': './lessonReview.mjs', '@/utils/studentStates': './studentStates.mjs', '@/utils/reviewPresentation': './reviewPresentation.mjs', './VisionFrameReview.vue': './VisionFrameReview.mjs' })
const { default: Behavior } = await import(compileComponent('src/components/review/LessonBehavior.vue', 'LessonBehavior.mjs', { '@/utils/classEngagement': './classEngagement.mjs', '@/stores/lessonReview': './store.mjs', '@/utils/lessonReview': './lessonReview.mjs', '@/utils/activityAnalysis': './activityAnalysis.mjs', '@/utils/reviewPresentation': './reviewPresentation.mjs', '@/components/common/AppIcon.vue': './AppIcon.mjs', './LessonStudentState.vue': './LessonStudentState.mjs', '@/utils/studentStates': './studentStates.mjs' }))
const { useLessonReviewStore } = await import(pathToFileURL(join(cache, 'store.mjs')).href)
const pinia = createPinia(); setActivePinia(pinia)
const store = useLessonReviewStore()
store.data = parsed
async function renderAt(time, view = 'overview') { store.behaviorView = view; store.seek(time); const app = createSSRApp(Behavior); app.use(pinia); return renderToString(app) }
const observedHtml = await renderAt(0)
assert.match(observedHtml, /Student behavior analysis/)
assert.match(observedHtml, /data-testid="visible-participation">0\.0/, 'Actual component retains valid zero')
assert.match(await renderAt(0, 'models'), /讲授 <b>80\.0%/)
assert.doesNotMatch(observedHtml, /track-table|source-grid|behavior-provenance/, 'Overview only renders the active analysis task')
assert.match(observedHtml, /S1-T1/)
assert.doesNotMatch(observedHtml, /C:\\private|\/mnt\/Data4\/private/)
assert.match(await renderAt(1), /data-testid="visible-participation">Unknown/)
assert.match(await renderAt(2), /No behavior sample interval at this time/)
const secondSceneHtml = await renderAt(4, 'heatmap')
assert.match(secondSceneHtml, /S2-T1/)
assert.doesNotMatch(secondSceneHtml, /<th>S1-T1<\/th>/, 'Rendered heatmap never associates tracks across scenes')
const recognizedWithoutRuleScore = bad(f => {
  const frame = f.activity_analysis.frames[0]
  frame.status = 'low_evidence'
  frame.known_people = 0; frame.coverage = 0; frame.participation_index = null
  frame.behaviors = { raise_hand: 2 }
  frame.participants = [
    { ...frame.participants[0], label: 'raise_hand', score: null },
    { ...frame.participants[0], track_id: 'S1-T2', label: 'raise_hand', score: null },
  ]
  f.activity_analysis.summary.known_observations = 1
  f.activity_analysis.evidence[0].text = '即时举手标签2个；规则分数可用对象0个；规则分数覆盖率0%；规则参与指数未知。'
})
store.data = util.parseLessonPackage(recognizedWithoutRuleScore)
const noScoreHtml = await renderAt(0)
assert.match(noScoreHtml, /Rule-score coverage/, 'Coverage describes available temporal rule scores, not instantaneous recognition')
assert.match(noScoreHtml, /data-testid="visible-participation">Unknown/)
assert.match(noScoreHtml, /Hand raising<\/span><b>2<\/b>/, 'Recognized labels remain visible even when no temporal rule score is available')
const rawReadWrite = structuredClone(recognizedWithoutRuleScore)
rawReadWrite.activity_analysis.frames[0].behaviors = { read: 1, write: 1 }
rawReadWrite.activity_analysis.frames[0].participants[0].label = 'read'
rawReadWrite.activity_analysis.frames[0].participants[1].label = 'write'
store.data = util.parseLessonPackage(rawReadWrite)
const readWriteHtml = await renderAt(0)
assert.match(readWriteHtml, /Reading<\/span><b>1<\/b>/)
assert.match(readWriteHtml, /Writing<\/span><b>1<\/b>/)
store.data = util.parseLessonPackage(recognizedWithoutRuleScore)
assert.match(noScoreHtml, /data-testid="visible-participation">Unknown/)
assert.doesNotMatch(noScoreHtml, /Recognized behavior|Recognized behavior coverage/)
const noScoreMarkdown = util.buildReviewMarkdown(store.data)
assert.match(noScoreMarkdown, /Observations with usable rule scores: 1/)
assert.match(noScoreMarkdown, /Detected 2. People with usable rule scores 0; Rule-score coverage 0\.0%/)
assert.match(noScoreMarkdown, /raise_hand: 2/)
assert.doesNotMatch(noScoreMarkdown, /Recognized behavior observations: /)
// V2 visible states never turn posture-only evidence into a psychological score.
const visible = structuredClone(fixture)
const state = (label, source = 'pose_geometry_rule') => ({ label, name: label, source, confidence: .8, evidence: ['合同夹具的可见依据'], keypoint_indices: [] })
const participation = (total, observed, active, index) => ({ index, status: total ? 'observed' : 'no_people', total_people: total, observed_people: observed, unknown_people: total - observed, coverage: total ? observed / total : null, active_people: active, active_fraction: total ? active / total : null, scored_people: active, score_coverage: total ? active / total : null, weights: { read: .8, write: .85, raise_hand: .95 }, semantics: 'visible_participation_rule_v2', population_scope: 'visible_people_roles_unverified' })
for (const frame of visible.activity_analysis.frames) { frame.width = 1280; frame.height = 720 }
visible.activity_analysis.frames[0].participants[0].student_state = state('read', 'pretrained_detector')
visible.activity_analysis.frames[0].student_state_counts = { read: 1, unknown: 1 }
visible.activity_analysis.frames[0].participation = participation(2, 1, 1, 80)
visible.activity_analysis.frames[1].student_state_counts = {}
visible.activity_analysis.frames[1].participation = participation(0, 0, 0, null)
visible.activity_analysis.frames[2].participants[0].student_state = state('head_up')
visible.activity_analysis.frames[2].student_state_counts = { head_up: 1 }
visible.activity_analysis.frames[2].participation = participation(1, 1, 0, null)
const originalVisible = JSON.stringify(visible)
const unmatchedPose = structuredClone(visible)
unmatchedPose.activity_analysis.frames[0].participants[0].keypoints = []
assert.doesNotThrow(() => util.parseLessonPackage(unmatchedPose), 'An unmatched pose retains its source action and an empty keypoint list')
const invalidPose = structuredClone(visible)
invalidPose.activity_analysis.frames[0].participants[0].keypoints = [[1, 1, .8]]
assert.throws(() => util.parseLessonPackage(invalidPose), /Invalid activity analysis data/i, 'Partial malformed keypoint arrays remain invalid')
store.data = util.parseLessonPackage(visible)
assert.equal(JSON.stringify(store.data), originalVisible)
const actionHtml = await renderAt(0)
assert.match(actionHtml, /data-testid="visible-action-fraction">50/)
assert.match(actionHtml, /data-testid="visible-participation">80\.0/)
assert.match(actionHtml, /student-state-boxes/)
assert.match(actionHtml, /viewBox="0 0 1280 720"/)
assert.match(actionHtml, /student-diagnostics/)
assert.match(await renderAt(1), /data-testid="visible-action-fraction">Unknown/)
const postureHtml = await renderAt(4)
assert.match(postureHtml, /Head raised/)
assert.match(postureHtml, /data-testid="visible-action-fraction">0/)
assert.match(postureHtml, /data-testid="visible-participation">Unknown/, 'Head position alone never receives an engagement score')
store.data = util.parseLessonPackage(cropFixture)
const sevenHtml = await renderAt(0)
assert.match(sevenHtml, /The seven-class candidates for this lesson have not passed human annotation review/)
assert.match(sevenHtml, /data-testid="overall-engagement-index">85\.0/)
assert.match(sevenHtml, /data-testid="visible-state-timeline"/)
assert.match(sevenHtml, /Geometric posture/)
assert.match(sevenHtml, /These proportions do not describe the whole class or psychological engagement/)
for (const mutation of [
  f => { f.activity_analysis.frames[0].student_state_counts.read = 2 },
  f => { f.activity_analysis.frames[0].participation.active_fraction = .9 },
  f => { f.activity_analysis.frames[0].participation.index = 95 },
  f => { f.activity_analysis.frames[0].participation.weights.head_up = .9 },
  f => { f.activity_analysis.frames[1].participation.index = 0 },
  f => { f.activity_analysis.frames[1].participation.coverage = 0 },
  f => { f.activity_analysis.frames[2].participation.index = 90 },
  f => { f.activity_analysis.frames[0].participants[0].bbox[2] = 1400 },
]) { const bad = structuredClone(visible); mutation(bad); assert.throws(() => util.parseLessonPackage(bad), /Invalid activity analysis data/i) }
store.data = old
const missingHtml = await renderAt(0)
assert.match(missingHtml, /No behavior analysis from the same source is available for this lesson/)
assert.doesNotMatch(missingHtml, /participation-chart|track-table/)
console.log('PASS: same-source optional activity contract, exact evidence/time citations, null versus zero, scene-local heatmap, curve gaps, readable provenance and export.')
