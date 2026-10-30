import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { join } from 'node:path'
import ts from 'typescript'
import { createPinia, setActivePinia } from 'pinia'

// Contract fixtures are only used here. They are never bundled or loaded by the UI.
const root = fileURLToPath(new URL('../', import.meta.url))
const cache = join(root, 'node_modules', '.cache', 'review-contract-tests')
mkdirSync(cache, { recursive: true })
function compile(source, filename, replacements = {}) {
  let text = ts.transpileModule(readFileSync(join(root, source), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText
  for (const [from, to] of Object.entries(replacements)) text = text.replaceAll(from, to)
  writeFileSync(join(cache, filename), text)
  return pathToFileURL(join(cache, filename)).href
}
compile('src/utils/coursewareValidation.ts', 'coursewareValidation.mjs')
compile('src/utils/classEngagement.ts', 'classEngagement.mjs')
compile('src/utils/activityAnalysis.ts', 'activityAnalysis.mjs', { './classEngagement': './classEngagement.mjs' })
compile('src/utils/workbenchNavigation.ts', 'workbenchNavigation.mjs')
compile('src/utils/teacherWorkspace.ts', 'teacherWorkspace.mjs')
const modelUpgradeUrl = compile('src/utils/modelUpgrade.ts', 'modelUpgrade.mjs')
const utilUrl = compile('src/utils/lessonReview.ts', 'lessonReview.mjs', { './coursewareValidation': './coursewareValidation.mjs', './activityAnalysis': './activityAnalysis.mjs', './modelUpgrade': './modelUpgrade.mjs', './teacherWorkspace': './teacherWorkspace.mjs' })
const serviceUrl = compile('src/api/reviewService.ts', 'reviewService.mjs', { '../utils/lessonReview': './lessonReview.mjs', '../utils/teacherWorkspace': './teacherWorkspace.mjs' })
const storeUrl = compile('src/stores/lessonReview.ts', 'store.mjs', { '@/api/reviewService': './reviewService.mjs', '@/utils/lessonReview': './lessonReview.mjs', '@/utils/workbenchNavigation': './workbenchNavigation.mjs', '@/utils/teacherWorkspace': './teacherWorkspace.mjs' })
const { parseLessonList, parseLessonPackage, activeAt, conceptIntervals, parseChatResponse, buildReviewMarkdown, formatReviewTime, reviewMediaUrl, evidenceIndex } = await import(utilUrl)
const { reviewService, ReviewServiceError } = await import(serviceUrl)
const { useLessonReviewStore } = await import(storeUrl)
const { modelUpgradeLanes, modelUpgradeTime } = await import(modelUpgradeUrl)

const fixture = {
  schema_version: '3.0', data_source: 'lesson_review',
  lesson: { id: 'contract-fixture', title: '合同测试夹具（非课堂结果）', duration_sec: 1144.4333, status: 'test_fixture', video_url: '/api/review/media/contract-fixture/source.mp4', source_video: { filename: 'fixture.mp4', sha256: 'fixture-hash-not-real' }, modalities: { audio: true, slides: true, behavior: false }, provenance: { fixture: true, method: 'contract test only' } },
  transcript: [{ id: 'T0001', start_sec: 3.21, end_sec: 8.96, text: '第一段概念A', source: 'test_fixture' }, { id: 'T0002', start_sec: 400.35, end_sec: 408.72, text: '再次讲到概念A', source: 'test_fixture' }],
  slides: [{ id: 'S0001', start_sec: 0, end_sec: 10, title: '夹具页', text: '概念A', image_url: '/api/review/media/contract-fixture/slide.jpg', method: 'test_fixture', uncertainty_sec: 2 }, { id: 'S0002', start_sec: 399, end_sec: 412, title: '重复主题页', text: '概念A再次出现', image_url: '', method: 'test_fixture', uncertainty_sec: 2 }],
  alignment: [], graph: { nodes: [{ id: 'C1', label: '概念A', properties: { kind: 'concept', source_refs: ['T0001', 'T0002'], status: 'source_extracted' } }], edges: [] },
  report: { title: '夹具报告', summary: '只用于测试。', sections: [{ title: '证据概览', text: '重复段落不应合并。', evidence_ids: ['T0001', 'T0002'] }], limitations: ['夹具不是实际课堂结果'], generated_by: { method: 'test_fixture' } }, snapshots: [],
}
writeFileSync(join(cache, 'fixture.json'), JSON.stringify(fixture))
const before = JSON.stringify(fixture)
const parsed = parseLessonPackage(fixture)
assert.equal(JSON.stringify(parsed), before, 'Parsing must preserve original provenance and optional source fields')
assert.equal(formatReviewTime(parsed.lesson.duration_sec), '19:04')
assert.equal(parseLessonList({ lessons: [] }).length, 0, 'Empty catalog must remain empty')
assert.equal(parsed.snapshots.length, 0, 'No synthetic behavior frames')
assert.equal(activeAt(parsed.transcript, 5)[0].id, 'T0001')
assert.equal(activeAt(parsed.transcript, 8.96).length, 0, 'Use half-open evidence intervals')
assert.equal(activeAt(parsed.slides, 10).length, 0, 'Do not smear a slide into a gap')
assert.deepEqual(conceptIntervals(parsed, 'C1').map(e => [e.start_sec, e.end_sec]), [[3.21, 8.96], [400.35, 408.72]])
assert.equal(activeAt(conceptIntervals(parsed, 'C1'), 200).length, 0, 'Repeated concept must not cover the unobserved middle')
assert.throws(() => parseLessonPackage({ ...fixture, data_source: 'synthetic_demo' }), /v3/)
assert.throws(() => parseLessonPackage({ ...fixture, transcript: [...fixture.transcript, fixture.transcript[0]] }), /重复/)
assert.throws(() => parseLessonPackage({ ...fixture, transcript: [{ ...fixture.transcript[0], end_sec: 9999 }] }), /超出/)
assert.throws(() => reviewMediaUrl('https://other.test/video.mp4'), /同源/)
assert.throws(() => reviewMediaUrl('/api/review/media/../secrets'), /同源/)
const markdown = buildReviewMarkdown(parsed)
assert.match(markdown, /fixture-hash-not-real/)
assert.match(markdown, /夹具不是实际课堂结果/)
assert.match(markdown, /00:03–00:08/)
assert.match(markdown, /06:40–06:48/)
assert.match(markdown, /contract test only/)

const goodAnswer = { answer: '合同响应，仅用于测试。', source: 'local_llm', model: 'test-fixture-model', citations: [{ ...fixture.transcript[0], kind: 'transcript' }], evidence_ids: ['T0001'], request_id: 'fixture-request' }
assert.equal(parseChatResponse(goodAnswer, parsed).source, 'local_llm')
assert.throws(() => parseChatResponse({ ...goodAnswer, source: 'local_template' }, parsed), /真实模型/)
assert.throws(() => parseChatResponse({ ...goodAnswer, evidence_ids: ['OTHER_LESSON'] }, parsed), /非本课/)
assert.throws(() => parseChatResponse({ ...goodAnswer, citations: [{ ...goodAnswer.citations[0], start_sec: 200 }] }, parsed), /范围|不符/)

const blockTail = { id: 'T0003', start_sec: 9.5, end_sec: 12.9, text: '该概念的连续解释', source: 'test_fixture' }
const block = { id: 'B0001', kind: 'transcript', start_sec: 3.21, end_sec: 12.9, text: '第一段概念A 该概念的连续解释', source_refs: ['T0001', 'T0003'] }
const blockFixture = { ...fixture, transcript: [fixture.transcript[0], blockTail, fixture.transcript[1]], evidence_blocks: [block] }
const blockSource = JSON.stringify(blockFixture)
const parsedBlocks = parseLessonPackage(blockFixture)
const teacherFixture = structuredClone(fixture)
teacherFixture.teacher_transcript = fixture.transcript.map((row, i) => ({ ...row, id: `HT${i}`, transcript_id: row.id, source_refs: [row.id], source: i ? 'machine_context' : 'teacher_revision', human_verified: !i, correction_id: i ? null : 'TC1', text: i ? row.text : '教师修订的概念甲' }))
const teacherBlock = { id: 'HB1', kind: 'transcript_teacher', text: teacherFixture.teacher_transcript[0].text, start_sec: fixture.transcript[0].start_sec, end_sec: fixture.transcript[0].end_sec, source_refs: ['T0001'], quality: { human_verified: true }, workspace_version: 1 }
teacherFixture.teacher_evidence_blocks = [teacherBlock]
assert.equal(evidenceIndex(parseLessonPackage(teacherFixture)).get('HB1')?.text, teacherBlock.text, 'QA resolves real teacher evidence')
assert.equal(parseChatResponse({ ...goodAnswer, citations: [teacherBlock], evidence_ids: ['HB1'] }, teacherFixture).citations[0].id, 'HB1')
assert.throws(() => parseChatResponse({ ...goodAnswer, citations: [{ ...teacherBlock, text: '无来源改写' }], evidence_ids: ['HB1'] }, teacherFixture), /文本或类型/)
assert.throws(() => parseLessonPackage({ ...teacherFixture, teacher_transcript: [{ ...teacherFixture.teacher_transcript[0], start_sec: 100 }] }), /教师/)
assert.equal(JSON.stringify(parsedBlocks), blockSource, 'Optional retrieval blocks preserve the original transcript and provenance')
assert.equal(parsedBlocks.transcript.length, 3, 'Retrieval blocks never replace original transcript rows')
assert.equal(evidenceIndex(parsedBlocks).get('B0001').text, block.text)
assert.equal(parseLessonPackage(fixture).evidence_blocks, undefined, 'Old v3 packages without retrieval blocks remain valid')
assert.throws(() => parseLessonPackage({ ...blockFixture, evidence_blocks: [{ ...block, source_refs: ['S0001'] }] }), /不属于原始转写/)
assert.throws(() => parseLessonPackage({ ...blockFixture, evidence_blocks: [{ ...block, source_refs: ['UNKNOWN'] }] }), /不属于原始转写/)
assert.throws(() => parseLessonPackage({ ...blockFixture, evidence_blocks: [{ ...block, source_refs: [] }] }), /缺少原始转写/)
assert.throws(() => parseLessonPackage({ ...blockFixture, evidence_blocks: [{ ...block, id: 'T0001' }] }), /重复/)
assert.throws(() => parseLessonPackage({ ...blockFixture, evidence_blocks: [{ ...block, start_sec: 2 }] }), /时间与原始转写不符/)
const blockAnswer = { ...goodAnswer, citations: [block], evidence_ids: ['B0001'] }
assert.equal(parseChatResponse(blockAnswer, parsedBlocks).citations[0].text, block.text)
assert.throws(() => parseChatResponse({ ...blockAnswer, citations: [{ ...block, text: '模型自行改写的证据' }] }, parsedBlocks), /文本或类型/)
assert.throws(() => parseChatResponse({ ...blockAnswer, citations: [{ ...block, start_sec: 3.71 }] }, parsedBlocks), /时间与原始证据不符/)
assert.throws(() => parseChatResponse({ ...blockAnswer, citations: [{ ...block, kind: 'slide' }] }, parsedBlocks), /文本或类型/)

const originalFetch = globalThis.fetch
// Machine proofreading is an additive, source-constrained default view of immutable ASR.
const proofFixture = structuredClone(fixture)
const corrected = { id: 'P0001', kind: 'transcript_proofread', transcript_id: 'T0001', start_sec: 3.21, end_sec: 8.96, original_text: fixture.transcript[0].text, text: '第一段概念甲', source_refs: ['T0001', 'S0001', 'U1_0001'], edits: [{ original: '概念A', replacement: '概念甲', evidence_ids: ['U1_0001'], reason: '测试中的同源字面依据' }], uncertainties: [], quality: { status: 'automatically_proofread_unreviewed' } }
proofFixture.transcript_proofreading = { schema_version: '1.0', status: 'complete', method: 'test_fixture', model: 'fixture-model', human_verified: false, source_asr_sha256: 'fixture-asr-hash', source_video_sha256: fixture.lesson.source_video.sha256, segments: [corrected], alternate_evidence: [{ id: 'U1_0001', kind: 'alternate_asr', original_id: 'alternative-1', variant: 'asr.unguided.json', text: '第一段概念甲', start_sec: 3, end_sec: 9 }], edit_count: 1, provenance: { fixture: true } }
proofFixture.proofread_evidence_blocks = [{ id: 'PB0001', kind: 'transcript_proofread', text: corrected.text, start_sec: corrected.start_sec, end_sec: corrected.end_sec, source_refs: ['P0001', 'T0001', 'S0001', 'U1_0001'], proofreading_ids: ['P0001'] }]
const proofOriginal = JSON.stringify(proofFixture)
const parsedProof = parseLessonPackage(proofFixture)
assert.equal(JSON.stringify(parsedProof), proofOriginal, 'Proofreading preserves raw ASR, alternative ASR and all provenance')
for (const id of ['P0001', 'PB0001', 'U1_0001']) assert.ok(evidenceIndex(parsedProof).has(id))
assert.equal(parsedProof.transcript[0].text, fixture.transcript[0].text)
assert.match(buildReviewMarkdown(parsedProof), /第一段概念甲/)
assert.match(buildReviewMarkdown(parsedProof), /原始转写/)
const proofAnswer = { ...goodAnswer, evidence_ids: ['PB0001'], citations: [proofFixture.proofread_evidence_blocks[0]] }
assert.equal(parseChatResponse(proofAnswer, parsedProof).citations[0].kind, 'transcript_proofread')
for (const mutation of [
  f => { f.transcript_proofreading.human_verified = true },
  f => { f.transcript_proofreading.source_video_sha256 = 'other-source' },
  f => { f.transcript_proofreading.segments[0].original_text = '覆盖原文' },
  f => { f.transcript_proofreading.segments[0].start_sec = 4 },
  f => { f.transcript_proofreading.segments[0].text = '未记录的整句改写' },
  f => { f.transcript_proofreading.segments[0].source_refs = ['OTHER'] },
  f => { f.transcript_proofreading.segments[0].edits[0].evidence_ids = ['OTHER'] },
  f => { f.proofread_evidence_blocks[0].proofreading_ids = ['OTHER'] },
  f => { f.proofread_evidence_blocks[0].text = '未记录的检索段改写' },
]) { const bad = structuredClone(proofFixture); mutation(bad); assert.throws(() => parseLessonPackage(bad), /校对/) }
assert.throws(() => parseChatResponse({ ...proofAnswer, citations: [{ ...proofAnswer.citations[0], text: '自行改写' }] }, parsedProof), /文本或类型/)
assert.throws(() => parseChatResponse({ ...proofAnswer, citations: [{ ...proofAnswer.citations[0], kind: 'transcript' }] }, parsedProof), /文本或类型/)
assert.throws(() => parseChatResponse({ ...proofAnswer, citations: [{ ...proofAnswer.citations[0], end_sec: 9 }] }, parsedProof), /时间/)
// A new recognizer has its own segmentation. Baseline IDs remain local even on collision.
const upgradeFixture = structuredClone(proofFixture)
upgradeFixture.lesson.provenance.asr = { audio_sha256: 'same-audio-hash', source_asr_sha256: 'fixture-asr-hash', config: { model_id: 'Qwen/Qwen3-ASR-1.7B' } }
upgradeFixture.transcript_proofreading.model = 'classroom-corrector-qwen35'
upgradeFixture.transcript_proofreading.provenance.model_label = 'Qwen3.5-4B'
upgradeFixture.transcript_proofreading.provenance.model_sha256 = 'fixture-qwen35'
upgradeFixture.transcript_proofreading.segments[0].edits[0].origin = 'model'
upgradeFixture.model_upgrade_experiment = {
  schema_version: '1.0', source_video_sha256: fixture.lesson.source_video.sha256, source_audio_sha256: 'same-audio-hash', evaluated_at: '2026-10-28T00:00:00Z', reference_status: 'no_human_verbatim_reference',
  asr_runs: [
    { id: 'baseline', model_label: 'faster-whisper-small', source_asr_sha256: 'old-raw-asr', segments: [{ id: 'T0001', start_sec: 0, end_sec: 4, text: '独立旧分段一' }, { id: 'OLD2', start_sec: 4, end_sec: 10, text: '独立旧分段二' }] },
    { id: 'qwen3_asr', model_label: 'Qwen3-ASR-1.7B', source_asr_sha256: 'fixture-asr-hash', segments: structuredClone(fixture.transcript), runtime: { custom_future_field: 'retained' } },
  ],
  correction: { model_label: 'Qwen3.5-4B', model_sha256: 'fixture-qwen35', source_asr_sha256: 'fixture-asr-hash', status: 'complete', accepted_edit_count: 1, model_accepted_edit_count: 1, source_rule_accepted_count: 0, segments: upgradeFixture.transcript_proofreading.segments.map(({ transcript_id, start_sec, end_sec, original_text, text, uncertainties }) => ({ transcript_id, start_sec, end_sec, original_text, text, uncertainties })) },
  external_summary: { unknown_future_field: ['must', 'roundtrip'] }, limitations: ['仅为合同夹具，非实际模型结果'], extra_provenance: { retained: true },
}
const upgradeBefore = JSON.stringify(upgradeFixture)
const parsedUpgrade = parseLessonPackage(upgradeFixture)
assert.equal(JSON.stringify(parsedUpgrade), upgradeBefore, 'All opaque experiment metadata roundtrips without rewriting raw evidence')
assert.equal(evidenceIndex(parsedUpgrade).get('T0001').text, fixture.transcript[0].text, 'Run-local collision cannot replace current raw evidence')
assert.equal(evidenceIndex(parsedUpgrade).has('OLD2'), false, 'Baseline-only IDs never become graph or QA evidence')
assert.equal(evidenceIndex(parsedUpgrade).size, evidenceIndex(parsedProof).size)
assert.throws(() => parseChatResponse({ ...goodAnswer, evidence_ids: ['OLD2'], citations: [] }, parsedUpgrade), /非本课/)
assert.deepEqual(modelUpgradeLanes(parsedUpgrade.model_upgrade_experiment, 4).map(lane => lane.segments.map(row => row.id)), [['OLD2'], ['T0001'], ['T0001']], 'Each lane independently uses its own half-open native clock')
assert.deepEqual(modelUpgradeLanes(parsedUpgrade.model_upgrade_experiment, 9).map(lane => lane.segments.length), [1, 0, 0], 'A baseline interval cannot fill a new recognition gap')
assert.deepEqual(modelUpgradeLanes(parsedUpgrade.model_upgrade_experiment, 9, true).map(lane => lane.segments.length), [2, 2, 1])
assert.equal(modelUpgradeTime(3.215), '3.215 秒', 'Native boundary precision must not be floored to whole seconds')
for (const mutate of [
  f => { f.model_upgrade_experiment.source_video_sha256 = 'other' },
  f => { f.model_upgrade_experiment.source_audio_sha256 = 'other' },
  f => { delete f.lesson.provenance.asr.audio_sha256 },
  f => { f.model_upgrade_experiment.reference_status = 'human_verified' },
  f => { f.model_upgrade_experiment.evaluated_at = 'invalid' },
  f => { f.model_upgrade_experiment.asr_runs[0].segments[0].start_sec = NaN },
  f => { f.model_upgrade_experiment.asr_runs[0].segments[0].end_sec = fixture.lesson.duration_sec + .01 },
  f => { f.model_upgrade_experiment.asr_runs[0].segments[0].text = '  ' },
  f => { f.model_upgrade_experiment.asr_runs[0].segments.push(f.model_upgrade_experiment.asr_runs[0].segments[0]) },
  f => { f.model_upgrade_experiment.asr_runs.push(f.model_upgrade_experiment.asr_runs[0]) },
  f => { f.model_upgrade_experiment.asr_runs.pop() },
  f => { f.model_upgrade_experiment.asr_runs[1].segments.pop() },
  f => { f.model_upgrade_experiment.asr_runs[1].segments[0].start_sec += .01 },
  f => { f.model_upgrade_experiment.asr_runs[1].segments[0].text = '并非当前原文' },
  f => { f.model_upgrade_experiment.asr_runs[1].source_asr_sha256 = 'other' },
  f => { f.model_upgrade_experiment.correction.source_asr_sha256 = 'other' },
  f => { f.model_upgrade_experiment.correction.model_label = 'other' },
  f => { f.model_upgrade_experiment.correction.model_sha256 = 'other' },
  f => { f.model_upgrade_experiment.correction.status = 'partial' },
  f => { f.model_upgrade_experiment.correction.accepted_edit_count = -.1 },
  f => { f.model_upgrade_experiment.correction.accepted_edit_count = 2 },
  f => { f.model_upgrade_experiment.correction.model_accepted_edit_count = 0 },
  f => { f.model_upgrade_experiment.correction.source_rule_accepted_count = 1 },
  f => { f.model_upgrade_experiment.correction.segments[0].text = '并非实际校对结果' },
  f => { f.model_upgrade_experiment.correction.segments[0].uncertainties = ['未记录'] },
  f => { f.model_upgrade_experiment.correction.segments.push(f.model_upgrade_experiment.correction.segments[0]) },
  f => { f.model_upgrade_experiment.correction.segments.pop() },
]) { const bad = structuredClone(upgradeFixture); mutate(bad); assert.throws(() => parseLessonPackage(bad), /新模型试验/) }
for (const label of ['', ' ', {}, 'invalid\nlabel', 'x'.repeat(161)]) {
  const bad = structuredClone(proofFixture); bad.transcript_proofreading.provenance.model_label = label
  assert.throws(() => parseLessonPackage(bad), /校对模型显示标签/)
}
// Comparison records share immutable TIDs, but never replace them in evidence lookup.
const comparisonFixture = structuredClone(fixture)
comparisonFixture.correction_comparison = { schema_version: '1.0', source_video_sha256: fixture.lesson.source_video.sha256, evaluated_at: '2026-10-28T00:00:00Z', scope: 'same_lesson_controlled_model_comparison', reference_status: 'no_human_verbatim_reference', limitations: ['仅为合同测试夹具'], runs: ['qwen', 'cec4'].map(id => ({ id, model_label: `${id}-fixture-only`, model_sha256: `${id}-hash`, accepted_edit_count: 0, guard_rejected_count: null, memory: { unit: 'MiB', scope: 'whole_gpu_sampled', baseline_mib: 0, loaded_mib: 12.5, peak_mib: null, sample_interval_ms: 100 }, segments: [{ transcript_id: 'T0001', start_sec: 3.21, end_sec: 8.96, original_text: fixture.transcript[0].text, text: `${id}模型对比文本`, uncertainties: [] }] })) }
const comparisonOriginal = JSON.stringify(comparisonFixture)
const parsedComparison = parseLessonPackage(comparisonFixture)
assert.equal(JSON.stringify(parsedComparison), comparisonOriginal, 'Comparison JSON, missing counts, unknown memory and extra provenance stay intact')
assert.equal(evidenceIndex(parsedComparison).get('T0001').text, fixture.transcript[0].text, 'Nested duplicate TIDs cannot replace original evidence')
assert.equal(evidenceIndex(parsedComparison).size, evidenceIndex(parsed).size, 'Comparison outputs never add QA/graph evidence')
const nativeComparison = structuredClone(comparisonFixture)
for (const run of nativeComparison.correction_comparison.runs) {
  run.model_accepted_edit_count = 0; run.source_rule_accepted_count = 0
  run.additional_changed_spans_vs_rules = run.id === 'qwen' ? 1 : null
  run.status = 'partial'; run.native_status = 'complete'
  run.native_segments = run.segments.map(row => ({ ...row, text: '未经校验的原生假设' }))
}
const nativeOriginal = JSON.stringify(nativeComparison)
assert.equal(JSON.stringify(parseLessonPackage(nativeComparison)), nativeOriginal, 'Native hypotheses and explicit contribution/status metadata are preserved')
assert.equal(evidenceIndex(parseLessonPackage(nativeComparison)).get('T0001').text, fixture.transcript[0].text)
for (const mutate of [
  f => { f.correction_comparison.runs[0].model_accepted_edit_count = -1 },
  f => { f.correction_comparison.runs[0].additional_changed_spans_vs_rules = -1 },
  f => { f.correction_comparison.runs[0].additional_changed_spans_vs_rules = 1.5 },
  f => { f.correction_comparison.runs[0].source_rule_accepted_count = 1.5 },
  f => { f.correction_comparison.runs[0].native_segments[0].original_text = '不同原文' },
  f => { f.correction_comparison.runs[0].native_segments[0].start_sec = 0 },
  f => { f.correction_comparison.runs[0].native_segments[0].transcript_id = 'OTHER' },
  f => { f.correction_comparison.runs[0].native_segments.push(f.correction_comparison.runs[0].native_segments[0]) },
]) { const bad = structuredClone(nativeComparison); mutate(bad); assert.throws(() => parseLessonPackage(bad), /校对模型对比/) }
assert.throws(() => parseChatResponse({ ...goodAnswer, evidence_ids: ['cec4'], citations: [] }, parsedComparison), /非本课/)
for (const mutate of [
  f => { f.correction_comparison.source_video_sha256 = 'other-video' },
  f => { f.correction_comparison.evaluated_at = 'missing-time' },
  f => { f.correction_comparison.reference_status = 'human_verified' },
  f => { f.correction_comparison.runs.push(f.correction_comparison.runs[0]) },
  f => { f.correction_comparison.runs[0].segments.push(f.correction_comparison.runs[0].segments[0]) },
  f => { f.correction_comparison.runs[0].segments[0].transcript_id = 'OTHER' },
  f => { f.correction_comparison.runs[0].segments[0].original_text = '覆盖原文' },
  f => { f.correction_comparison.runs[0].segments[0].start_sec = 3.3 },
  f => { f.correction_comparison.runs[0].segments[0].end_sec = 9 },
  f => { f.correction_comparison.runs[0].accepted_edit_count = -1 },
  f => { f.correction_comparison.runs[0].memory.scope = 'file_weight_proxy' },
  f => { f.correction_comparison.runs[0].memory.peak_mib = -1 },
]) { const bad = structuredClone(comparisonFixture); mutate(bad); assert.throws(() => parseLessonPackage(bad), /校对模型对比/) }
// Independent courseware is an observed overlay, never a replacement timeline.
const courseFixture = structuredClone(fixture)
courseFixture.transcript[0].text = '鄭賢定理'
courseFixture.courseware = { document_id: 'Dtest', filename: 'synthetic.pdf', sha256: 'test-only', format: 'pdf', source_url: '/api/review/media/contract-fixture/source.pdf', warnings: [], provenance: { test: true }, pages: [{ id: 'DtestP1', page_number: 1, text: '正弦定理', image_url: '/api/review/media/contract-fixture/page.png', text_source: 'synthetic' }], matches: [{ id: 'Mtest1', slide_id: 'S0001', page_id: 'DtestP1', start_sec: 0, end_sec: 10, status: 'confirmed', score: .9, margin: .3, method: 'synthetic' }] }
courseFixture.courseware_evidence = [{ id: 'Ctest1', kind: 'courseware', page_id: 'DtestP1', match_id: 'Mtest1', start_sec: 0, end_sec: 10, text: '正弦定理' }]
courseFixture.transcript_revisions = [{ id: 'Rtest1', transcript_id: 'T0001', original_text: '鄭賢定理', corrected_text: '正弦定理', start_sec: 3.21, end_sec: 8.96, status: 'suggested', edits: [{ start_char: 0, end_char: 4, original: '鄭賢定理', replacement: '正弦定理', page_ids: ['DtestP1'], match_ids: ['Mtest1'], method: 'synthetic' }] }]
courseFixture.revised_evidence_blocks = [{ id: 'BRtest1', kind: 'transcript_revision', start_sec: 3.21, end_sec: 8.96, text: '正弦定理', source_refs: ['T0001'], revision_ids: ['Rtest1'] }]
assert.equal(parseLessonPackage(courseFixture).transcript[0].text, '鄭賢定理', 'Native ASR remains unchanged')
assert.equal(evidenceIndex(courseFixture).get('Ctest1').start_sec, 0)
assert.match(buildReviewMarkdown(courseFixture), /鄭賢定理/)
assert.match(buildReviewMarkdown(courseFixture), /synthetic.pdf/)
const courseAnswer = { ...goodAnswer, citations: [courseFixture.revised_evidence_blocks[0]], evidence_ids: ['BRtest1'] }
assert.equal(parseChatResponse(courseAnswer, courseFixture).citations[0].text, '正弦定理')
assert.throws(() => parseChatResponse({ ...courseAnswer, citations: [{ ...courseAnswer.citations[0], text: '伪造修订内容' }] }, courseFixture), /文本或类型/)
for (const mutate of [
  f => { f.courseware.matches[0].start_sec = 2 },
  f => { f.courseware.matches[0].page_id = 'OTHER' },
  f => { f.courseware.matches[0].status = 'ambiguous' },
  f => { f.transcript_revisions[0].original_text = '已覆盖的原文' },
  f => { f.transcript_revisions[0].edits[0].replacement = '不存在的术语' },
  f => { f.revised_evidence_blocks[0].text = '未经记录的改写' },
  f => { f.revised_evidence_blocks[0].source_refs = ['S0001'] },
]) { const bad = structuredClone(courseFixture); mutate(bad); assert.throws(() => parseLessonPackage(bad), /课件证据无效/) }
writeFileSync(join(cache, 'courseware-fixture.json'), JSON.stringify(courseFixture))
globalThis.window = { location: { search: '' } }
let requests = [], response = fixture, status = 200
globalThis.fetch = async (url, init) => { requests.push({ url, init }); return new Response(JSON.stringify(response), { status, headers: { 'Content-Type': 'application/json' } }) }
try {
  for (const code of ['workspace_changed', 'workspace_source_changed']) {
    response = { error: { code, message: '记录发生并发变化' } }; status = 409
    const draftInput = { text: '尚未保存的教师输入', expected_version: 3 }
    await assert.rejects(() => reviewService.saveWorkspace(fixture.lesson.id, '/goals', draftInput), error => error instanceof ReviewServiceError && error.status === 409 && error.code === code)
    assert.deepEqual(draftInput, { text: '尚未保存的教师输入', expected_version: 3 }, 'Version conflicts must not mutate caller input')
    assert.deepEqual(JSON.parse(requests.at(-1).init.body), draftInput)
  }
  response = { error: { code: 'MODEL_UNAVAILABLE', message: '测试模型服务不可用' } }; status = 503
  await assert.rejects(() => reviewService.chat({ lesson_id: fixture.lesson.id, query: '测试问题', at_sec: 3.21, scope: 'current' }, parsed), /测试模型服务不可用/)
  setActivePinia(createPinia())
  const store = useLessonReviewStore()
  response = { lessons: [] }; status = 200
  await store.loadLessons()
  assert.equal(store.data, null)
  assert.equal(store.loading, false)
  assert.equal(store.duration, 0)
  response = fixture
  await store.selectLesson(fixture.lesson.id)
  assert.equal(store.duration, 1144.4333, 'Lesson time does not depend on snapshots')
  store.seek(200)
  assert.equal(store.currentTime, 200)
  assert.equal(store.currentConceptIds.length, 0)
  store.selectNode('C1')
  assert.equal(store.currentTime, 3.21)
  store.jumpToEvidence('T0002')
  assert.equal(store.currentTime, 400.35)
  assert.equal(store.currentSlide.id, 'S0002')
  assert.equal(store.transcripts[0].id, 'T0002')
  store.scope = 'current'
  response = { error: { code: 'MODEL_UNAVAILABLE', message: '测试模型服务不可用' } }; status = 503
  await store.ask('测试问题')
  assert.equal(store.messages[0].status, 'error')
  assert.equal(store.messages[0].response, undefined, 'Failure must not fabricate any model answer')
  assert.equal(store.isThinking, false)
  assert.deepEqual(JSON.parse(requests.at(-1).init.body), { lesson_id: fixture.lesson.id, query: '测试问题', at_sec: 400.35, scope: 'current' })
  response = goodAnswer; status = 200
  await store.retry(store.messages[0].id)
  assert.equal(store.messages.length, 1, 'Retry reuses the failed query')
  assert.equal(store.messages[0].status, 'answered')
  assert.equal(store.messages[0].response.model, 'test-fixture-model')
  response = fixture
  const beforeRefreshMessages = store.messages, beforeRefreshSeek = store.seekVersion
  store.isPlaying = true
  await store.selectLesson(fixture.lesson.id)
  assert.equal(store.currentTime, 400.35, 'Same-recording refresh preserves media time')
  assert.equal(store.isPlaying, true, 'Same-recording refresh preserves playback state')
  assert.equal(store.seekVersion, beforeRefreshSeek, 'Refreshing source records never seeks the stable player')
  assert.equal(store.messages, beforeRefreshMessages, 'Same-recording refresh retains conversation')
  response = { ...goodAnswer, evidence_ids: ['OTHER_LESSON'] }
  await store.ask('错误引用')
  assert.equal(store.messages[1].status, 'error')
  assert.equal(store.messages[1].response, undefined)
  response = { error: { message: '课程服务故障' } }; status = 503
  await store.selectLesson('missing')
  assert.equal(store.data.lesson.id, fixture.lesson.id, 'A failed replacement retains the current source package')
  assert.equal(store.error, '课程服务故障')
  response = blockFixture; status = 200
  await store.selectLesson(fixture.lesson.id)
  store.seek(200)
  store.jumpToEvidence('B0001')
  assert.equal(store.currentTime, 3.21, 'Retrieval block citations seek the original media time')
  assert.equal(store.transcripts[0].id, 'T0001', 'Block seek activates the original transcript row')
  assert.equal(store.currentSlide.id, 'S0001', 'Block seek activates the original slide')
  response = blockAnswer
  await store.ask('带连续证据段的问题')
  assert.equal(store.messages[0].status, 'answered')
  assert.equal(store.messages.at(-1).response.citations[0].id, 'B0001')

  // A history transition may return to the still-visible lesson before another lesson finishes.
  // Control the actual service response so both a late success and a late failure are exercised.
  const otherLesson = structuredClone(fixture)
  otherLesson.lesson.id = 'history-race-other'
  otherLesson.lesson.video_url = '/api/review/media/history-race-other/source.mp4'
  otherLesson.lesson.source_video.sha256 = 'other-recording'
  store.lessons = [fixture.lesson, otherLesson.lesson]
  for (const outcome of ['success', 'failure']) {
    let release
    let fetchCount = 0
    globalThis.fetch = async () => {
      fetchCount++
      return new Promise(resolve => { release = resolve })
    }
    store.seek(17.5); store.isPlaying = true
    const conversation = store.messages, beforeRouteSeek = store.seekVersion
    window.location.search = '?view=report&lesson=history-race-other'
    const backToOther = store.restoreRoute()
    assert.equal(store.loading, true)
    assert.equal(store.data.lesson.id, fixture.lesson.id, 'The current recording stays visible until the replacement arrives')
    window.location.search = `?view=behavior&lesson=${fixture.lesson.id}`
    await store.restoreRoute()
    assert.equal(store.loading, false, 'Forward to the visible lesson finishes loading immediately and invalidates the pending replacement')
    assert.equal(store.activeView, 'behavior')
    release(new Response(JSON.stringify(outcome === 'success' ? otherLesson : { error: { message: 'late obsolete failure' } }), { status: outcome === 'success' ? 200 : 503, headers: { 'Content-Type': 'application/json' } }))
    await backToOther
    assert.equal(store.data.lesson.id, fixture.lesson.id, 'A late history response cannot overwrite the lesson named by the current URL')
    assert.equal(store.loading, false)
    assert.equal(store.error, '', 'A stale failure cannot replace the current route with an error')
    assert.equal(store.currentTime, 17.5)
    assert.equal(store.isPlaying, true)
    assert.equal(store.seekVersion, beforeRouteSeek)
    assert.equal(store.messages, conversation)
    assert.equal(fetchCount, 1, 'Returning to the visible recording requires no redundant request')
  }
} finally { globalThis.fetch = originalFetch }

console.log('PASS: v3 contract, optional retrieval blocks/native transcript provenance, exact time, empty catalog, disjoint evidence, citation text/time validation, model errors/retry, no fabricated answers.')
