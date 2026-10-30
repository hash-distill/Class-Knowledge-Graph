import type { ActivityFrame, ActivityParticipant, LessonPackage } from '../types/lessonReview'
import { engagementExport } from './classEngagement'

// Activity is an optional observed overlay of the original recording clock.
export function validateActivityAnalysis(data: any, media: (url: unknown) => string) {
  if (data.activity_analysis === undefined) return
  const fail = (field: string): never => { throw new Error(`行为分析数据无效：${field}`) }
  const obj = (x: any, key: string): Record<string, any> => x && typeof x === 'object' && !Array.isArray(x) ? x : fail(key)
  const array = (x: any, key: string): any[] => Array.isArray(x) ? x : fail(key)
  const text = (x: any, key: string): string => typeof x === 'string' && x.trim() ? x : fail(key)
  const number = (x: any, key: string, max = Infinity): number => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= max ? x : fail(key)
  const count = (x: any, key: string): number => Number.isInteger(number(x, key)) ? x : fail(key)
  const duration = data.lesson.duration_sec
  const timed = (x: any, field: string) => {
    obj(x, field); text(x.id, `${field}.id`)
    number(x.start_sec, `${field}.start_sec`, duration); number(x.end_sec, `${field}.end_sec`, duration)
    if (x.end_sec <= x.start_sec) fail(`${field} 时间区间为空或倒置`)
  }
  const activity = obj(data.activity_analysis, 'activity_analysis')
  if (activity.schema_version !== '1.0' || activity.status !== 'complete') fail('仅接受已完整发布的 1.0 分析')
  if (text(activity.source_sha256, 'source_sha256') !== data.lesson.source_video.sha256) fail('来源哈希与原录像不同源')
  if (!number(activity.sampling_interval_sec, 'sampling_interval_sec')) fail('采样间隔必须大于零')
  const knownIds = new Set<string>([
    ...data.transcript, ...data.slides, ...data.alignment, ...(data.evidence_blocks || []), ...(data.courseware_evidence || []),
    ...(data.transcript_revisions || []), ...(data.revised_evidence_blocks || []).filter((row: any) => row.kind !== 'transcript'),
    ...(data.transcript_proofreading?.segments || []), ...(data.transcript_proofreading?.alternate_evidence || []), ...(data.proofread_evidence_blocks || []),
    ...(data.courseware?.pages || []), ...(data.courseware?.matches || []),
  ].map((row: any) => row.id))
  const unique = (row: any) => { if (knownIds.has(row.id)) fail(`重复证据 ID ${row.id}`); knownIds.add(row.id) }
  const contexts = array(activity.context_segments, 'context_segments')
  for (const row of contexts) {
    timed(row, 'context_segment'); unique(row); text(row.text, 'context.text'); text(row.model_id, 'context.model_id')
    if (row.method !== 'trained_context') fail('情境必须声明 trained_context 来源')
    const labels = new Set<string>()
    for (const label of array(row.labels, 'context.labels')) {
      obj(label, 'context.label'); text(label.label, 'context.label'); text(label.name, 'context.name'); number(label.probability, 'context.probability', 1)
      if (labels.has(label.label)) fail('情境标签重复'); labels.add(label.label)
    }
  }
  const frames = array(activity.frames, 'frames')
  let previous: any = null
  const sceneIds = new Set<number>()
  for (const frame of frames) {
    timed(frame, 'frame'); unique(frame)
    number(frame.timestamp_sec, 'frame.timestamp_sec', duration)
    if (frame.timestamp_sec < frame.start_sec || frame.timestamp_sec >= frame.end_sec) fail('帧时间戳必须位于证据区间内')
    if (frame.end_sec - frame.start_sec > activity.sampling_interval_sec + 1e-6) fail('帧区间越过采样间隔')
    if (!count(frame.scene_id, 'frame.scene_id') || typeof frame.scene_cut !== 'boolean') fail('镜头标记')
    if (previous && (frame.start_sec < previous.end_sec || frame.scene_id < previous.scene_id || (frame.scene_id !== previous.scene_id) !== frame.scene_cut)) fail('帧重叠或镜头边界不一致')
    sceneIds.add(frame.scene_id)
    text(frame.status, 'frame.status')
    if (typeof frame.image_url !== 'string') fail('frame.image_url')
    if (frame.image_url) {
      media(frame.image_url)
      let decoded: string
      try { decoded = decodeURIComponent(frame.image_url) } catch { fail('帧媒体编码无效') }
      const prefix = `/api/review/media/${encodeURIComponent(data.lesson.id)}/`
      if (!frame.image_url.startsWith(prefix) || /(?:^|\/)\.\.(?:\/|$)|[\\?#]/.test(decoded!)) fail('帧媒体必须属于本课同源目录')
    }
    count(frame.detected_people, 'frame.detected_people'); count(frame.known_people, 'frame.known_people')
    if (frame.known_people > frame.detected_people) fail('规则分数可用对象数不能超过检测数')
    if (frame.coverage !== null) {
      number(frame.coverage, 'frame.coverage', 1)
      const ratio = frame.detected_people ? frame.known_people / frame.detected_people : 0
      if (Math.abs(frame.coverage - ratio) > .0001) fail('覆盖率与观测人数不符')
    }
    if (frame.participation_index !== null) number(frame.participation_index, 'frame.participation_index', 100)
    if (!frame.known_people && frame.participation_index !== null) fail('无规则分数可用对象时规则指数必须为 null')
    for (const [label, value] of Object.entries(obj(frame.behaviors, 'frame.behaviors'))) { text(label, 'behavior.label'); count(value, `behavior.${label}`) }
    const tracks = new Set<string>()
    const participants = array(frame.participants, 'frame.participants')
    if (frame.width !== undefined && !number(frame.width, 'frame.width')) fail('帧宽度必须大于零')
    if (frame.height !== undefined && !number(frame.height, 'frame.height')) fail('帧高度必须大于零')
    if (participants.length > frame.detected_people) fail('对象列表超过检测数')
    for (const participant of participants) {
      obj(participant, 'participant'); text(participant.track_id, 'participant.track_id')
      if (!participant.track_id.startsWith(`S${frame.scene_id}-T`) || tracks.has(participant.track_id)) fail('对象轨迹必须在镜头内唯一，不能跨镜头关联')
      tracks.add(participant.track_id)
      const box = array(participant.bbox, 'participant.bbox')
      if (box.length !== 4) fail('对象框必须有四个坐标')
      box.forEach(value => number(value, 'participant.bbox'))
      if (box[2] <= box[0] || box[3] <= box[1]) fail('对象框范围无效')
      if ((frame.width && box[2] > frame.width + 1) || (frame.height && box[3] > frame.height + 1)) fail('对象框超出原始帧范围')
      text(participant.label, 'participant.label'); text(participant.source, 'participant.source'); number(participant.confidence, 'participant.confidence', 1)
      if (participant.score !== null) number(participant.score, 'participant.score', 1)
      if (participant.student_state !== undefined) validateStudentState(participant.student_state, obj, text, number, array, fail)
      if (participant.posture_tags !== undefined) for (const state of array(participant.posture_tags, 'posture_tags')) validateStudentState(state, obj, text, number, array, fail)
      if (participant.keypoints !== undefined) {
        const points = array(participant.keypoints, 'keypoints')
        if (points.length !== 0 && points.length !== 17) fail('姿态关键点必须为空或17个')
        for (const point of points) { if (array(point, 'keypoint').length !== 3) fail('关键点必须包含坐标和置信度'); number(point[0], 'keypoint.x'); number(point[1], 'keypoint.y'); number(point[2], 'keypoint.confidence', 1) }
      }
    }
    if (frame.student_state_counts !== undefined || frame.participation !== undefined) {
      const states = obj(frame.student_state_counts, 'student_state_counts')
      const expectedStates: Record<string, number> = {}
      for (const participant of participants) {
        const state = participant.student_state
        if (!state) fail('新增状态分布缺少逐对象状态')
        expectedStates[state.label] = (expectedStates[state.label] || 0) + 1
      }
      if (frame.detected_people > participants.length) expectedStates.unknown = (expectedStates.unknown || 0) + frame.detected_people - participants.length
      for (const key of new Set([...Object.keys(states), ...Object.keys(expectedStates)])) if (count(states[key] ?? 0, `student_state_counts.${key}`) !== (expectedStates[key] || 0)) fail('状态分布与逐对象记录不符')
      const participation = obj(frame.participation, 'participation')
      const total = frame.detected_people, active = ['read', 'write', 'raise_hand'].reduce((n, key) => n + (states[key] || 0), 0), observed = total - (states.unknown || 0)
      for (const [key, expected] of Object.entries({ total_people: total, observed_people: observed, unknown_people: total - observed, active_people: active, scored_people: active })) if (count(participation[key], `participation.${key}`) !== expected) fail(`participation.${key} 与状态记录不符`)
      for (const [key, numerator] of Object.entries({ coverage: observed, active_fraction: active, score_coverage: active })) {
        if (!total) { if (participation[key] !== null) fail(`无对象时 ${key} 必须未知`) }
        else if (Math.abs(number(participation[key], `participation.${key}`, 1) - numerator / total) > .0001) fail(`participation.${key} 与状态记录不符`)
      }
      const weights = obj(participation.weights, 'participation.weights')
      for (const [key, value] of Object.entries(weights)) { if (!['read', 'write', 'raise_hand'].includes(key)) fail('不能给姿态线索赋积极度分'); number(value, `participation.weights.${key}`, 1) }
      if (!active) { if (participation.index !== null) fail('没有明确参与动作时参考积极度必须未知') }
      else {
        const expected = ['read', 'write', 'raise_hand'].reduce((n, key) => n + (states[key] || 0) * number(weights[key], `participation.weights.${key}`, 1), 0) / active * 100
        if (Math.abs(number(participation.index, 'participation.index', 100) - expected) > .05) fail('参考积极度与动作规则不符')
      }
      if (!['observed', 'low_evidence', 'no_people'].includes(participation.status) || participation.semantics !== 'visible_participation_rule_v2' || participation.population_scope !== 'visible_people_roles_unverified') fail('可见参与来源或解释范围无效')
    }
    if (frame.visual_reference !== undefined) validateVisualReference(frame, previous, fail)
    if (frame.detection_review !== undefined) {
      const review = obj(frame.detection_review, 'detection_review')
      if (!['full_plus_tiles_pose_behavior_fusion', 'full_plus_tiles_person_pose_then_crop_classification'].includes(review.method) || review.population_recall_evaluated !== false) fail('切片检测方法或评估范围无效')
      if (count(review.retained_people, 'retained_people') !== frame.detected_people) fail('切片保留框与当前检测数不符')
      if (count(review.proposal_count, 'proposal_count') < frame.detected_people) fail('检测候选少于保留对象')
      count(review.excluded_detection_proposals, 'excluded_detection_proposals'); count(review.tile_windows, 'tile_windows'); number(review.wall_seconds, 'wall_seconds')
      if (review.previous_detected_people !== undefined) count(review.previous_detected_people, 'previous_detected_people')
    }
    previous = frame
  }
  const summary = obj(activity.summary, 'summary')
  const expected = { sampled_frames: frames.length, scene_count: sceneIds.size, person_observations: frames.reduce((n, f) => n + f.detected_people, 0), known_observations: frames.reduce((n, f) => n + f.known_people, 0) }
  for (const [key, value] of Object.entries(expected)) if (count(summary[key], `summary.${key}`) !== value) fail(`summary.${key} 与帧观测不一致`)
  obj(activity.models, 'models')
  array(activity.limitations, 'limitations').forEach(line => text(line, 'limitation'))
  for (const row of array(activity.evidence, 'evidence')) {
    timed(row, 'evidence'); unique(row); text(row.text, 'evidence.text')
    if (row.kind !== 'behavior') fail('行为证据类型必须为 behavior')
    // An evidence interval must be covered by actual sampled frames or context records, with no unknown gap.
    const supports = [...frames, ...contexts].filter(s => s.start_sec < row.end_sec && s.end_sec > row.start_sec).sort((a, b) => a.start_sec - b.start_sec)
    let covered = row.start_sec
    for (const support of supports) { if (support.start_sec > covered + 1e-6) break; covered = Math.max(covered, support.end_sec) }
    if (covered < row.end_sec - 1e-6) fail(`行为证据 ${row.id} 超出已观测区间`)
  }
}

function validateStudentState(state: any, obj: any, text: any, number: any, array: any, fail: (field: string) => never) {
  obj(state, 'student_state'); text(state.name, 'student_state.name')
  if (!['read', 'write', 'raise_hand', 'head_up', 'head_lowered', 'hand_near_head', 'standing', 'unknown', 'lookup', 'turn_head', 'stand', 'discuss'].includes(state.label)) fail('未知的学生状态标签')
  if (['lookup', 'turn_head', 'stand', 'discuss'].includes(state.label) && !['pretrained_detector', 'trained_classifier'].includes(state.source)) fail('七类模型标签不能由姿态规则冒充')
  if (!['pretrained_detector', 'trained_classifier', 'pose_geometry_rule', 'unavailable'].includes(state.source)) fail('学生状态来源无效')
  if (state.confidence !== null) number(state.confidence, 'student_state.confidence', 1)
  array(state.evidence, 'student_state.evidence').forEach((value: unknown) => text(value, 'student_state.evidence'))
  array(state.keypoint_indices, 'student_state.keypoint_indices').forEach((value: unknown) => { if (!Number.isInteger(number(value, 'keypoint_index', 16))) fail('关键点索引必须为整数') })
}

function validateVisualReference(frame: ActivityFrame, previous: ActivityFrame | undefined, fail: (field: string) => never) {
  const ref = frame.visual_reference!
  const weights: Record<string, number> = { read: .8, write: .85, lookup: .85, raise_hand: .95, discuss: .75 }
  if (!ref || typeof ref !== 'object' || ref.semantics !== 'configured_visible_behavior_reference_v1' || ref.calibrated_engagement_probability !== false || ref.population_scope !== 'directly_detected_people_roles_unverified' || ref.weight_source !== 'legacy_project_action_policy_subset') fail('视觉规则参考的来源或解释范围无效')
  if (!ref.weights || Object.keys(ref.weights).length !== Object.keys(weights).length || Object.entries(weights).some(([key, value]) => ref.weights[key] !== value)) fail('视觉规则参考权重与声明不符')
  const scored = frame.participants.filter(person => ['pretrained_detector', 'trained_classifier'].includes(person.source) && person.student_state?.source === person.source && person.label === person.student_state.label && Object.prototype.hasOwnProperty.call(weights, person.label))
  const total = frame.detected_people, support = total ? scored.length / total : null
  const enough = total >= 3 && support !== null && support >= .6 && scored.length > 0
  const index = enough ? scored.reduce((sum, row) => sum + weights[row.label]!, 0) / scored.length * 100 : null
  const old = previous?.visual_reference
  const smooth = index === null ? null : old?.smoothed_index !== undefined && old.smoothed_index !== null && previous!.scene_id === frame.scene_id && frame.timestamp_sec - previous!.timestamp_sec <= 1.2 ? .3 * old.smoothed_index + .7 * index : index
  const same = (value: number | null, expected: number | null) => expected === null ? value === null : typeof value === 'number' && Number.isFinite(value) && Math.abs(value - expected) <= .0002
  if (ref.scored_people !== scored.length || ref.total_people !== total || !same(ref.support_fraction, support) || ref.minimum_people !== 3 || ref.minimum_support_fraction !== .6 || !same(ref.index, index) || !same(ref.smoothed_index, smooth)) fail('视觉规则参考与有效状态支持或连续观测不符')
  if (ref.status !== (!total ? 'no_people' : enough ? 'estimated' : 'insufficient_support') || ref.smoothing?.previous !== .3 || ref.smoothing.current !== .7 || ref.smoothing.maximum_gap_sec !== 1.2) fail('视觉规则参考平滑规则无效')
}

export function participationRuns(frames: ActivityFrame[]): ActivityFrame[][] {
  const runs: ActivityFrame[][] = []
  let run: ActivityFrame[] = []
  for (const frame of frames) {
    const previous = run[run.length - 1]
    if (frame.participation_index === null || (previous && (frame.scene_id !== previous.scene_id || frame.start_sec > previous.end_sec + 1e-6))) {
      if (run.length) runs.push(run)
      run = []
    }
    if (frame.participation_index !== null) run.push(frame)
  }
  if (run.length) runs.push(run)
  return runs
}

export function sceneTrackRows(frames: ActivityFrame[], sceneId: number): { track_id: string; cells: (ActivityParticipant | null)[] }[] {
  const sceneFrames = frames.filter(frame => frame.scene_id === sceneId)
  const ids = [...new Set(sceneFrames.flatMap(frame => frame.participants.map(participant => participant.track_id)))]
  return ids.map(track_id => ({ track_id, cells: sceneFrames.map(frame => frame.participants.find(participant => participant.track_id === track_id) || null) }))
}

export function readableModelFields(models: Record<string, unknown>): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = []
  const obj = (value: unknown): Record<string, any> => value && typeof value === 'object' ? value as Record<string, any> : {}
  const add = (label: string, value: unknown) => {
    if (typeof value === 'number' && Number.isFinite(value)) rows.push({ label, value: String(value) })
  }
  const context = obj(models.context), model = obj(context.model), metrics = obj(context.metrics)
  const observation = obj(models.observation)
  if (observation.pipeline === 'dense_person_then_seven_crop_observation_v2') {
    rows.push({ label: '当前画面的行为识别', value: '整图与切片定位人物 → 姿态融合与去重 → MobileNetV3 Small 逐人七类候选；本课尚未人工标注验收' })
    const classifier = Array.isArray(observation.models) ? observation.models.find((row: any) => row.task === 'person_crop_classification') : undefined
    if (classifier) {
      rows.push({ label: '状态分类权重', value: `${classifier.architecture} · ${(classifier.weight_bytes / 1e6).toFixed(2)} MB` })
      add('给定标注裁剪框 / 测试对象数', obj(classifier.test_metrics).samples)
      add('给定标注裁剪框 / 测试 Macro-F1', obj(classifier.test_metrics).macro_f1)
      rows.push({ label: '测试适用范围', value: '测试图与训练图存在相同课堂、相邻时刻；不是独立课堂验证，也不是本课准确率或检测端到端准确率' })
      rows.push({ label: '角色与积极度', value: '未确认师生身份；分类置信度、可辨状态支持率均不是心理积极度概率' })
    }
  } else if (models.observation) rows.push({ label: '当前画面的行为识别', value: '预训练行为检测与姿态规则；尚未在本课人工标注评测' })
  if (models.context) {
    rows.push({ label: '教学情境预测', value: context.status === 'unavailable' ? '模型暂不可用' : '整段视频分类预测，与逐人行为分开解释' })
    add('情境模型 / 数据集总样本', model.dataset_count)
    add('情境模型 / 训练样本', model.training_count)
    for (const [part, label] of [['train', '训练'], ['val', '验证'], ['test', '测试']]) add(`情境模型 / ${label}样本`, obj(metrics.split_counts)[part!])
    for (const [part, label] of [['validation', '验证集'], ['test', '测试集']]) {
      add(`情境模型 / ${label}宏平均综合得分`, obj(metrics[part!]).macro_f1)
      add(`情境模型 / ${label}平均精确率均值`, obj(metrics[part!]).mean_average_precision)
    }
  }
  if (models.aric) {
    rows.push({ label: '辅助行为数据训练记录', value: '仅限离线研究，未用于当前画面的逐人预测' })
    const report = obj(obj(models.aric).report), test = obj(report.test)
    add('离线模型 / 测试集宏平均综合得分', test.macro_f1)
    add('离线模型 / 测试集平均精确率均值', test.mean_average_precision)
  }
  return rows
}

export function activityMarkdown(data: LessonPackage): string[] {
  const activity = data.activity_analysis
  if (!activity) return ['## 学生行为与参与分析\n\n本课未提供同源行为分析；未知时段不计为零分。']
  const engagement = engagementExport(activity.frames)
  const scoreByFrame = new Map(engagement.frames.map(row => [row.frame_id, row]))
  const time = (value: number) => `${Math.floor(value / 60).toString().padStart(2, '0')}:${Math.floor(value % 60).toString().padStart(2, '0')}`
  const span = (row: { start_sec: number; end_sec: number }) => `${time(row.start_sec)}–${time(row.end_sec)}`
  const names: Record<string, string> = { read: '阅读', write: '书写', raise_hand: '举手', lookup: '模型抬头线索', turn_head: '转头线索', stand: '站立线索', discuss: '讨论线索', head_up: '头部抬起', head_lowered: '头部降低', hand_near_head: '手靠近头部', standing: '站姿线索', unknown: '状态待辨' }
  const percent = (value: number | null) => value === null ? '未知' : `${(value * 100).toFixed(1)}%`
  const visible = (frame: ActivityFrame) => {
    const current = frame.participation
    if (!current) return ''
    const states = Object.entries(frame.student_state_counts || {}).map(([label, count]) => `${names[label] || label}: ${count}`).join('、') || '无人可见'
    const weights = Object.entries(current.weights).map(([label, weight]) => `${names[label] || label} ${weight}`).join('、')
    const reference = frame.visual_reference
    const score = scoreByFrame.get(frame.id)
    const dense = reference && score ? `；学生整体积极度 ${score.smoothed_index === null ? '暂无评分' : `${score.smoothed_index.toFixed(1)}/100`}；有效行为 ${score.scored_people}/${score.total_people}（${percent(score.support_fraction)}）；${score.status === 'limited_support' ? '低支持估计' : '行为估计值'}；七类权重与离散修正；0.3旧值+0.7当前值连续平滑` : ''
    return `；当前可见状态 ${states}；可辨状态 ${current.observed_people}/${current.total_people}（${percent(current.coverage)}）；已检出参与动作 ${current.active_people}/${current.total_people}（${percent(current.active_fraction)}）；行为积极度参考 ${current.index === null ? '未知' : `${current.index.toFixed(1)}/100`}（仅当前明确动作的规则均值；${weights}；支持 ${current.scored_people}/${current.total_people}，${percent(current.score_coverage)}；不是心理积极度概率）${dense}`
  }
  const participant = (p: ActivityParticipant) => {
    const state = p.student_state
    const current = state ? `当前状态 ${names[state.label] || state.name}；状态来源 ${state.source}；状态置信度 ${state.confidence}；依据 ${state.evidence.join('、')}；姿态线索 ${(p.posture_tags || []).map(tag => `${names[tag.label] || tag.name}（${tag.evidence.join('、')}；关键点 ${tag.keypoint_indices.join(',')}）`).join('、') || '未知'}；` : ''
    return `  - 可见对象 ${p.track_id}：${current}原行为标签 ${p.label}；置信度 ${p.confidence}；时序规则分数 ${p.score === null ? '未知' : p.score}；来源 ${p.source}`
  }
  return [
    `## 学生行为与参与分析\n\n来源 SHA-256：${activity.source_sha256}\n采样间隔：${activity.sampling_interval_sec} 秒\n可见对象观测：${activity.summary.person_observations}；规则分数可用观测：${activity.summary.known_observations}；采样帧：${activity.summary.sampled_frames}；镜头：${activity.summary.scene_count}\n\n行为积极度参考只对当前明确阅读、书写、举手线索按公开权重求均值；头部抬起、降低等姿态单独展示，不赋参与分。已检出参与动作比例是当前明确动作人数 / 当前检测人物数，未识别不等于未参与。时序规则参与指数另需连续行为证据，为 0–100 的规则指数；对象时序分数为 0–1。它们不是积极度概率或知识掌握结果。镜头内轨迹仅表示可见对象，不确认师生身份；未观测或未知值保留为未知。规则分数覆盖率是时序规则分数可用观测数 / 检测对象数；即时行为标签独立展示，短片段或镜头切换可使已出现标签的对象仍无可用时序规则分数。`,
    ...(activity.frames.some(frame => frame.visual_reference) ? [`### 学生整体积极度\n\n本课平均积极度：${engagement.summary.average === null ? '暂无评分' : `${engagement.summary.average.toFixed(1)}/100`}；有效评分 ${engagement.summary.scored_frames}/${engagement.summary.total_frames} 帧，按有效画面时长加权。\n\n七类权重：${Object.entries(engagement.weights).map(([label,weight]) => `${names[label] || label} ${weight*100}分`).join('、')}。先求有效行为均值，再乘以 exp(-0.6 × 标准差)；连续同镜头按0.3旧值+0.7当前值平滑。低覆盖提示低支持估计，不清空已有评分；未知和几何姿态不补零，缺少视线数据时不补视线分。方法版本 ${engagement.method}，属于检测对象的辅助行为估计。原始模型记录与旧规则值保留。`] : []),
    `### 逐帧行为观测\n\n${activity.frames.map(frame => `- [${frame.id}] ${span(frame)} 镜头 ${frame.scene_id}；${frame.status}；检测 ${frame.detected_people}，规则分数可用对象 ${frame.known_people}；规则分数覆盖率 ${frame.coverage === null || !frame.detected_people ? '未知（无可用检测）' : `${(frame.coverage * 100).toFixed(1)}%`}；时序规则参与指数 ${frame.participation_index === null ? '未知' : frame.participation_index.toFixed(1)}${visible(frame)}；即时行为标签 ${Object.entries(frame.behaviors).map(([label, count]) => `${label}: ${count}`).join('、') || '未知'}${frame.image_url ? `；原始帧 ${frame.image_url}` : ''}\n${frame.participants.map(participant).join('\n')}`).join('\n') || '没有采样帧。'}`,
    `### 课堂情境模型预测\n\n${activity.context_segments.map(row => `- [${row.id}] ${span(row)} ${row.text}；${row.labels.map(label => `${label.name} (${label.label}) ${(label.probability * 100).toFixed(1)}%`).join('、') || '未提供概率'}；模型 ${row.model_id}；方法 ${row.method}`).join('\n') || '没有情境预测记录。'}`,
    `### 原始行为证据（生成时记录）\n\n${activity.evidence.map(row => `- [${row.id}] ${span(row)} ${row.text}`).join('\n') || '没有行为证据文本。'}`,
    `### 行为模型与训练记录\n\n${readableModelFields(activity.models).map(row => `- ${row.label}：${row.value}`).join('\n') || '未提供可显示的模型记录。'}`,
    `### 行为分析使用边界\n\n${activity.limitations.map(line => `- ${line}`).join('\n') || '未提供额外边界记录。'}`,
  ]
}
