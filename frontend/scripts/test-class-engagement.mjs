import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import ts from 'typescript'
const cache = new URL('../node_modules/.cache/class-engagement/', import.meta.url)
mkdirSync(cache, { recursive: true })
const target = new URL('classEngagement.mjs', cache)
writeFileSync(target, ts.transpileModule(readFileSync(new URL('../src/utils/classEngagement.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText)
const scoring = await import(target.href)
const frame = (labels, timestamp = 0, scene = 1) => ({ id: `F${timestamp}`, scene_id: scene,
  timestamp_sec: timestamp, start_sec: timestamp, end_sec: timestamp+.5, detected_people: labels.length,
  participants: labels.map((label, i) => ({ track_id: `P${i}`, label,
    source: label === 'head_up' ? 'pose_rule' : 'trained_classifier',
    student_state: { label, source: label === 'head_up' ? 'pose_geometry_rule' : 'trained_classifier' } })),
})
const input = frame(['write', 'read', 'lookup', 'turn_head', 'raise_hand', 'stand', 'discuss'])
const original = structuredClone(input)
const result = scoring.overallEngagement([input])[0]
const values = [.85,.8,.85,.3,.95,.9,.75], mean = values.reduce((a,b) => a+b)/7
const deviation = Math.sqrt(values.reduce((sum,v) => sum+(v-mean)**2,0)/7)
assert.ok(Math.abs(result.index-mean*Math.exp(-.6*deviation)*100)<.0002)
assert.equal(result.scored_people, 7)
assert.equal(result.status, 'estimated')
assert.deepEqual(input, original, 'Scores are derived without rewriting observations')
const sparse = scoring.overallEngagement([frame(['read','unknown','head_up','unknown'])])[0]
assert.equal(sparse.index, 80, 'Low support receives an explicit estimate instead of permanent unknown')
assert.equal(sparse.status, 'limited_support'); assert.equal(sparse.support_fraction, .25)
assert.equal(scoring.overallEngagement([frame(['head_up','unknown'])])[0].index, null)
assert.equal(scoring.overallEngagement([frame([])])[0].status, 'no_people')
const old = frame(['read','read','read']), current = frame(['raise_hand','raise_hand','raise_hand'], .5)
assert.equal(scoring.overallEngagement([old,current])[1].smoothed_index, 90.5)
assert.equal(scoring.overallEngagement([old,{...current,scene_id:2}])[1].smoothed_index, 95)
assert.equal(scoring.overallEngagement([old,{...current,timestamp_sec:3}])[1].smoothed_index, 95)
const unknown = frame(['unknown'], .5)
assert.equal(scoring.overallEngagement([old,unknown,frame(['raise_hand'],1)])[2].smoothed_index, 95)
const duplicate = [old, {...current,timestamp_sec:0}]
assert.equal(scoring.overallEngagement(duplicate)[1].smoothed_index, 95, 'Duplicate timestamps are not temporal predecessors')
const impostor = frame(['read']); impostor.participants[0].student_state.label = 'raise_hand'
assert.equal(scoring.overallEngagement([impostor])[0].index, null)
const rows = [frame(['read']), frame([], .5), frame(['raise_hand'], 1, 2)]
const scores = scoring.overallEngagement(rows)
assert.equal(scoring.engagementSummary(rows,scores).average, 87.5)
assert.equal(scoring.engagementSummary(rows,scores).scored_frames, 2)
assert.equal(scoring.engagementSummary([],[]).average, null)
const legacy = readFileSync(new URL('../../vision/Class_Detection/configs/pipeline_legacy.yaml', import.meta.url), 'utf8')
for (const [label,weight] of Object.entries(scoring.engagementWeights)) {
  assert.ok(legacy.includes(`${label}: ${weight.toFixed(2)}`), `Weight ${label} follows the existing project policy`)
}
console.log('PASS: seven-state class score, variance penalty, support disclosure, source bounds, smoothing, gaps and duration-weighted summary')
