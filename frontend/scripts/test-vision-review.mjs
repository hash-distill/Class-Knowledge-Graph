import assert from 'node:assert/strict'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import ts from 'typescript'
const directory = new URL('../node_modules/.cache/vision-review/', import.meta.url)
mkdirSync(directory, { recursive: true })
const source = readFileSync(new URL('../src/utils/visionReview.ts', import.meta.url), 'utf8')
const target = new URL('visionReview.mjs', directory)
writeFileSync(target, ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText)
const util = await import(target.href)
const frame = { participants: [{ track_id: 'P1', bbox: [1, 2, 30, 50], label: 'head_up' }, { track_id: 'P2', bbox: [30, 2, 60, 50], label: 'read' }] }
const blank = util.reviewRows(frame, [])
assert.equal(blank[0].label, 'unknown', 'Geometry alone is not a reviewed seven-class label')
assert.ok(blank.every(row => row.status === 'unreviewed' && row.role === 'unknown'))
assert.deepEqual(util.savedAnnotations(blank), [], 'Candidates never become truth without explicit confirmation')
const corrected = { ...blank[0], label: 'write', role: 'student', status: 'confirmed' }
const manual = { ...corrected, id: 'manual-1', origin: 'manual' }
const merged = util.reviewRows(frame, [corrected, manual])
assert.equal(merged.length, 3)
assert.equal(merged[0].label, 'write')
assert.equal(util.savedAnnotations(merged).length, 2)
assert.deepEqual(util.drawBox([80, 45], [10, -5], 100, 60), [10, 0, 80, 45])
assert.equal(util.drawBox([10, 10], [11, 40], 100, 60), null)
assert.equal(frame.participants[0].label, 'head_up', 'Human correction never mutates original model output')
console.log('Visual review draft and coordinate tests passed')
