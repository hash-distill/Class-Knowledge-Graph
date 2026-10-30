<script lang="ts">
import type { ActivityFrame } from '@/types/lessonReview'
import type { VisualDraft, VisualLabel, VisualReviewDocument, VisualReviewHistory } from '@/types/visionReview'
// Drafts survive view switches in this browser tab, and never enter model data or exported truth.
const draftCache = new Map<string, { doc: VisualReviewDocument; rows: VisualDraft[] }>()
const draftBackups = new Map<string, { doc: VisualReviewDocument; rows: VisualDraft[] }>()
</script>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { visionReview } from '@/api/visionReview'
import { drawBox, reviewRows, savedAnnotations, visualLabels } from '@/utils/visionReview'
import { formatReviewTime } from '@/utils/lessonReview'

const props = defineProps<{ lessonId: string; frame: ActivityFrame | null }>()
const emit = defineEmits<{ lock: [seconds: number] }>()
const opened = ref(false), loading = ref(false), saving = ref(false), exporting = ref(false)
const doc = ref<VisualReviewDocument | null>(null), rows = ref<VisualDraft[]>([])
const selectedId = ref(''), error = ref(''), notice = ref(''), drawing = ref(false), original = ref(false)
const imageFailed = ref(false), historyOpen = ref(false), history = ref<VisualReviewHistory[]>([])
const svg = ref<SVGSVGElement | null>(null)
const reviewRoot = ref<HTMLElement | null>(null)
const enlarged = ref(false)
const dragStart = ref<number[] | null>(null), dragEnd = ref<number[] | null>(null)
let requestGeneration = 0
const locked = computed(() => doc.value?.frame)
const key = computed(() => doc.value ? `${doc.value.lesson_id}/${doc.value.frame_id}` : '')
const selected = computed(() => rows.value.find(row => row.id === selectedId.value))
const dirty = computed(() => Boolean(doc.value) && JSON.stringify(rows.value) !== JSON.stringify(reviewRows(doc.value!.frame, doc.value!.annotations)))
const confirmed = computed(() => rows.value.filter(row => row.status === 'confirmed').length)
const ignored = computed(() => rows.value.filter(row => row.status === 'ignored').length)
const pendingBox = computed(() => dragStart.value && dragEnd.value && doc.value ? drawBox(dragStart.value, dragEnd.value, doc.value.width, doc.value.height) : null)
const imageUrl = computed(() => locked.value ? `${locked.value.image_url}?review_image=${doc.value!.image_sha256}` : '')
const currentDifferent = computed(() => props.frame && props.frame.id !== doc.value?.frame_id)
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value))
const message = (reason: unknown) => reason instanceof Error ? reason.message : 'Operation failed. Please try again'
async function enlarge() {
  try {
    if (document.fullscreenElement === reviewRoot.value) await document.exitFullscreen()
    else await reviewRoot.value?.requestFullscreen()
  }
  catch { error.value = 'Fullscreen is unavailable in this window. Zoom in using the browser to review the frame' }
}
function updateFullscreen() { enlarged.value = document.fullscreenElement === reviewRoot.value }
onMounted(() => document.addEventListener('fullscreenchange', updateFullscreen))
onBeforeUnmount(() => document.removeEventListener('fullscreenchange', updateFullscreen))

function cacheDraft() {
  if (!doc.value) return
  if (dirty.value) {
    draftCache.set(key.value, { doc: clone(doc.value), rows: clone(rows.value) })
    if (draftCache.size > 50) draftCache.delete(draftCache.keys().next().value!)
  } else draftCache.delete(key.value)
}
watch(rows, cacheDraft, { deep: true, flush: 'sync' })
onBeforeUnmount(() => { cacheDraft(); requestGeneration++ })
watch(() => props.lessonId, () => {
  cacheDraft(); requestGeneration++; opened.value = false; doc.value = null; rows.value = []
  error.value = ''; notice.value = ''; loading.value = false; saving.value = false
})

async function loadCurrent() {
  if (!props.frame || loading.value || saving.value) return
  cacheDraft()
  const lesson = props.lessonId, frame = props.frame.id, generation = ++requestGeneration
  opened.value = true; loading.value = true; error.value = ''; notice.value = ''; drawing.value = false
  emit('lock', props.frame.timestamp_sec)
  try {
    const loaded = await visionReview.get(lesson, frame)
    if (generation !== requestGeneration) return
    const draft = draftCache.get(`${lesson}/${frame}`)
    doc.value = loaded; rows.value = reviewRows(loaded.frame, loaded.annotations)
    if (draft && draft.doc.basis_fingerprint === loaded.basis_fingerprint) {
      // Keep its original version: saving a stale draft must yield a conflict, never overwrite a newer review.
      doc.value = draft.doc; rows.value = draft.rows; notice.value = 'Restored the unsaved draft for this frame'
    } else if (draft) {
      draftBackups.set(`${lesson}/${frame}`, draft); draftCache.delete(`${lesson}/${frame}`)
      notice.value = 'Source has changed. The previous draft was not applied. Download the previous draft backup before reviewing again'
    }
    selectedId.value = rows.value[0]?.id || ''; original.value = false; imageFailed.value = false
    history.value = []; historyOpen.value = false
  } catch (reason) { if (generation === requestGeneration) error.value = message(reason) }
  finally { if (generation === requestGeneration) loading.value = false }
}

function modify(change: Partial<VisualDraft>) {
  if (!selected.value || saving.value) return
  Object.assign(selected.value, change)
  notice.value = ''
}
function editLabel(event: Event) { modify({ label: (event.target as HTMLSelectElement).value as VisualLabel, status: 'confirmed' }) }
function editRole(event: Event) { modify({ role: (event.target as HTMLSelectElement).value as VisualDraft['role'] }) }
function coordinate(index: number, event: Event) {
  if (!selected.value) return
  const box = [...selected.value.bbox] as VisualDraft['bbox']; box[index] = Number((event.target as HTMLInputElement).value)
  modify({ bbox: box })
}
function ignore() {
  if (!selected.value?.reason.trim()) { error.value = 'Enter a reason for exclusion first. For example, a mirror duplicate or a non-person detection'; return }
  error.value = ''; modify({ status: 'ignored' })
}
function unreview() {
  if (!selected.value) return
  if (selected.value.origin === 'manual') rows.value = rows.value.filter(row => row.id !== selectedId.value)
  else {
    const fresh = reviewRows(locked.value!, []).find(row => row.id === selectedId.value)!
    Object.assign(selected.value, fresh)
  }
}
function point(event: PointerEvent) {
  const rect = svg.value!.getBoundingClientRect()
  return [(event.clientX-rect.left)/rect.width*doc.value!.width, (event.clientY-rect.top)/rect.height*doc.value!.height]
}
function pointerDown(event: PointerEvent) {
  if (!drawing.value || !doc.value || event.button !== 0 || saving.value || loading.value) return
  event.preventDefault(); svg.value?.setPointerCapture(event.pointerId)
  dragStart.value = point(event); dragEnd.value = point(event)
}
function pointerMove(event: PointerEvent) { if (dragStart.value) dragEnd.value = point(event) }
function pointerUp(event: PointerEvent) {
  if (!dragStart.value || !doc.value) return
  dragEnd.value = point(event)
  const box = pendingBox.value
  dragStart.value = null; dragEnd.value = null
  if (!box) { notice.value = 'Box too small. Draw the box again'; return }
  if (rows.value.length >= 200) { error.value = 'Each frame supports up to 200 reviewed people'; return }
  const id = `manual-${crypto.randomUUID()}`
  rows.value.push({ id, origin: 'manual', bbox: box, status: 'confirmed', label: 'unknown', role: 'unknown', reason: '' })
  selectedId.value = id; drawing.value = false; notice.value = 'Added bounding box. Check the state and role before saving'
}
function cancelDrag() { dragStart.value = null; dragEnd.value = null }

async function save(reset = false) {
  if (!doc.value || saving.value || loading.value) return
  const baseline = new Map(reviewRows(doc.value.frame, doc.value.annotations).map(row => [row.id, row]))
  if (!reset && rows.value.some(row => row.status === 'unreviewed' && JSON.stringify(row) !== JSON.stringify(baseline.get(row.id)))) {
    error.value = 'Some edited people have not been confirmed. Confirm this person or restore the unreviewed state first. Save again'; return
  }
  const generation = requestGeneration, current = clone(doc.value)
  saving.value = true; error.value = ''; notice.value = ''
  try {
    const result = await visionReview.save(current, reset ? [] : savedAnnotations(rows.value))
    if (generation !== requestGeneration) return
    doc.value = result; rows.value = reviewRows(result.frame, result.annotations)
    draftCache.delete(key.value); historyOpen.value = false
    notice.value = reset ? 'Revoked review for this frame. History remains recoverable' : `Saved version ${result.version}`
  } catch (reason) { if (generation === requestGeneration) error.value = message(reason) }
  finally { if (generation === requestGeneration) saving.value = false }
}
async function reload() {
  if (!doc.value || loading.value || saving.value) return
  cacheDraft()
  if (dirty.value) draftBackups.set(key.value, { doc: clone(doc.value), rows: clone(rows.value) })
  drawing.value = false; cancelDrag()
  const generation = ++requestGeneration, lesson = doc.value.lesson_id, frame = doc.value.frame_id
  loading.value = true; error.value = ''
  try {
    const result = await visionReview.get(lesson, frame)
    if (generation !== requestGeneration) return
    doc.value = result; rows.value = reviewRows(result.frame, result.annotations)
    notice.value = 'Loaded the server version. The previous draft is available as a backup download. Then compare and edit again'; imageFailed.value = false
  } catch (reason) { if (generation === requestGeneration) error.value = message(reason) }
  finally { if (generation === requestGeneration) loading.value = false }
}
async function showHistory() {
  if (!doc.value || loading.value || saving.value) return
  const generation = requestGeneration
  try {
    const result = await visionReview.history(doc.value.lesson_id, doc.value.frame_id)
    if (generation === requestGeneration) { history.value = result; historyOpen.value = !historyOpen.value }
  } catch (reason) { if (generation === requestGeneration) error.value = message(reason) }
}
async function restore(version: number) {
  if (!doc.value || saving.value || loading.value || dirty.value) return
  const generation = requestGeneration; saving.value = true; error.value = ''
  try {
    const result = await visionReview.restore(doc.value, version)
    if (generation !== requestGeneration) return
    doc.value = result; rows.value = reviewRows(result.frame, result.annotations); historyOpen.value = false
    notice.value = `Restored historical content. Current version ${result.version}`
  } catch (reason) { if (generation === requestGeneration) error.value = message(reason) }
  finally { if (generation === requestGeneration) saving.value = false }
}
function backupDraft() {
  const draft = dirty.value ? { doc: doc.value, rows: rows.value } : draftBackups.get(key.value) || { doc: doc.value, rows: rows.value }
  const url = URL.createObjectURL(new Blob([JSON.stringify({ kind: 'unverified_draft_backup', ...draft }, null, 2)], { type: 'application/json' }))
  const link = document.createElement('a'); link.href = url; link.download = 'vision-review-draft.json'; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
async function exportSamples() {
  if (dirty.value || saving.value || exporting.value) return
  exporting.value = true; error.value = ''
  const generation = requestGeneration
  try { await visionReview.export(props.lessonId); if (generation === requestGeneration) notice.value = 'Exported the saved human review samples for this lesson ZIP' }
  catch (reason) { if (generation === requestGeneration) error.value = message(reason) }
  finally { exporting.value = false }
}
</script>

<template>
  <section ref="reviewRoot" class="vision-review" data-testid="vision-review" aria-label="Visual review and correction">
    <header class="review-heading"><div><p>HUMAN REVIEW</p><h3>Visual review and correction</h3><small>Confirm each person. Correct missed detections and misclassifications</small></div><button class="primary" :disabled="!frame || loading || saving" @click="loadCurrent">{{ opened ? 'Load current frame' : 'Start reviewing the current frame' }}</button></header>
    <p v-if="!opened" class="intro">Choose state, Exclude false detection. Or draw a box on the source image to add a missed person. Each confirmation is stored separately. Can be revoked and exported.</p>
    <div v-if="opened" class="review-content">
      <p v-if="loading" role="status">Loading review records for this frame</p>
      <p v-if="doc?.source_changed" class="warning">The source image or model results have changed. Previous review records are no longer active. Please verify again. History is preserved</p>
      <template v-if="doc && locked">
        <div class="review-toolbar"><span class="frame-chip">Locked {{ formatReviewTime(locked.timestamp_sec) }} · Scene {{ locked.scene_id }}</span><span>Confirmed {{ confirmed }} · Excluded {{ ignored }} · Unreviewed {{ rows.length-confirmed-ignored }}</span><span v-if="dirty" class="draft-chip">Unsaved draft</span><span v-else>Version {{ doc.version }}</span></div>
        <p v-if="currentDifferent" class="locked-note">The playback position has changed. This editor is still showing the locked frame. Loading the current frame preserves the draft for the original frame</p>
        <div class="review-grid" :aria-busy="saving || loading">
          <div class="review-canvas-column">
            <div class="canvas-toolbar"><button :class="{ active: drawing }" :disabled="saving || loading || original || imageFailed" @click="drawing = !drawing; cancelDrag()">{{ drawing ? 'Drawing a box · Select to cancel' : '＋ Draw a missed person' }}</button><button :aria-pressed="original" @click="original = !original; drawing = false">{{ original ? 'Show review annotations' : 'Compare source image' }}</button><button @click="enlarge">{{ enlarged ? 'Exit enlarged view' : 'Enlarge for review' }}</button></div>
            <div class="review-canvas" :class="{ drawing }" :style="{ aspectRatio: `${doc.width}/${doc.height}` }">
              <img :src="imageUrl" alt="Locked source lesson frame" @error="imageFailed = true" />
              <svg v-if="!original && !imageFailed" ref="svg" :viewBox="`0 0 ${doc.width} ${doc.height}`" aria-label="Human review canvas" data-testid="vision-review-canvas" @pointerdown="pointerDown" @pointermove="pointerMove" @pointerup="pointerUp" @pointercancel="cancelDrag" @keydown.escape="drawing = false; cancelDrag()">
                <g v-for="(row, index) in rows" :key="row.id" class="review-object" :class="[row.status, { selected: selectedId === row.id }]" role="button" :tabindex="drawing ? -1 : 0" :aria-label="`Review person ${index+1} ${visualLabels[row.label]}`" @click="!drawing && (selectedId = row.id)" @keydown.enter="selectedId = row.id" @keydown.space.prevent="selectedId = row.id">
                  <rect :x="row.bbox[0]" :y="row.bbox[1]" :width="Math.max(0,row.bbox[2]-row.bbox[0])" :height="Math.max(0,row.bbox[3]-row.bbox[1])" />
                  <text :x="row.bbox[0]+3" :y="Math.max(17,row.bbox[1]-5)">{{ index+1 }}{{ row.status === 'confirmed' ? ' ✓' : row.status === 'ignored' ? ' ×' : '' }}</text>
                </g>
                <rect v-if="pendingBox" class="pending-box" :x="pendingBox[0]" :y="pendingBox[1]" :width="pendingBox[2]-pendingBox[0]" :height="pendingBox[3]-pendingBox[1]" />
              </svg>
              <p v-if="imageFailed" class="image-error">Unable to load source frame. Reload before reviewing again</p>
            </div>
            <div class="canvas-legend"><span>Dashed line Unreviewed</span><span>Solid line Confirmed</span><span>Gray line Excluded</span><small>Box numbers identify people in the current frame only</small></div>
            <div class="object-list" aria-label="People to review"><button v-for="(row,index) in rows" :key="row.id" :aria-pressed="selectedId === row.id" :class="[row.status,{ active: selectedId === row.id }]" @click="selectedId = row.id">{{ index+1 }}<span>{{ row.status === 'confirmed' ? '✓' : row.status === 'ignored' ? '×' : '·' }}</span></button><p v-if="!rows.length">No people in this frame. Draw a box to add a person</p></div>
          </div>
          <fieldset class="review-inspector" :disabled="saving || loading || imageFailed">
            <template v-if="selected"><div class="inspector-heading"><strong>Person {{ rows.indexOf(selected)+1 }}</strong><span>{{ selected.origin === 'manual' ? 'Manually added box' : 'Model candidate' }}</span></div>
              <label>Visible states<select data-testid="review-label" :value="selected.label" @change="editLabel"><option v-for="(name,label) in visualLabels" :key="label" :value="label">{{ name }}</option></select></label>
              <label>Person role<select data-testid="review-role" :value="selected.role" @change="editRole"><option value="unknown">Unconfirmed</option><option value="student">Student</option><option value="teacher">Teacher</option></select></label>
              <p class="field-note">Only confirmed students with a known state enter the classification training samples</p>
              <div class="coordinates"><label v-for="(name,index) in ['Left','Top','Right','Bottom']" :key="name">{{ name }}<input type="number" :aria-label="`Box ${name.toLowerCase()} coordinate`" :value="selected.bbox[index]" step="1" min="0" :max="index%2 ? doc.height : doc.width" @change="coordinate(index,$event)" /></label></div>
              <label>Verification notes<textarea v-model="selected.reason" maxlength="500" rows="2" placeholder="Explain the reason for exclusion. For example, a mirror duplicate" /></label>
              <div class="inspector-actions"><button class="primary" data-testid="confirm-visual-object" @click="modify({ status: 'confirmed' })">Confirm this person</button><button @click="ignore">Exclude this person</button><button @click="unreview">{{ selected.origin === 'manual' ? 'Delete added box' : 'Restore unreviewed state' }}</button></div>
              <p class="field-note">{{ selected.status === 'confirmed' ? 'Confirmed. Effective after saving' : selected.status === 'ignored' ? 'Excluded. Effective after saving' : 'Not yet confirmed by a person. Excluded from training samples' }}</p>
            </template><p v-else>Select a box in the image or a number below. Start review</p>
          </fieldset>
        </div>
        <div class="review-footer"><div class="save-actions"><button class="primary" data-testid="save-vision-review" :disabled="saving || loading || imageFailed || !dirty" @click="save()">{{ saving ? 'Saving' : 'Save frame review' }}</button><button :disabled="saving || loading" @click="reload">Reload records</button><button :disabled="saving || loading || dirty || !doc.annotations.length" @click="save(true)">Revoke this frame review</button></div><div><button :disabled="saving || loading" @click="showHistory">Version history</button><button :disabled="saving || exporting || dirty" data-testid="export-vision-samples" @click="exportSamples">{{ exporting ? 'Exporting' : 'Export lesson samples' }}</button><button v-if="dirty || draftBackups.has(key)" @click="backupDraft">Draft backup</button></div></div>
        <div v-if="historyOpen" class="review-history"><p v-if="!history.length">No saved records</p><div v-for="entry in history" :key="entry.version"><span>Version {{ entry.version }} · {{ entry.annotations.length }} reviewed people · {{ new Date(entry.updated_at).toLocaleString('en-US') }}</span><button :disabled="saving || dirty || entry.basis_fingerprint !== doc.basis_fingerprint || entry.version === doc.version" @click="restore(entry.version)">{{ entry.basis_fingerprint !== doc.basis_fingerprint ? 'Source has changed' : 'Restore this version' }}</button></div></div>
        <p class="export-note">The export includes a source manifest, full-frame images, and confirmed student-state crops. Unreviewed, excluded, or unknown states and people with unconfirmed roles are excluded from classification samples; Full frames are partially annotated. They are not complete detection training labels</p>
      </template>
      <p v-if="error" class="review-error" role="alert">{{ error }}</p><p v-if="notice" class="review-notice" role="status">{{ notice }}</p>
    </div>
    <p class="review-boundary">Human reviews are saved in separate records. The model results above and whole-lesson curves retain their original calculations. Local corrections are not interpreted as class-wide engagement or model accuracy</p>
  </section>
</template>

<style scoped>
.vision-review{margin-top:24px;border:1px solid #cddff7;background:#f8fbff;border-radius:12px;overflow:hidden;min-width:0}.review-heading{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:20px 22px;background:linear-gradient(115deg,#e8f2ff,#f8fbff)}.review-heading p{font-size:8px;letter-spacing:.18em;color:#6f8fba}.review-heading h3{font-size:17px;font-weight:600;color:#234b81;margin:6px 0}.review-heading small,.intro{font-size:11px;color:#6985aa;line-height:1.8}.intro{padding:0 22px 18px}.vision-review button{font:inherit;font-size:11px;line-height:1.5;cursor:pointer;padding:8px 11px;border:1px solid #c7daf4;border-radius:6px;background:#fff;color:#426fa7;transition:background .16s,border-color .16s}.vision-review button:hover:enabled{border-color:#578fd9;background:#edf5ff}.vision-review button:focus-visible,.vision-review select:focus-visible,.vision-review input:focus-visible,.vision-review textarea:focus-visible{outline:2px solid #2466c3;outline-offset:2px}.vision-review button.primary{background:#2867bb;color:#fff;border-color:#2867bb}.vision-review button.primary:hover:enabled{background:#1f56a0}.vision-review button:disabled{opacity:.5;cursor:default}.review-content{padding:18px 22px;animation:review-in .22s ease}.review-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:9px;font-size:10px;color:#6d87a8;margin-bottom:14px}.frame-chip{border:1px solid #cfdef2;background:#edf4ff;border-radius:5px;padding:6px 9px;color:#3f6caa}.draft-chip{color:#2c61a8;font-weight:600}.locked-note,.warning{font-size:11px;color:#5375a2;background:#e9f2ff;border-radius:5px;padding:9px 11px;margin:0 0 12px;line-height:1.8}.review-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,.42fr);gap:18px;align-items:stretch}.review-canvas-column{min-width:0}.canvas-toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.canvas-toolbar .active{color:#fff;background:#356fbc}.review-canvas{position:relative;background:#142841;border-radius:7px;overflow:hidden;width:100%}.review-canvas img,.review-canvas svg{position:absolute;inset:0;width:100%;height:100%;display:block}.review-canvas img{object-fit:contain}.review-canvas svg{touch-action:none}.review-canvas.drawing{cursor:crosshair}.review-canvas.drawing .review-object{pointer-events:none}.review-object{cursor:pointer;outline:none}.review-object rect{fill:#a7c6f5;fill-opacity:.03;stroke:#b7cce9;stroke-width:1.5;stroke-dasharray:5 4;vector-effect:non-scaling-stroke}.review-object.confirmed rect{stroke:#6eb7ff;stroke-dasharray:none;fill-opacity:.08}.review-object.ignored{opacity:.4}.review-object.ignored rect{stroke:#a0aab9;stroke-dasharray:3 3}.review-object.selected rect,.review-object:focus-visible rect{stroke:#fff;stroke-width:3;fill:#78b6ff;fill-opacity:.15}.review-object text{fill:#fff;font-size:18px;paint-order:stroke;stroke:#163259;stroke-width:4px;stroke-linejoin:round;pointer-events:none}.pending-box{stroke:#fff;stroke-width:2;fill:#69aeff;fill-opacity:.25;vector-effect:non-scaling-stroke}.canvas-legend{display:flex;gap:12px;flex-wrap:wrap;line-height:1.8;color:#7c95b3;font-size:9px;padding:8px 0}.canvas-legend small{margin-left:auto;font-size:9px}.object-list{display:flex;flex-wrap:wrap;gap:5px;padding-top:8px;max-height:140px;overflow:auto}.object-list button{min-width:36px;padding:5px 6px;font-size:10px}.object-list button span{margin-left:4px}.object-list button.confirmed{background:#edf4ff;color:#2867bb}.object-list button.ignored{color:#93a4ba;text-decoration:line-through}.object-list button.active{border-color:#2867bb;box-shadow:0 0 0 1px #2867bb}.review-inspector{margin:0;border:1px solid #d5e3f5;border-radius:8px;background:#fff;min-width:0;padding:15px;display:flex;flex-direction:column;gap:12px}.inspector-heading{display:flex;justify-content:space-between;gap:8px;color:#3b6293;font-size:12px}.inspector-heading span{font-size:10px;color:#8299b7}.review-inspector label{display:flex;flex-direction:column;gap:6px;font-size:10px;color:#54749e}.review-inspector select,.review-inspector input,.review-inspector textarea{width:100%;box-sizing:border-box;border:1px solid #cddcf0;background:#f8fbff;color:#3c608f;border-radius:5px;padding:8px;font:inherit;font-size:11px}.review-inspector textarea{resize:vertical;line-height:1.7}.field-note{font-size:9px;color:#8499b6;line-height:1.8}.coordinates{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.coordinates input{padding:6px 3px;font-size:10px}.inspector-actions{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto}.inspector-actions button{font-size:10px;padding:7px 8px}.review-footer,.save-actions,.review-footer>div{display:flex;align-items:center;flex-wrap:wrap;gap:7px}.review-footer{justify-content:space-between;margin-top:18px;gap:12px}.review-footer button{font-size:10px}.review-boundary{font-size:9px;line-height:1.9;color:#7b94b4;border-top:1px solid #d9e6f6;padding:11px 22px}.export-note{font-size:9px;color:#8098b7;line-height:1.9;margin-top:12px}.review-error,.review-notice{font-size:11px;line-height:1.8;padding:10px 12px;border-radius:5px;margin-top:12px}.review-error{color:#8a4b60;background:#fff1f5;border:1px solid #edd5de}.review-notice{color:#416f9e;background:#e9f4ff}.review-history{margin-top:14px;padding:10px;border:1px solid #d4e2f4;border-radius:7px;font-size:10px;color:#6d87aa;max-height:200px;overflow:auto}.review-history>div{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px}.review-history button{padding:5px 7px;font-size:9px}.image-error{position:absolute;inset:0;display:grid;place-items:center;background:#162c49;color:#d7e6f9;font-size:12px;text-align:center;padding:20px}
@keyframes review-in{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}@media(prefers-reduced-motion:reduce){.review-content{animation:none}.vision-review button{transition:none}}@media(max-width:1050px){.review-grid{grid-template-columns:1fr}.review-inspector{display:grid;grid-template-columns:1fr 1fr}.inspector-heading,.coordinates,.inspector-actions,.review-inspector>.field-note{grid-column:1/-1}.review-heading{padding:17px}.review-content{padding:15px}.review-boundary{padding:10px 15px}}@media(max-width:550px){.review-heading{align-items:flex-start;flex-direction:column;gap:11px}.review-heading h3{font-size:15px}.review-heading .primary{width:100%}.review-inspector{grid-template-columns:1fr}.review-footer{align-items:stretch}.review-footer>div{width:100%}.review-toolbar{font-size:9px}.review-history>div{align-items:flex-start;flex-direction:column}}
.vision-review:fullscreen{margin:0;border:0;border-radius:0;width:100vw;height:100vh;box-sizing:border-box;overflow:auto;background:#f5f9ff}.vision-review:fullscreen .review-heading,.vision-review:fullscreen .review-content,.vision-review:fullscreen .review-boundary{max-width:1500px;margin-left:auto;margin-right:auto}.vision-review:fullscreen .review-grid{grid-template-columns:minmax(0,1fr) minmax(260px,.32fr)}@media(max-width:850px){.vision-review:fullscreen .review-grid{grid-template-columns:1fr}}
</style>
