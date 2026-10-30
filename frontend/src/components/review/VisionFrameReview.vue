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
const message = (reason: unknown) => reason instanceof Error ? reason.message : '操作失败，请重试'
async function enlarge() {
  try {
    if (document.fullscreenElement === reviewRoot.value) await document.exitFullscreen()
    else await reviewRoot.value?.requestFullscreen()
  }
  catch { error.value = '当前窗口无法全屏，可在浏览器中放大页面后复核' }
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
      doc.value = draft.doc; rows.value = draft.rows; notice.value = '已恢复此帧未保存的草稿'
    } else if (draft) {
      draftBackups.set(`${lesson}/${frame}`, draft); draftCache.delete(`${lesson}/${frame}`)
      notice.value = '来源已变化，旧草稿没有应用。可下载旧草稿备份后重新核对'
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
  if (!selected.value?.reason.trim()) { error.value = '请先填写排除原因，例如镜面重复或并非人物'; return }
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
  if (!box) { notice.value = '框太小，请重新拖画'; return }
  if (rows.value.length >= 200) { error.value = '每帧最多支持 200 个复核对象'; return }
  const id = `manual-${crypto.randomUUID()}`
  rows.value.push({ id, origin: 'manual', bbox: box, status: 'confirmed', label: 'unknown', role: 'unknown', reason: '' })
  selectedId.value = id; drawing.value = false; notice.value = '已补入边框，请核对状态和角色后保存'
}
function cancelDrag() { dragStart.value = null; dragEnd.value = null }

async function save(reset = false) {
  if (!doc.value || saving.value || loading.value) return
  const baseline = new Map(reviewRows(doc.value.frame, doc.value.annotations).map(row => [row.id, row]))
  if (!reset && rows.value.some(row => row.status === 'unreviewed' && JSON.stringify(row) !== JSON.stringify(baseline.get(row.id)))) {
    error.value = '有修改过的对象尚未确认，请先确认此对象或恢复待复核，再保存'; return
  }
  const generation = requestGeneration, current = clone(doc.value)
  saving.value = true; error.value = ''; notice.value = ''
  try {
    const result = await visionReview.save(current, reset ? [] : savedAnnotations(rows.value))
    if (generation !== requestGeneration) return
    doc.value = result; rows.value = reviewRows(result.frame, result.annotations)
    draftCache.delete(key.value); historyOpen.value = false
    notice.value = reset ? '已撤回本帧复核，历史仍可恢复' : `已保存版本 ${result.version}`
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
    notice.value = '已载入服务器版本。原草稿可下载备份，再对照重新修改'; imageFailed.value = false
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
    notice.value = `已恢复历史内容，当前版本 ${result.version}`
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
  try { await visionReview.export(props.lessonId); if (generation === requestGeneration) notice.value = '已导出本课保存的人工复核样本 ZIP' }
  catch (reason) { if (generation === requestGeneration) error.value = message(reason) }
  finally { exporting.value = false }
}
</script>

<template>
  <section ref="reviewRoot" class="vision-review" data-testid="vision-review" aria-label="视觉复核与纠错">
    <header class="review-heading"><div><p>HUMAN REVIEW</p><h3>视觉复核与纠错</h3><small>逐对象确认，让漏检和误判有地方修正</small></div><button class="primary" :disabled="!frame || loading || saving" @click="loadCurrent">{{ opened ? '载入当前画面' : '开始复核当前帧' }}</button></header>
    <p v-if="!opened" class="intro">选择状态、排除误检，或在原图拖框补入漏检对象。每次确认单独留档，可撤回并导出。</p>
    <div v-if="opened" class="review-content">
      <p v-if="loading" role="status">正在读取此帧复核记录</p>
      <p v-if="doc?.source_changed" class="warning">原图片或模型结果已变化，旧复核记录已停用。请重新核对，历史记录仍保留</p>
      <template v-if="doc && locked">
        <div class="review-toolbar"><span class="frame-chip">已锁定 {{ formatReviewTime(locked.timestamp_sec) }} · 镜头 {{ locked.scene_id }}</span><span>确认 {{ confirmed }} · 排除 {{ ignored }} · 待复核 {{ rows.length-confirmed-ignored }}</span><span v-if="dirty" class="draft-chip">未保存草稿</span><span v-else>版本 {{ doc.version }}</span></div>
        <p v-if="currentDifferent" class="locked-note">播放位置已变化，此处仍编辑锁定帧。载入当前画面会保留原帧草稿</p>
        <div class="review-grid" :aria-busy="saving || loading">
          <div class="review-canvas-column">
            <div class="canvas-toolbar"><button :class="{ active: drawing }" :disabled="saving || loading || original || imageFailed" @click="drawing = !drawing; cancelDrag()">{{ drawing ? '正在拖框 · 点击取消' : '＋ 拖框补漏检' }}</button><button :aria-pressed="original" @click="original = !original; drawing = false">{{ original ? '显示复核标记' : '对照原图' }}</button><button @click="enlarge">{{ enlarged ? '退出放大' : '放大复核' }}</button></div>
            <div class="review-canvas" :class="{ drawing }" :style="{ aspectRatio: `${doc.width}/${doc.height}` }">
              <img :src="imageUrl" alt="锁定的课堂原始帧" @error="imageFailed = true" />
              <svg v-if="!original && !imageFailed" ref="svg" :viewBox="`0 0 ${doc.width} ${doc.height}`" aria-label="人工复核画布" data-testid="vision-review-canvas" @pointerdown="pointerDown" @pointermove="pointerMove" @pointerup="pointerUp" @pointercancel="cancelDrag" @keydown.escape="drawing = false; cancelDrag()">
                <g v-for="(row, index) in rows" :key="row.id" class="review-object" :class="[row.status, { selected: selectedId === row.id }]" role="button" :tabindex="drawing ? -1 : 0" :aria-label="`复核对象 ${index+1} ${visualLabels[row.label]}`" @click="!drawing && (selectedId = row.id)" @keydown.enter="selectedId = row.id" @keydown.space.prevent="selectedId = row.id">
                  <rect :x="row.bbox[0]" :y="row.bbox[1]" :width="Math.max(0,row.bbox[2]-row.bbox[0])" :height="Math.max(0,row.bbox[3]-row.bbox[1])" />
                  <text :x="row.bbox[0]+3" :y="Math.max(17,row.bbox[1]-5)">{{ index+1 }}{{ row.status === 'confirmed' ? ' ✓' : row.status === 'ignored' ? ' ×' : '' }}</text>
                </g>
                <rect v-if="pendingBox" class="pending-box" :x="pendingBox[0]" :y="pendingBox[1]" :width="pendingBox[2]-pendingBox[0]" :height="pendingBox[3]-pendingBox[1]" />
              </svg>
              <p v-if="imageFailed" class="image-error">原始帧加载失败，请重新加载后复核</p>
            </div>
            <div class="canvas-legend"><span>虚线 待复核</span><span>实线 已确认</span><span>灰线 已排除</span><small>框编号仅指当前画面的对象</small></div>
            <div class="object-list" aria-label="复核对象列表"><button v-for="(row,index) in rows" :key="row.id" :aria-pressed="selectedId === row.id" :class="[row.status,{ active: selectedId === row.id }]" @click="selectedId = row.id">{{ index+1 }}<span>{{ row.status === 'confirmed' ? '✓' : row.status === 'ignored' ? '×' : '·' }}</span></button><p v-if="!rows.length">当前无对象，可拖框补入</p></div>
          </div>
          <fieldset class="review-inspector" :disabled="saving || loading || imageFailed">
            <template v-if="selected"><div class="inspector-heading"><strong>对象 {{ rows.indexOf(selected)+1 }}</strong><span>{{ selected.origin === 'manual' ? '人工补框' : '模型候选' }}</span></div>
              <label>可见状态<select data-testid="review-label" :value="selected.label" @change="editLabel"><option v-for="(name,label) in visualLabels" :key="label" :value="label">{{ name }}</option></select></label>
              <label>对象角色<select data-testid="review-role" :value="selected.role" @change="editRole"><option value="unknown">暂不确认</option><option value="student">学生</option><option value="teacher">教师</option></select></label>
              <p class="field-note">仅确认学生且状态明确的对象进入分类训练样本</p>
              <div class="coordinates"><label v-for="(name,index) in ['左','上','右','下']" :key="name">{{ name }}<input type="number" :aria-label="`边框${name}坐标`" :value="selected.bbox[index]" step="1" min="0" :max="index%2 ? doc.height : doc.width" @change="coordinate(index,$event)" /></label></div>
              <label>核对说明<textarea v-model="selected.reason" maxlength="500" rows="2" placeholder="排除时请说明原因，例如镜面重复" /></label>
              <div class="inspector-actions"><button class="primary" data-testid="confirm-visual-object" @click="modify({ status: 'confirmed' })">确认此对象</button><button @click="ignore">排除此对象</button><button @click="unreview">{{ selected.origin === 'manual' ? '删除补框' : '恢复待复核' }}</button></div>
              <p class="field-note">{{ selected.status === 'confirmed' ? '已确认，保存后生效' : selected.status === 'ignored' ? '已排除，保存后生效' : '尚未人工确认，不进入训练样本' }}</p>
            </template><p v-else>点击画面中的边框或下方编号，开始复核</p>
          </fieldset>
        </div>
        <div class="review-footer"><div class="save-actions"><button class="primary" data-testid="save-vision-review" :disabled="saving || loading || imageFailed || !dirty" @click="save()">{{ saving ? '正在保存' : '保存本帧复核' }}</button><button :disabled="saving || loading" @click="reload">重新加载记录</button><button :disabled="saving || loading || dirty || !doc.annotations.length" @click="save(true)">撤回本帧复核</button></div><div><button :disabled="saving || loading" @click="showHistory">版本历史</button><button :disabled="saving || exporting || dirty" data-testid="export-vision-samples" @click="exportSamples">{{ exporting ? '正在导出' : '导出本课样本' }}</button><button v-if="dirty || draftBackups.has(key)" @click="backupDraft">草稿备份</button></div></div>
        <div v-if="historyOpen" class="review-history"><p v-if="!history.length">暂无保存记录</p><div v-for="entry in history" :key="entry.version"><span>版本 {{ entry.version }} · {{ entry.annotations.length }} 个复核对象 · {{ new Date(entry.updated_at).toLocaleString() }}</span><button :disabled="saving || dirty || entry.basis_fingerprint !== doc.basis_fingerprint || entry.version === doc.version" @click="restore(entry.version)">{{ entry.basis_fingerprint !== doc.basis_fingerprint ? '来源已变化' : '恢复此版本' }}</button></div></div>
        <p class="export-note">导出包含来源清单、整帧图片和已确认的学生状态裁剪图。待复核、排除、未知状态和身份未确认的对象不进入分类样本；整帧属于部分标注，不作为完整检测训练标签</p>
      </template>
      <p v-if="error" class="review-error" role="alert">{{ error }}</p><p v-if="notice" class="review-notice" role="status">{{ notice }}</p>
    </div>
    <p class="review-boundary">人工复核保存在独立记录中。上方模型结果和全课曲线保留原始计算，不会将局部修正解释为全班积极度或模型准确率</p>
  </section>
</template>

<style scoped>
.vision-review{margin-top:24px;border:1px solid #cddff7;background:#f8fbff;border-radius:12px;overflow:hidden;min-width:0}.review-heading{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:20px 22px;background:linear-gradient(115deg,#e8f2ff,#f8fbff)}.review-heading p{font-size:8px;letter-spacing:.18em;color:#6f8fba}.review-heading h3{font-size:17px;font-weight:600;color:#234b81;margin:6px 0}.review-heading small,.intro{font-size:11px;color:#6985aa;line-height:1.8}.intro{padding:0 22px 18px}.vision-review button{font:inherit;font-size:11px;line-height:1.5;cursor:pointer;padding:8px 11px;border:1px solid #c7daf4;border-radius:6px;background:#fff;color:#426fa7;transition:background .16s,border-color .16s}.vision-review button:hover:enabled{border-color:#578fd9;background:#edf5ff}.vision-review button:focus-visible,.vision-review select:focus-visible,.vision-review input:focus-visible,.vision-review textarea:focus-visible{outline:2px solid #2466c3;outline-offset:2px}.vision-review button.primary{background:#2867bb;color:#fff;border-color:#2867bb}.vision-review button.primary:hover:enabled{background:#1f56a0}.vision-review button:disabled{opacity:.5;cursor:default}.review-content{padding:18px 22px;animation:review-in .22s ease}.review-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:9px;font-size:10px;color:#6d87a8;margin-bottom:14px}.frame-chip{border:1px solid #cfdef2;background:#edf4ff;border-radius:5px;padding:6px 9px;color:#3f6caa}.draft-chip{color:#2c61a8;font-weight:600}.locked-note,.warning{font-size:11px;color:#5375a2;background:#e9f2ff;border-radius:5px;padding:9px 11px;margin:0 0 12px;line-height:1.8}.review-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,.42fr);gap:18px;align-items:stretch}.review-canvas-column{min-width:0}.canvas-toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.canvas-toolbar .active{color:#fff;background:#356fbc}.review-canvas{position:relative;background:#142841;border-radius:7px;overflow:hidden;width:100%}.review-canvas img,.review-canvas svg{position:absolute;inset:0;width:100%;height:100%;display:block}.review-canvas img{object-fit:contain}.review-canvas svg{touch-action:none}.review-canvas.drawing{cursor:crosshair}.review-canvas.drawing .review-object{pointer-events:none}.review-object{cursor:pointer;outline:none}.review-object rect{fill:#a7c6f5;fill-opacity:.03;stroke:#b7cce9;stroke-width:1.5;stroke-dasharray:5 4;vector-effect:non-scaling-stroke}.review-object.confirmed rect{stroke:#6eb7ff;stroke-dasharray:none;fill-opacity:.08}.review-object.ignored{opacity:.4}.review-object.ignored rect{stroke:#a0aab9;stroke-dasharray:3 3}.review-object.selected rect,.review-object:focus-visible rect{stroke:#fff;stroke-width:3;fill:#78b6ff;fill-opacity:.15}.review-object text{fill:#fff;font-size:18px;paint-order:stroke;stroke:#163259;stroke-width:4px;stroke-linejoin:round;pointer-events:none}.pending-box{stroke:#fff;stroke-width:2;fill:#69aeff;fill-opacity:.25;vector-effect:non-scaling-stroke}.canvas-legend{display:flex;gap:12px;flex-wrap:wrap;line-height:1.8;color:#7c95b3;font-size:9px;padding:8px 0}.canvas-legend small{margin-left:auto;font-size:9px}.object-list{display:flex;flex-wrap:wrap;gap:5px;padding-top:8px;max-height:140px;overflow:auto}.object-list button{min-width:36px;padding:5px 6px;font-size:10px}.object-list button span{margin-left:4px}.object-list button.confirmed{background:#edf4ff;color:#2867bb}.object-list button.ignored{color:#93a4ba;text-decoration:line-through}.object-list button.active{border-color:#2867bb;box-shadow:0 0 0 1px #2867bb}.review-inspector{margin:0;border:1px solid #d5e3f5;border-radius:8px;background:#fff;min-width:0;padding:15px;display:flex;flex-direction:column;gap:12px}.inspector-heading{display:flex;justify-content:space-between;gap:8px;color:#3b6293;font-size:12px}.inspector-heading span{font-size:10px;color:#8299b7}.review-inspector label{display:flex;flex-direction:column;gap:6px;font-size:10px;color:#54749e}.review-inspector select,.review-inspector input,.review-inspector textarea{width:100%;box-sizing:border-box;border:1px solid #cddcf0;background:#f8fbff;color:#3c608f;border-radius:5px;padding:8px;font:inherit;font-size:11px}.review-inspector textarea{resize:vertical;line-height:1.7}.field-note{font-size:9px;color:#8499b6;line-height:1.8}.coordinates{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.coordinates input{padding:6px 3px;font-size:10px}.inspector-actions{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto}.inspector-actions button{font-size:10px;padding:7px 8px}.review-footer,.save-actions,.review-footer>div{display:flex;align-items:center;flex-wrap:wrap;gap:7px}.review-footer{justify-content:space-between;margin-top:18px;gap:12px}.review-footer button{font-size:10px}.review-boundary{font-size:9px;line-height:1.9;color:#7b94b4;border-top:1px solid #d9e6f6;padding:11px 22px}.export-note{font-size:9px;color:#8098b7;line-height:1.9;margin-top:12px}.review-error,.review-notice{font-size:11px;line-height:1.8;padding:10px 12px;border-radius:5px;margin-top:12px}.review-error{color:#8a4b60;background:#fff1f5;border:1px solid #edd5de}.review-notice{color:#416f9e;background:#e9f4ff}.review-history{margin-top:14px;padding:10px;border:1px solid #d4e2f4;border-radius:7px;font-size:10px;color:#6d87aa;max-height:200px;overflow:auto}.review-history>div{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px}.review-history button{padding:5px 7px;font-size:9px}.image-error{position:absolute;inset:0;display:grid;place-items:center;background:#162c49;color:#d7e6f9;font-size:12px;text-align:center;padding:20px}
@keyframes review-in{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}@media(prefers-reduced-motion:reduce){.review-content{animation:none}.vision-review button{transition:none}}@media(max-width:1050px){.review-grid{grid-template-columns:1fr}.review-inspector{display:grid;grid-template-columns:1fr 1fr}.inspector-heading,.coordinates,.inspector-actions,.review-inspector>.field-note{grid-column:1/-1}.review-heading{padding:17px}.review-content{padding:15px}.review-boundary{padding:10px 15px}}@media(max-width:550px){.review-heading{align-items:flex-start;flex-direction:column;gap:11px}.review-heading h3{font-size:15px}.review-heading .primary{width:100%}.review-inspector{grid-template-columns:1fr}.review-footer{align-items:stretch}.review-footer>div{width:100%}.review-toolbar{font-size:9px}.review-history>div{align-items:flex-start;flex-direction:column}}
.vision-review:fullscreen{margin:0;border:0;border-radius:0;width:100vw;height:100vh;box-sizing:border-box;overflow:auto;background:#f5f9ff}.vision-review:fullscreen .review-heading,.vision-review:fullscreen .review-content,.vision-review:fullscreen .review-boundary{max-width:1500px;margin-left:auto;margin-right:auto}.vision-review:fullscreen .review-grid{grid-template-columns:minmax(0,1fr) minmax(260px,.32fr)}@media(max-width:850px){.vision-review:fullscreen .review-grid{grid-template-columns:1fr}}
</style>
