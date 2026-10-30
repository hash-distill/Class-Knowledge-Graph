<script setup lang="ts">
import { reviewTerm } from '@/utils/reviewPresentation'
import { computed, onDeactivated, onUnmounted, ref, watch } from 'vue'
import { useLessonReviewStore } from '@/stores/lessonReview'
import { reviewRequest } from '@/api/reviewService'
import { formatReviewTime } from '@/utils/lessonReview'
import { parseReviewJobState } from '@/utils/teacherWorkspace'
const store = useLessonReviewStore()
const busy = ref(false), error = ref(''), progress = ref(''), selectedPageId = ref(''), selectedMatchId = ref(''), chosenPage = ref(''), onlyPending = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
let epoch = 0
const document = computed(() => store.data?.courseware)
const currentMatch = computed(() => document.value?.matches.find(m => m.start_sec <= store.currentTime && store.currentTime < m.end_sec))
const page = computed(() => selectedPageId.value ? document.value?.pages.find(p => p.id === selectedPageId.value) : document.value?.pages.find(p => p.id === currentMatch.value?.page_id))
const matches = computed(() => (document.value?.matches || []).filter(m => !onlyPending.value || !['auto_matched', 'confirmed', 'rejected'].includes(m.status)))
const selectedMatch = computed(() => document.value?.matches.find(m => m.id === selectedMatchId.value))
const matchedCount = computed(() => document.value?.matches.filter(m => ['auto_matched', 'confirmed'].includes(m.status)).length || 0)
const pageNumber = (id: string | null) => document.value?.pages.find(p => p.id === id)?.page_number
const statusLabel = (status: string) => ({ auto_matched: 'Automatically matched', confirmed: 'Human confirmation', ambiguous: 'Unconfirmed', unmatched: 'Unmatched', rejected: 'No corresponding page' }[status] || status)
watch(() => store.data?.lesson.id, () => { epoch++; busy.value = false; error.value = ''; progress.value = ''; selectedPageId.value = ''; selectedMatchId.value = '' })
watch(() => currentMatch.value?.id, () => { selectedPageId.value = '' })
onUnmounted(() => { epoch++ })
onDeactivated(() => { epoch++; busy.value = false })
function chooseMatch(id: string) {
  selectedMatchId.value = id
  const row = selectedMatch.value
  if (row) { chosenPage.value = row.page_id || ''; selectedPageId.value = row.page_id || ''; store.seek(row.start_sec) }
}
async function refresh(id: string, version: number) {
  if (version !== epoch || store.data?.lesson.id !== id) return
  await store.selectLesson(id)
}
async function upload(event: Event) {
  const input = event.target as HTMLInputElement, file = input.files?.[0], id = store.data?.lesson.id
  if (!file || !id) return
  input.value = ''
  if (!/\.(pdf|pptx|ppt)$/i.test(file.name) || file.size > 64 * 1024 * 1024 || !file.size) { error.value = 'Choose a file no larger than 64 MiB   PDF, PPTX or PPT File.'; return }
  const version = ++epoch
  busy.value = true; error.value = ''; progress.value = 'Uploading courseware…'
  try {
    const job = await reviewRequest(`/lessons/${encodeURIComponent(id)}/courseware`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name) }, body: file }) as { job_id: string }
    progress.value = 'Parsing pages, Match recorded frames and verify terminology. Keep current records until processing completes…'
    for (let count = 0; count < 1200; count++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      if (version !== epoch) return
      const state = parseReviewJobState(await reviewRequest(`/jobs/${job.job_id}`))
      if (version !== epoch) return
      if (state.status === 'failed') throw new Error((typeof state.error === 'string' ? state.error : state.error?.message) || 'Courseware processing failed. The current lesson is retained.')
      if (state.status === 'cancelled') { progress.value = 'Task cancelled. View or retry the task on the Teaching improvement page.'; return }
      if (state.status === 'complete') {
        await refresh(id, version)
        progress.value = 'Courseware imported. Check the automatic matches and suggested terminology revisions.'
        return
      }
      progress.value = `${state.status === 'queued' ? 'Task queued' : 'Processing courseware'}${state.progress === undefined ? '' : ` · ${state.progress}%`}. View the actual task status on the Teaching improvement page.`
    }
    throw new Error('Processing is not finished. Refresh the lesson later to view the result; Closing the panel does not cancel server tasks.')
  } catch (reason) { if (version === epoch) error.value = reason instanceof Error ? reason.message : 'Courseware import failed.' }
  finally { if (version === epoch) busy.value = false }
}
async function confirm() {
  const id = store.data?.lesson.id, cw = document.value, match = selectedMatch.value
  if (!id || !cw || !match) return
  const version = ++epoch
  busy.value = true; error.value = ''; progress.value = ''
  try {
    await reviewRequest(`/lessons/${encodeURIComponent(id)}/courseware/matches`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ document_id: cw.document_id, match_id: match.id, page_id: chosenPage.value || null }) })
    await refresh(id, version)
    progress.value = 'Saved the page confirmation for this observed interval. Related terminology suggestions have been recalculated.'
  } catch (reason) { if (version === epoch) error.value = reason instanceof Error ? reason.message : 'Save failed.' }
  finally { if (version === epoch) busy.value = false }
}
async function detach() {
  const id = store.data?.lesson.id
  if (!id) return
  const version = ++epoch
  busy.value = true; error.value = ''; progress.value = ''
  try {
    await reviewRequest(`/lessons/${encodeURIComponent(id)}/courseware`, { method: 'DELETE' })
    await refresh(id, version)
    selectedPageId.value = ''; selectedMatchId.value = ''; progress.value = 'Restored the original lesson evidence. Previous courseware files are preserved.'
  } catch (reason) { if (version === epoch) error.value = reason instanceof Error ? reason.message : 'Removal failed.' }
  finally { if (version === epoch) busy.value = false }
}
</script>
<template>
  <section id="courseware" class="panel courseware-panel">
    <div class="panel-heading"><div><h2>Uploaded courseware alignment</h2><p>Verify recorded frames against the original courseware pages. Trace terminology to its source</p></div><span class="tag">{{ document ? `${document.pages.length} pages` : 'PDF / PPT' }}</span></div>
    <div class="courseware-actions"><input ref="fileInput" class="file-input" type="file" accept=".pdf,.pptx,.ppt" aria-label="Import courseware" :disabled="busy" @change="upload" /><button class="btn" :disabled="busy" @click="fileInput?.click()">{{ document ? 'Replace courseware' : 'Import PPT / PDF' }}</button><button v-if="document" class="btn subtle" :disabled="busy" @click="detach">Remove association</button><small>Up to 64 MiB · Original audio preserved</small></div>
    <p v-if="progress" class="cw-message" role="status">{{ progress }}</p><p v-if="error" class="cw-error" role="alert">{{ error }}</p>
    <div v-if="!document" class="cw-empty"><strong>Add the original courseware for this lesson</strong><p>Pages are matched using actual screen observations. Insufficient matches remain unconfirmed. Pages can be assigned per observed segment; Courseware pages are not distributed evenly across the timeline.</p></div>
    <template v-else>
      <div class="cw-source"><strong>{{ document.filename }}</strong><span>{{ matchedCount }} / {{ document.matches.length }} observed intervals matched</span><a :href="document.source_url" target="_blank" rel="noopener">View original file ↗</a></div>
      <div class="cw-preview"><img v-if="page" :src="page.image_url" :alt="`Uploaded courseware page ${page.page_number}`" /><p v-else class="cw-message">No uploaded courseware page is matched to the current time. Replay the video to confirm or browse the source pages manually.</p><div><label>View page <select v-model="selectedPageId" aria-label="Preview uploaded courseware pages"><option value="">Follow lesson</option><option v-for="p in document.pages" :key="p.id" :value="p.id">Page {{ p.page_number }}</option></select></label><span v-if="page">Page {{ page.page_number }} · {{ reviewTerm(page.text_source) }}</span></div><details v-if="page"><summary>Extracted courseware source text</summary><p>{{ page.text || 'No readable text is available on this page.' }}</p></details></div>
      <div class="cw-match-heading"><h3>Pages and lesson time</h3><label><input v-model="onlyPending" type="checkbox" />Show unconfirmed only</label></div>
      <div class="cw-matches"><button v-for="match in matches" :key="match.id" class="cw-match" :class="{ active: selectedMatchId === match.id, playing: currentMatch?.id === match.id }" @click="chooseMatch(match.id)"><span class="mono">{{ formatReviewTime(match.start_sec) }}–{{ formatReviewTime(match.end_sec) }}</span><strong>{{ match.page_id ? `Page ${pageNumber(match.page_id)}` : 'No corresponding page' }}</strong><small>{{ statusLabel(match.status) }}</small></button><p v-if="!matches.length" class="cw-message">No observed intervals match the filters.</p></div>
      <form v-if="selectedMatch" class="cw-confirm" @submit.prevent="confirm"><p>Confirm {{ formatReviewTime(selectedMatch.start_sec) }}–{{ formatReviewTime(selectedMatch.end_sec) }} — page actually displayed</p><div><select v-model="chosenPage" aria-label="Assign a courseware page to an observed interval" :disabled="busy"><option value="">No corresponding page / Teacher scene</option><option v-for="p in document.pages" :key="p.id" :value="p.id">Page {{ p.page_number }}</option></select><button class="btn" type="submit" :disabled="busy">Save page confirmation</button></div><small>Uses only the existing times of this screen segment. Automatic similarity is a matching cue. Not accuracy.</small></form>
      <div class="cw-revisions"><h3>Courseware-assisted terminology revisions <span>{{ store.data?.transcript_revisions?.length || 0 }} suggestions</span></h3><p class="cw-note">Only homophonic terms from aligned pages are used for revisions. Numbers, formulas, and negations remain unchanged; Suggestions have not been approved by a person.</p><div class="cw-revision-list"><article v-for="revision in store.data?.transcript_revisions" :key="revision.id"><button class="cw-time" @click="store.seek(revision.start_sec)">{{ formatReviewTime(revision.start_sec) }} Replay original audio ↗</button><p><small>Original</small>{{ revision.original_text }}</p><p class="revised"><small>Suggestion</small>{{ revision.corrected_text }}</p><div v-for="edit in revision.edits" :key="`${edit.start_char}-${edit.end_char}`" class="cw-edit"><span>{{ edit.original }} → {{ edit.replacement }}</span><button v-for="id in edit.page_ids" :key="id" @click="selectedPageId = id">Courseware page {{ pageNumber(id) }}</button></div></article></div><p v-if="!store.data?.transcript_revisions?.length" class="cw-note">No eligible revisions at present. The system does not rewrite source text merely to show an improvement.</p></div>
      <details v-if="document.warnings?.length" class="cw-warnings"><summary>Courseware processing notes</summary><p v-for="warning in document.warnings" :key="warning">{{ warning }}</p></details>
    </template>
  </section>
</template>
<style scoped>
.courseware-panel{overflow:hidden}.courseware-actions{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:0 23px 18px}.courseware-actions small{font-size:10px;color:#637a9b}.file-input{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden}.subtle{background:transparent}.cw-message,.cw-error{margin:0 23px 15px;padding:12px;font-size:12px;line-height:1.8;border-radius:7px;background:#e2ecf9;color:#486183}.cw-error{background:#dfeaf9;color:#3468b0}.cw-empty{padding:8px 23px 28px}.cw-empty strong{font-size:13px;color:var(--navy)}.cw-empty p,.cw-note{font-size:11px;line-height:1.9;color:#637a9b;margin-top:8px}.cw-source{margin:0 23px 15px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;font-size:11px;color:#627999}.cw-source strong{color:#4c6384;overflow-wrap:anywhere}.cw-source a{color:var(--blue);margin-left:auto}.cw-preview{margin:0 23px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#f0f5fc}.cw-preview img{width:100%;max-height:370px;display:block;object-fit:contain;background:white}.cw-preview>div{padding:13px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:9px;font-size:10px;color:#5f789c}select{max-width:100%;border:1px solid var(--line);background:#fff;border-radius:5px;padding:7px;color:var(--navy);font-size:11px}.cw-preview details{padding:0 13px 12px;font-size:10px;color:#60789a}.cw-preview details p{white-space:pre-wrap;max-height:180px;overflow:auto;line-height:1.8;margin-top:8px}.cw-match-heading{display:flex;justify-content:space-between;align-items:center;margin:22px 23px 9px;gap:10px}h3{font-size:13px;color:var(--navy);font-weight:500}.cw-match-heading label{font-size:10px;color:#60789a;white-space:nowrap}.cw-matches{max-height:220px;overflow:auto;margin:0 15px 15px}.cw-match{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:10px;align-items:center;width:100%;text-align:left;background:transparent;padding:12px 10px;border-radius:6px;border-bottom:1px solid #e0eaf9;color:#60799c;font-size:11px}.cw-match.active,.cw-match:hover{background:#e0eaf9}.cw-match.playing{border-left:3px solid var(--blue)}.cw-match strong{font-weight:500;color:#4f6585}.cw-match small{text-align:right;font-size:10px}.cw-confirm{margin:0 23px 18px;padding:14px;border-radius:7px;background:#e2ecf9}.cw-confirm p{font-size:12px;color:#526888;margin-bottom:10px}.cw-confirm>div{display:flex;gap:10px;flex-wrap:wrap}.cw-confirm small{display:block;font-size:10px;color:#627ca1;line-height:1.8;margin-top:8px}.cw-revisions{padding:18px 23px 24px;border-top:1px solid var(--line)}.cw-revisions h3 span{font-size:10px;margin-left:10px;font-weight:400;color:#617da3}.cw-revision-list{max-height:340px;overflow:auto}.cw-revision-list article{padding:16px 0;border-bottom:1px solid var(--line)}.cw-time{font-size:10px;color:var(--blue);background:transparent;margin-bottom:8px}.cw-revision-list p{font-size:12px;line-height:1.9;color:#617897;overflow-wrap:anywhere}.cw-revision-list p small{font-size:9px;margin-right:9px}.cw-revision-list .revised{color:#445b7c}.cw-edit{display:flex;gap:9px;flex-wrap:wrap;font-size:10px;color:#5a759b;margin-top:8px}.cw-edit button{font-size:10px;background:#e0eaf9;color:#567095;padding:3px 7px;border-radius:4px}.cw-warnings{margin:0 23px 20px;font-size:10px;color:#5d7597;line-height:1.8}.cw-warnings p{margin-top:7px}@media(max-width:550px){.courseware-actions{padding:0 15px 15px}.cw-source,.cw-preview,.cw-confirm{margin-left:15px;margin-right:15px}.cw-revisions{padding:15px}.cw-match{font-size:10px;gap:6px}.cw-source a{margin-left:0}}
</style>
