<script setup lang="ts">
import { headingText } from '@/utils/reviewPresentation'
import { computed, ref } from 'vue'
import { useLessonReviewStore } from '@/stores/lessonReview'
import { modelUpgradeLanes, modelUpgradeTime } from '@/utils/modelUpgrade'

const store = useLessonReviewStore()
const showAll = ref(false)
const experiment = computed(() => store.data?.model_upgrade_experiment)
const lanes = computed(() => experiment.value ? modelUpgradeLanes(experiment.value, store.currentTime, showAll.value) : [])
const count = (value: number) => value.toLocaleString('en-US')
const correctionStatus = computed(() => {
  const value = experiment.value?.correction.status
  return value ? ({ complete: 'Completed', partial: 'Partially completed', running: 'Processing', failed: 'Failed', pending: 'Not completed' }[value] || value) : 'Unknown'
})
</script>

<template>
  <details v-if="experiment" class="model-upgrade" data-testid="model-upgrade-experiment">
    <summary data-testid="model-upgrade-toggle"><span>Model upgrade experiment<small>Same video · Baseline ASR, New ASR and conservative correction</small></span><span class="expand-label">Expand review <i aria-hidden="true">⌄</i></span></summary>
    <div class="upgrade-body">
      <p class="upgrade-boundary">This lesson has no human verbatim transcript. Character error rate has not been evaluated. Recognized text, Time boundaries and corrections are machine-generated. Check against the original audio.</p>
      <p class="upgrade-note">Conservative correction run: {{ correctionStatus }} · Edit counts describe run results only.</p>
      <dl class="upgrade-metrics"><div><dt>Lesson character error rate</dt><dd>Not evaluated</dd></div><div><dt>Accepted model candidates</dt><dd>{{ count(experiment.correction.model_accepted_edit_count) }}<small>edits</small></dd></div><div><dt>Source-rule revisions</dt><dd>{{ count(experiment.correction.source_rule_accepted_count) }}<small>edits</small></dd></div><div><dt>Total accepted edits</dt><dd>{{ count(experiment.correction.accepted_edit_count) }}<small>edits</small></dd></div></dl>
      <div class="upgrade-tools"><span>Video time {{ modelUpgradeTime(store.currentTime) }}</span><button data-testid="model-upgrade-all-toggle" :aria-pressed="showAll" @click="showAll = !showAll">{{ showAll ? 'Show current time only' : 'View all segments' }}</button></div>
      <p class="upgrade-note">Each ASR run uses its own time boundaries; Segment counts and boundaries may differ between the three columns.</p>
      <div class="upgrade-lanes">
        <section v-for="lane in lanes" :key="lane.id" class="upgrade-lane" :data-testid="`model-upgrade-lane-${lane.id}`">
          <h3>{{ headingText(lane.title) }}<small>{{ lane.model }}</small></h3>
          <div class="upgrade-segments">
            <button v-for="segment in lane.segments" :key="segment.id" class="upgrade-segment" :data-testid="`model-upgrade-${lane.id}-${segment.id}`" :aria-label="`${lane.title} ${modelUpgradeTime(segment.start_sec)} Replay original audio`" @click="store.seek(segment.start_sec)"><span>{{ modelUpgradeTime(segment.start_sec) }}–{{ modelUpgradeTime(segment.end_sec) }} ↗</span><p>{{ segment.text }}</p><small>{{ segment.id }} · {{ lane.id === 'correction' ? 'Machine correction · Unverified' : 'Automatic recognition · Unverified' }}</small></button>
            <p v-if="!lane.segments.length" class="upgrade-empty">No {{ lane.title }} segment at the current time. Replay the audio or view all segments.</p>
          </div>
        </section>
      </div>
      <p class="upgrade-note">Comparison copies are for audio verification only. It adds no knowledge graph entries or Q&A citations; The current lesson retains separate sources for the raw transcript and correction evidence.</p>
      <details class="upgrade-provenance"><summary>View experiment provenance and notes</summary><p>Run time: {{ experiment.evaluated_at }}</p><p>Video SHA-256: {{ experiment.source_video_sha256 }}</p><p>Audio SHA-256: {{ experiment.source_audio_sha256 }}</p><p v-for="run in experiment.asr_runs" :key="run.id">{{ run.model_label }} · ASR file SHA-256: {{ run.source_asr_sha256 }}</p><p>Correction model SHA-256: {{ experiment.correction.model_sha256 }}</p><p v-for="limit in experiment.limitations" :key="limit">{{ limit }}</p><p v-if="experiment.external_summary !== undefined">External-sample evaluation records are preserved in the exported data; It does not represent the character error rate for this lesson.</p></details>
    </div>
  </details>
</template>

<style scoped>
.model-upgrade{margin:0 22px 20px;border:1px solid #d2dceb;border-radius:7px;background:#f3f7fd;overflow:hidden}.model-upgrade>summary{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:15px 16px;color:#4c668a;font-size:12px;cursor:pointer;list-style:none}.model-upgrade>summary::-webkit-details-marker{display:none}.model-upgrade>summary small{display:block;color:#5f7ca5;font-size:9px;line-height:1.8;margin-top:5px}.expand-label{white-space:nowrap;font-size:9px;color:#5776a2}.expand-label i{display:inline-block;font-style:normal;margin-left:5px}.model-upgrade[open] .expand-label i{transform:rotate(180deg)}.upgrade-body{border-top:1px solid #d1e0f6;padding:15px}.upgrade-boundary{font-size:10px;line-height:1.9;background:#e1ebf9;color:#4b6a96;border:1px solid #c9d9f0;border-radius:5px;padding:10px 12px}.upgrade-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}.upgrade-metrics>div{border-right:1px solid #cddef6;padding-left:9px}.upgrade-metrics>div:first-child{padding-left:0}.upgrade-metrics>div:last-child{border:0}.upgrade-metrics dt{font-size:9px;color:#5c7aa5;line-height:1.7}.upgrade-metrics dd{font-size:19px;color:#4d678c;margin:6px 0 0}.upgrade-metrics dd small{font-size:9px;margin-left:4px;color:#5d7ca7}.upgrade-tools{display:flex;gap:10px;align-items:center;justify-content:space-between;border-top:1px solid #cddef6;padding-top:12px;color:#526e94;font-size:10px}.upgrade-tools button{font-size:9px;padding:7px 9px;border:1px solid #ccd9eb;background:#f9fbfe;color:#4e6f9d;border-radius:4px}.upgrade-note{font-size:9px;line-height:1.9;color:#617da3;margin-top:10px}.upgrade-lanes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:12px 0}.upgrade-lane{min-width:0;border:1px solid #d2dded;border-radius:5px;overflow:hidden;background:#f7fafe}.upgrade-lane h3{font-size:11px;color:#4b678f;font-weight:500;background:#e1ebf9;padding:11px 10px;line-height:1.5;margin:0}.upgrade-lane h3 small{font-size:8px;font-weight:400;color:#5b7aa6;display:block;margin-top:5px;overflow-wrap:anywhere}.upgrade-segments{max-height:290px;overflow:auto}.upgrade-segment{display:block;background:none;text-align:left;width:100%;padding:12px 10px;border-bottom:1px solid #dde9f9}.upgrade-segment:hover{background:#e9f1fb}.upgrade-segment>span{font-size:8px;color:#5173a2;line-height:1.8}.upgrade-segment p{font-size:11px;line-height:1.9;color:#4e698f;margin:6px 0;overflow-wrap:anywhere}.upgrade-segment>small{font-size:8px;color:#5e7ca7;line-height:1.8}.upgrade-empty{font-size:10px;color:#617da3;line-height:1.9;padding:12px 10px}.upgrade-provenance{font-size:9px;line-height:1.9;color:#617da3;margin-top:12px;border-top:1px solid #cddef6;padding-top:9px}.upgrade-provenance summary{cursor:pointer}.upgrade-provenance p{overflow-wrap:anywhere;margin:8px 0 0}.upgrade-segment:focus-visible,.upgrade-tools button:focus-visible,.model-upgrade summary:focus-visible{outline:2px solid #4774b4;outline-offset:-2px}@media(max-width:650px){.upgrade-lanes{grid-template-columns:1fr}.upgrade-segments{max-height:210px}.upgrade-metrics{grid-template-columns:repeat(2,minmax(0,1fr));row-gap:15px}.upgrade-metrics>div:nth-child(3){padding-left:0}.upgrade-metrics>div:nth-child(2){border:0}}@media(max-width:550px){.model-upgrade{margin:0 15px 18px}.upgrade-body{padding:12px}.model-upgrade>summary{padding:12px}.upgrade-tools{flex-wrap:wrap}}
</style>
