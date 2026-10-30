import type { TeacherTranscript, TeacherEvidenceBlock } from './teacherWorkspace'
export interface TimedEvidence {
  id: string; start_sec: number; end_sec: number; text: string
  [key: string]: unknown
}
export interface TranscriptSegment extends TimedEvidence { source: string; quality?: unknown }
export interface ProofreadSegment extends TimedEvidence {
  transcript_id: string; original_text: string; source_refs: string[]
  edits: { original: string; replacement: string; evidence_ids: string[]; reason: string }[]
  uncertainties: unknown[]; quality: { status: string }
}
export interface TranscriptProofreading {
  schema_version: string; status: string; method: string; model: string; human_verified: false
  source_asr_sha256: string; source_video_sha256: string; segments: ProofreadSegment[]
  counts?: Record<string, number>; edited_segment_count?: number; edit_count?: number; rejected_edit_count?: number; uncertain_segment_count?: number; provenance: Record<string, unknown>
  alternate_evidence?: (TimedEvidence & { kind: 'alternate_asr'; original_id: string; variant: string })[]
}
export interface CorrectionComparisonSegment {
  transcript_id: string; start_sec: number; end_sec: number; original_text: string; text: string; uncertainties: unknown[]
}
export interface CorrectionComparisonRun {
  id: 'qwen' | 'cec4'; model_label: string; model_sha256: string
  accepted_edit_count?: number | null; model_proposal_count?: number | null; guard_rejected_count?: number | null
  model_accepted_edit_count?: number | null; source_rule_accepted_count?: number | null; status?: string
  additional_changed_spans_vs_rules?: number | null
  term_presence_count?: number | null; terminology_group_count?: number | null
  memory?: {
    unit: 'MiB'; scope: 'whole_gpu_sampled'; baseline_mib?: number | null; loaded_mib?: number | null; peak_mib?: number | null; sample_interval_ms?: number | null
  } | null
  segments: CorrectionComparisonSegment[]
  native_segments?: CorrectionComparisonSegment[]; native_status?: string
  [key: string]: unknown
}
export interface CorrectionComparison {
  schema_version: '1.0'; source_video_sha256: string; evaluated_at: string
  scope: 'same_lesson_controlled_model_comparison'; reference_status: 'no_human_verbatim_reference'
  runs: CorrectionComparisonRun[]; limitations: string[]
  [key: string]: unknown
}
export interface ModelUpgradeAsrRun {
  id: 'baseline' | 'qwen3_asr'; model_label: string; source_asr_sha256: string
  segments: TimedEvidence[]; runtime?: Record<string, unknown>
  [key: string]: unknown
}
export interface ModelUpgradeExperiment {
  schema_version: '1.0'; source_video_sha256: string; source_audio_sha256: string
  evaluated_at: string; reference_status: 'no_human_verbatim_reference'
  asr_runs: ModelUpgradeAsrRun[]
  correction: {
    model_label: string; model_sha256: string; source_asr_sha256: string; status: string
    accepted_edit_count: number; model_accepted_edit_count: number; source_rule_accepted_count: number
    segments: CorrectionComparisonSegment[]
    [key: string]: unknown
  }
  external_summary?: unknown; limitations: string[]
  [key: string]: unknown
}
export interface EvidenceBlock extends TimedEvidence { kind: 'transcript'; source_refs: string[] }
export interface SlideSegment extends TimedEvidence { title: string; image_url: string; method: string; uncertainty_sec: number }
export interface AlignmentSegment {
  id: string; start_sec: number; end_sec: number; transcript_ids: string[]; slide_id: string | null; concept_ids: string[]; method: string; status: string; score: number | null
  [key: string]: unknown
}
export interface EvidenceNode {
  id: string; label: string
  properties: { kind: string; source_refs: string[]; status: string; [key: string]: unknown }
}
export interface EvidenceEdge {
  id: string; source: string; target: string; label: string
  properties: { source_refs: string[]; [key: string]: unknown }
}
export interface LessonSummary { id: string; title: string; duration_sec: number; status: string }
export interface CoursewarePage { id: string; page_number: number; text: string; image_url: string; text_source: string }
export interface CoursewareMatch { id: string; slide_id: string; page_id: string | null; start_sec: number; end_sec: number; status: string; score: number; margin: number; method: string; candidates?: { page_id: string; score: number }[] }
export interface CoursewareDocument { document_id: string; filename: string; sha256: string; format: string; source_url: string; pages: CoursewarePage[]; matches: CoursewareMatch[]; warnings: string[]; provenance: Record<string, unknown> }
export interface TranscriptRevision extends TimedEvidence { transcript_id: string; original_text: string; corrected_text: string; status: string; edits: { original: string; replacement: string; page_ids: string[]; match_ids: string[]; start_char: number; end_char: number; method: string }[] }
export interface RevisedBlock extends TimedEvidence { kind: 'transcript' | 'transcript_revision'; source_refs: string[]; revision_ids?: string[] }
export interface ActivityContext extends TimedEvidence { labels: { label: string; name: string; probability: number }[]; model_id: string; method: 'trained_context' }
export interface StudentState { label: string; name: string; source: string; confidence: number | null; evidence: string[]; keypoint_indices: number[] }
export interface VisibleParticipation { index: number | null; status: string; observed_people: number; total_people: number; unknown_people: number; coverage: number | null; active_people: number; active_fraction: number | null; scored_people: number; score_coverage: number | null; weights: Record<string, number>; semantics: string; population_scope: string }
export interface VisualBehaviorReference { semantics: 'configured_visible_behavior_reference_v1'; status: 'estimated' | 'insufficient_support' | 'no_people'; index: number | null; smoothed_index: number | null; scored_people: number; total_people: number; support_fraction: number | null; weights: Record<string, number>; weight_source: string; minimum_support_fraction: number; minimum_people: number; smoothing: { previous: number; current: number; maximum_gap_sec: number }; calibrated_engagement_probability: false; population_scope: string; definition?: string }
export interface ActivityParticipant { track_id: string; bbox: [number, number, number, number]; label: string; confidence: number; score: number | null; source: string; student_state?: StudentState; posture_tags?: StudentState[]; keypoints?: number[][] }
export interface ActivityFrame {
  id: string; start_sec: number; end_sec: number; timestamp_sec: number; scene_id: number; scene_cut: boolean; image_url: string; status: string
  // known_people / coverage count usable temporal rule scores, independently of instantaneous labels.
  detected_people: number; known_people: number; coverage: number | null; participation_index: number | null; behaviors: Record<string, number>; participants: ActivityParticipant[]
  width?: number; height?: number; student_state_counts?: Record<string, number>; participation?: VisibleParticipation
  visual_reference?: VisualBehaviorReference
  detection_review?: { method: string; proposal_count: number; retained_people: number; previous_detected_people?: number; excluded_detection_proposals: number; tile_windows: number; wall_seconds: number; population_recall_evaluated: false }
}
export interface ActivityAnalysis {
  schema_version: '1.0'; status: 'complete'; source_sha256: string; sampling_interval_sec: number
  context_segments: ActivityContext[]; frames: ActivityFrame[]
  summary: { sampled_frames: number; scene_count: number; person_observations: number; known_observations: number }
  models: Record<string, unknown>; limitations: string[]; evidence: (TimedEvidence & { kind: 'behavior' })[]
}
export interface LessonPackage {
  schema_version: '3.0'; data_source: 'lesson_review'
  lesson: LessonSummary & {
    video_url: string
    source_video: { filename: string; sha256: string; [key: string]: unknown }
    modalities: { audio: unknown; slides: unknown; behavior: unknown }
    provenance: Record<string, unknown>
    [key: string]: unknown
  }
  transcript: TranscriptSegment[]; slides: SlideSegment[]; alignment: AlignmentSegment[]
  teacher_transcript?: TeacherTranscript[]
  teacher_evidence_blocks?: TeacherEvidenceBlock[]
  teacher_review?: { version: number; source_revision: string; corrected_segment_count: number }
  transcript_proofreading?: TranscriptProofreading
  correction_comparison?: CorrectionComparison
  model_upgrade_experiment?: ModelUpgradeExperiment
  proofread_evidence_blocks?: (TimedEvidence & { kind: 'transcript_proofread'; source_refs: string[]; proofreading_ids: string[] })[]
  evidence_blocks?: EvidenceBlock[]
  courseware?: CoursewareDocument
  courseware_evidence?: (TimedEvidence & { kind: 'courseware'; page_id: string; match_id?: string; match_ids?: string[] })[]
  transcript_revisions?: TranscriptRevision[]
  revised_evidence_blocks?: RevisedBlock[]
  activity_analysis?: ActivityAnalysis
  graph: { nodes: EvidenceNode[]; edges: EvidenceEdge[]; [key: string]: unknown }
  report: {
    title: string; summary: string; sections: { title: string; text: string; evidence_ids: string[] }[]; limitations: string[]; generated_by: unknown
    [key: string]: unknown
  }
  snapshots: unknown[]
  [key: string]: unknown
}
export interface ReviewCitation extends TimedEvidence { kind: string; slide_id?: string }
export type ChatScope = 'lesson' | 'current'
export interface ReviewChatRequest { lesson_id: string; query: string; at_sec: number; scope: ChatScope }
export interface ReviewChatResponse {
  warnings?: string[]
  answer: string; source: 'local_llm' | 'remote_llm'; model: string; citations: ReviewCitation[]; evidence_ids: string[]; request_id: string
}
export interface ReviewMessage {
  id: number; query: string; at_sec: number; scope: ChatScope; status: 'pending' | 'answered' | 'error'; response?: ReviewChatResponse; error?: string
}
