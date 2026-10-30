"""Bind completed measured runs to the original lesson; never change raw ASR."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review.common import ROOT, atomic_json, read_json, sha256
from review.correction_comparison import validate_comparison
from review.correction_provider import CorrectionProvider
from review.mutation import LessonMutationLock


def assemble(directory, measured):
    bundle=read_json(directory/'bundle.json')
    asr_hash=sha256(directory/'asr.json')
    source=bundle['lesson']['source_video']['sha256']
    report=read_json(measured/'controlled-report/comparison.json')
    if report['source']['source_asr_sha256'] != asr_hash or report['source']['source_video_sha256'] != source:
        raise ValueError('Measured report is from another source')
    runs=[]
    for name,label in [('qwen','Qwen3-4B · Q4_K_M'),('cec4','ChineseErrorCorrector4-4B · Q4_K_M')]:
        path=measured/f'{name}-controlled.json'
        native_path=measured/f'{name}-native.json'
        envelope,native=read_json(path),read_json(native_path)
        result=envelope['result']
        lane=next(row for row in report['runs'] if row['label']==name+'-controlled')
        if lane['sha256'] != sha256(path):
            raise ValueError('Measured report is stale relative to its run: '+name)
        if result['source_asr_sha256'] != asr_hash or result['source_video_sha256'] != source:
            raise ValueError('Measured run source changed')
        if native['result']['source_asr_sha256'] != asr_hash or native['result']['source_video_sha256'] != source:
            raise ValueError('Native run source changed')
        origins=Counter(edit.get('origin','unrecorded') for row in result['segments'] for edit in row['edits'])
        runtime=envelope['runtime']
        model_hash=result['input_fingerprint']['provider']['model_sha256']
        alias='classroom-local' if name=='qwen' else 'classroom-corrector'
        if (result.get('model')!=alias or native.get('model')!=alias or
                native['result'].get('model')!=alias or
                native.get('benchmark_contract',{}).get('model_sha256')!=model_hash):
            raise ValueError('Native/controlled model identities differ: '+name)
        for row in native['native_outputs']:
            if row.get('status')=='complete':
                metadata=row.get('metadata',{})
                if metadata.get('model')!=alias or metadata.get('model_sha256')!=model_hash:
                    raise ValueError('Native response model identity differs: '+name)
        runs.append({'id':name,'model_label':label,'model_sha256':model_hash,
            'status':result['status'],'native_status':native['result']['status'],
            'accepted_edit_count':result['edit_count'],
            'model_accepted_edit_count':origins.get('model',0),
            'source_rule_accepted_count':sum(n for origin,n in origins.items() if origin!='model'),
            'additional_changed_spans_vs_rules':lane['deterministic_comparison']['aligned_changed_spans_additional_to_deterministic'],
            'model_proposal_count':lane['raw_model_proposal_count'],
            'guard_rejected_count':lane['guard_rejected_proposal_count'],
            'term_presence_count':lane['accepted']['summary']['terminology_group_presence_count'],
            'terminology_group_count':lane['accepted']['summary']['terminology_group_count'],
            'memory':{'unit':'MiB','scope':'whole_gpu_sampled',
                'baseline_mib':runtime.get('baseline_device_used_mib'),
                'loaded_mib':runtime.get('loaded_device_used_mib'),
                'peak_mib':runtime.get('peak_device_used_mib'),
                'sample_interval_ms':runtime.get('actual_median_sampling_interval_sec',.2)*1000},
            'segments':[{key:row[key] for key in ('transcript_id','start_sec','end_sec','original_text','text','uncertainties')}
                        for row in result['segments']],
            'native_segments':[{'transcript_id':row['transcript_id'],'start_sec':row['start_sec'],'end_sec':row['end_sec'],
                'original_text':row['original_text'],'text':row['corrected_text'],
                'uncertainties':[] if row.get('status')=='complete' else [row.get('error',{}).get('message','Request failed; original text retained')]}
                for row in native['native_outputs']],
            'conditions':envelope['conditions'],'native_conditions':native['conditions'],
            'runtime':runtime,'native_runtime':native['runtime'],
            'audit':{'controlled_artifact_sha256':sha256(path),'native_artifact_sha256':sha256(native_path),
                    'model_proposals':result['model_proposals'],'proposal_ledger':result['proposal_ledger'],
                    'native_call_records':native['native_outputs']}})
    data={'schema_version':'1.0','source_video_sha256':source,'source_asr_sha256':asr_hash,
        'evaluated_at':datetime.now(timezone.utc).isoformat(),
        'scope':'same_lesson_controlled_model_comparison',
        'reference_status':'no_human_verbatim_reference','runs':runs,
        'limitations':[
            'This lesson has no independent manual transcript. Reliable correction accuracy or actual erroneous correction rates cannot be calculated.',
            'Guarded results include contributions from OCR, the secondary recognizer, and contextual rules. Shared changes cannot all be attributed to the model.',
            'Accepted model candidates are counted by recorded origin. One of the two Qwen candidates is also repaired by rules alone; only one change goes beyond the rule-only result and still requires verification.',
            'Direct model output is unapproved and contains additions, deletions, and rewrites. It is for audio review only and is excluded from graph and Q&A evidence.',
            'Rule rejection statistics include model and source-rule candidates. Counts of standard-form groups describe only the diagnostic vocabulary for this lesson.',
            'GPU memory was sampled across the local RTX 4060 Laptop 8GB device, including display and other processes. Peaks between samples may be missed.',
            'See the external Chinese-LiPS report for complete manual-reference scoring. Its three samples are separate from this lesson and cannot establish whole-lesson accuracy.']}
    validate_comparison(data,bundle,asr_hash)
    return data


def publish(directory, measured, publish_proofreading=False):
    with LessonMutationLock(directory):
        data=assemble(directory,measured)
        baseline=read_json(measured/'source-before.json')
        for filename,expected in baseline.items():
            if sha256(directory/filename) != expected:
                raise ValueError('Immutable source changed: '+filename)
        if publish_proofreading:
            result=read_json(measured/'cec4-controlled.json')['result']
            if result['status']!='complete':
                raise ValueError('Partial model generation must not be published as completed correction')
            if result['input_fingerprint']['provider'] != CorrectionProvider(managed_lifecycle=False).status():
                raise ValueError('Measured correction provider differs from production profile')
            atomic_json(directory/'transcript.proofreading.json',result)
        atomic_json(directory/'correction.comparison.json',data)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-dir',type=Path,default=ROOT/'artifacts/unified-chemistry-001-9')
    parser.add_argument('--measured-dir',type=Path,default=ROOT/'artifacts/verification/cec4-comparison')
    parser.add_argument('--publish-proofreading',action='store_true')
    args=parser.parse_args()
    data=publish(args.media_dir.resolve(),args.measured_dir.resolve(),args.publish_proofreading)
    print('Bound measured runs:',[(row['model_label'],row['status']) for row in data['runs']])
