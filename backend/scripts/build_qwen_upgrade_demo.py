"""Create a separate same-video Qwen-ASR/Qwen3.5 lesson without editing its parent."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
from urllib.parse import unquote

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review.common import ROOT,atomic_json,read_json,sha256,valid_id
from review.mutation import LessonMutationLock
from review.correction_provider import CorrectionProvider
from review.courseware_service import overlay_bundle,persist_bundle
from review.qwen_asr import inspect_wav
from scripts.build_unified_chemistry_demo import filter_graph_with_observed_courseware

EXPERIMENT=ROOT/'artifacts/verification/qwen-upgrade-20261028'
PARENT=ROOT/'artifacts/unified-chemistry-001-9'


def now():
    return datetime.now(timezone.utc).isoformat()


def remap(value,old,new):
    if isinstance(value,dict):return {k:remap(v,old,new) for k,v in value.items()}
    if isinstance(value,list):return [remap(v,old,new) for v in value]
    if isinstance(value,str):return value.replace(f'/api/review/media/{old}/',f'/api/review/media/{new}/')
    return value


def prepare(parent,native_path,directory,lesson_id):
    valid_id(lesson_id)
    parent,directory,native_path=Path(parent).resolve(),Path(directory).resolve(),Path(native_path).resolve()
    if parent==directory or directory.is_relative_to(parent) or parent.is_relative_to(directory):
        raise ValueError('New lesson must be separate from its immutable parent')
    native=read_json(native_path);old=read_json(parent/'asr.json');source=old['provenance']['source_video']
    audio_hash=sha256(parent/'audio.wav')
    audio=inspect_wav(parent/'audio.wav')
    declared=native['provenance'].get('source_audio',{}).get('sha256') or native['provenance'].get('audio_sha256')
    if native.get('status')!='complete' or declared!=audio_hash:
        raise ValueError('New ASR is incomplete or belongs to another WAV')
    if not native.get('segments'):
        raise ValueError('No native ASR segments')
    p=native['provenance']
    if (native.get('time_base')!='source_audio_seconds' or p.get('generator_exhausted') is not True
        or abs(p.get('audio_duration_sec',-1)-audio['duration_sec'])>.01
        or p.get('processed_range_sec')!=[0.0,audio['duration_sec']]
        or abs(audio['duration_sec']-source['audio_duration_sec'])>.01
        or p.get('timestamp_offset_sec')!=0):
        raise ValueError('New ASR does not cover the verified zero-offset source WAV')
    if (directory/'asr.json').is_file():
        previous=read_json(directory/'asr.json')['provenance'].get('native_asr_artifact',{})
        if previous.get('sha256')!=sha256(native_path):
            raise ValueError('Existing lesson uses another ASR snapshot; choose a new lesson id')
    directory.mkdir(parents=True,exist_ok=True)
    with LessonMutationLock(directory):
        for name in ['audio.wav','audio.provenance.json','asr.unguided.json','asr.secondary.json','slides.json']:
            if (parent/name).is_file():
                shutil.copyfile(parent/name,directory/name)
                if sha256(parent/name)!=sha256(directory/name):raise ValueError('Copied source changed')
        shutil.copyfile(parent/'asr.json',directory/'asr.baseline.json')
        shutil.copyfile(native_path,directory/'asr.native.json')
        pages=read_json(parent/'slides.json')
        for row in pages['slides']:
            relative=Path(row['image_path']);image=(parent/relative).resolve()
            if not image.is_relative_to(parent):raise ValueError('Unsafe slide path')
            (directory/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(image,directory/relative)
        # Publish a wrapper of native recognition; the machine text and its
        # actual forced-alignment boundaries are retained without rewriting.
        artifact=deepcopy(native);provenance=artifact['provenance']
        provenance['model_assets']=deepcopy(provenance.get('model',{}))
        provenance.update(source_video=deepcopy(source),audio_sha256=audio_hash,
            source_audio_sha256=audio_hash,audio_duration_sec=source['audio_duration_sec'],
            timestamp_offset_sec=0,generator_exhausted=True,processed_range_sec=[0,source['audio_duration_sec']],
            native_asr_artifact={'filename':'asr.native.json','sha256':sha256(native_path),
                'policy':'native_text_and_predicted_timestamps_preserved; source clock inherited from verified WAV'},
            model='Qwen3-ASR-1.7B')
        provenance.setdefault('config',{})['audio_time_offset_sec']=0
        artifact.setdefault('coverage',{})['processed_duration_sec']=source['audio_duration_sec']
        for row in artifact['segments']:
            row.setdefault('source','qwen3_asr')
        atomic_json(directory/'asr.json',artifact)
        old_id=read_json(parent/'bundle.json')['lesson']['id']
        activity=read_json(parent/'activity.json')
        prefix=f'/api/review/media/{old_id}/'
        manifest=[]
        for frame in activity['frames']:
            if not frame['image_url'].startswith(prefix):raise ValueError('Behavior URL source mismatch')
            relative=Path(unquote(frame['image_url'][len(prefix):]))
            source_image=(parent/relative).resolve()
            if not source_image.is_relative_to(parent):raise ValueError('Unsafe behavior image path')
            (directory/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source_image,directory/relative)
            if sha256(source_image)!=sha256(directory/relative):raise ValueError('Behavior image bytes changed')
            manifest.append({'path':relative.as_posix(),'sha256':sha256(source_image)})
        activity=remap(activity,old_id,lesson_id)
        activity['provenance']['qwen_upgrade_reuse']={'parent_lesson':old_id,'parent_activity_sha256':sha256(parent/'activity.json'),
            'predictions_changed':False,'source_video_sha256':source['sha256']}
        atomic_json(directory/'activity.json',activity)
        atomic_json(directory/'reuse-manifest.json',{'source_video_sha256':source['sha256'],
            'audio_sha256':audio_hash,'baseline_asr_sha256':sha256(parent/'asr.json'),
            'new_native_asr_sha256':sha256(native_path),'behavior_images':manifest,'prepared_at':now()})


def build(video,directory,lesson_id):
    from review.builder import _build_lesson
    from review.proofreading import load_or_proofread
    from review.common import validate_segments
    config_path=ROOT/'configs/qwen35_llm.json';config=read_json(config_path)
    with LessonMutationLock(directory):
        # Label the measured file before the builder fingerprints it for graph
        # extraction, so cached rebuilds do not oscillate between audit wrappers.
        asr,pages=read_json(directory/'asr.json'),read_json(directory/'slides.json')
        transcript=validate_segments(asr['segments'],asr['provenance']['source_video']['duration_sec'],'transcript')
        slides=validate_segments(pages['slides'],asr['provenance']['source_video']['duration_sec'],'slide')
        provider=CorrectionProvider(config_path=config_path)
        try:
            proof=load_or_proofread(directory,transcript,slides,provider,sha256(video))
        finally:
            provider.close()
        if proof['status']!='complete':raise ValueError('Qwen3.5 correction is incomplete; not publishing a completed lesson')
        proof['provenance'].update(model_label='Qwen3.5-4B',model_sha256=config['model']['sha256'])
        atomic_json(directory/'transcript.proofreading.json',proof)
        bundle=_build_lesson(lesson_id,video,directory,'认识氨基酸 · Qwen 新语音与保守校对',
            correction_provider=CorrectionProvider(config_path=config_path))
        base=read_json(directory/'bundle.base.json') if bundle.get('courseware') else bundle
        base['transcript_unguided']=read_json(directory/'asr.unguided.json')['segments']
        base['lesson']['provenance']['transcript_variants']={'displayed':'qwen3_asr_with_qwen35_guarded_correction',
            'raw_field':'transcript','proofread_field':'transcript_proofreading.segments',
            'baseline_artifact':'asr.baseline.json','unguided_export_field':'transcript_unguided'}
        filtered,archive=filter_graph_with_observed_courseware(base)
        archive_name='graph-candidates-qwen-upgrade.json'
        if not (directory/archive_name).is_file():atomic_json(directory/archive_name,archive)
        filtered['graph']['provenance']['display_policy']['original_candidates_artifact']=archive_name
        graph_cache=read_json(directory/'graph.json');graph_cache['graph']=deepcopy(filtered['graph'])
        atomic_json(directory/'graph.json',graph_cache)
        if bundle.get('courseware'):
            atomic_json(directory/'bundle.base.json',filtered)
            filtered=overlay_bundle(filtered,bundle['courseware'])
        persist_bundle(directory,filtered)
    return filtered


def publish_experiment(parent,directory,profile_path):
    from review.model_upgrade import overlay_model_upgrade
    with LessonMutationLock(directory):
        bundle=read_json(directory/'bundle.json');proof=bundle['transcript_proofreading']
        config=read_json(profile_path)
        origins=Counter(e.get('origin','unrecorded') for row in proof['segments'] for e in row['edits'])
        payload={'schema_version':'1.0','source_video_sha256':bundle['lesson']['source_video']['sha256'],
            'source_audio_sha256':sha256(directory/'audio.wav'),'evaluated_at':now(),
            'reference_status':'no_human_verbatim_reference',
            'asr_runs':[{'id':'baseline','model_label':'Whisper small · 同课课件词汇提示',
                'source_asr_sha256':sha256(parent/'asr.json'),'segments':read_json(parent/'asr.json')['segments']},
                {'id':'qwen3_asr','model_label':'Qwen3-ASR-1.7B · 无提示识别',
                'source_asr_sha256':sha256(directory/'asr.json'),'segments':bundle['transcript']}],
            'correction':{'model_label':'Qwen3.5-4B','model_sha256':config['model']['sha256'],
                'source_asr_sha256':sha256(directory/'asr.json'),'status':proof['status'],
                'accepted_edit_count':proof['edit_count'],'model_accepted_edit_count':origins.get('model',0),
                'source_rule_accepted_count':sum(n for k,n in origins.items() if k!='model'),
                'segments':[{k:row[k] for k in ('transcript_id','original_text','start_sec','end_sec','text','uncertainties')}
                    for row in proof['segments']]},
            'limitations':['本课没有独立人工逐字稿，不计算本课字错率或真实误改率。',
                '旧识别使用同课 OCR 词汇提示，新识别没有提示；两者设置不同，识别差异仅作描述。',
                '不同识别器按各自预测时间展示，段落编号和边界不代表一一对应。',
                '最终校对包含来源规则修订，改动数量不等于模型纠错收益；请回听原声核验。',
                '行为图像和预测来自同一原录像，不因更换语音或文字模型而改变。']}
        report_path=EXPERIMENT/'evaluation/comparison.json'
        if report_path.is_file():payload['external_summary']=read_json(report_path).get('external_summary')
        atomic_json(directory/'model.upgrade.json',payload)
        bundle=overlay_model_upgrade(bundle,directory)
        if 'model_upgrade_experiment' not in bundle:raise ValueError('Model upgrade evidence validation failed')
        base_path=directory/'bundle.base.json'
        if base_path.is_file():atomic_json(base_path,overlay_model_upgrade(read_json(base_path),directory))
        persist_bundle(directory,bundle)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--lesson-id',default='qwen35-chemistry-001-9')
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/qwen35-chemistry-001-9')
    parser.add_argument('--native-asr',type=Path,default=EXPERIMENT/'asr/unified-chemistry-001-9/asr.json')
    parser.add_argument('--stage',choices=['prepare','build','publish','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare(args.parent,args.native_asr,args.directory,args.lesson_id)
    if args.stage in ['build','all']:build(args.video,args.directory,args.lesson_id)
    if args.stage in ['publish','all']:publish_experiment(args.parent,args.directory,ROOT/'configs/qwen35_llm.json')
    print('Prepared Qwen upgrade stage:',args.stage,args.directory)
