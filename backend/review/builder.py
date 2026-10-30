from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
import json
from .alignment import align,extract_concepts
from .common import ROOT,atomic_json,read_json,sha256,union_duration,valid_id,validate_segments
from .provider import LLMProvider
from .correction_provider import CorrectionProvider
from .timebase import validate_media_timebase
from .blocks import speech_blocks
from .mutation import LessonMutationLock
from .proofreading import load_or_proofread, overlay_proofreading, graph_transcript


def build_lesson(lesson_id,video_path,media_dir,title,provider=None,*,allow_no_speech=False,correction_provider=None):
    with LessonMutationLock(media_dir):
        return _build_lesson(lesson_id,video_path,media_dir,title,provider,allow_no_speech=allow_no_speech,correction_provider=correction_provider)


def _build_lesson(lesson_id,video_path,media_dir,title,provider=None,*,allow_no_speech=False,correction_provider=None):
    valid_id(lesson_id)
    video_path,media_dir=Path(video_path).resolve(),Path(media_dir).resolve()
    previous=read_json(media_dir/'bundle.json') if (media_dir/'bundle.json').exists() else {}
    asr_path,slides_path=media_dir/'asr.json',media_dir/'slides.json'
    asr,pages=read_json(asr_path),read_json(slides_path)
    video_hash=sha256(video_path)
    for data in [asr,pages]:
        if data.get('status')!='complete':raise ValueError('Media extraction is incomplete')
        source=data.get('provenance',{}).get('source_video',{})
        if source.get('sha256')!=video_hash:raise ValueError('ASR/slides/video belong to different sources')
    duration=validate_media_timebase(asr,pages)
    transcript=validate_segments(asr['segments'],duration,'transcript')
    slides=validate_segments(pages['slides'],duration,'slide')
    if not transcript and not allow_no_speech:raise ValueError('No speech transcript was produced')
    if not slides:raise ValueError('No slide observations were produced')
    if set(r['id'] for r in transcript)&set(r['id'] for r in slides):raise ValueError('Evidence IDs must be globally unique')
    for page in slides:
        relative=Path(page['image_path'])
        image=(media_dir/relative).resolve()
        if not image.is_relative_to(media_dir) or not image.is_file():raise ValueError('Slide image missing or outside lesson')
        page['image_url']=f'/api/review/media/{lesson_id}/{quote(relative.as_posix())}'
    # Explicit injection keeps historical test/adaptor behavior; production uses
    # a dedicated correction role and restores QA before graph generation.
    corrector = correction_provider or provider or CorrectionProvider()
    try:
        proofreading = load_or_proofread(media_dir,transcript,slides,corrector,video_hash) if transcript else None
    finally:
        if hasattr(corrector,'close'):
            corrector.close()
    derived_transcript = proofreading['segments'] if proofreading else transcript
    fingerprint={'asr':sha256(asr_path),'slides':sha256(slides_path),'extractor':'literal_terms_v3_proofread',
                 'proofreading':sha256(media_dir/'transcript.proofreading.json') if proofreading else None}
    graph_path=media_dir/'graph.json'
    cached=read_json(graph_path) if graph_path.exists() else {}
    if cached.get('input_fingerprint')==fingerprint:
        graph=cached['graph']
    else:
        graph=extract_concepts(graph_transcript(proofreading) if proofreading else transcript,slides,provider or LLMProvider())
        atomic_json(graph_path,{'input_fingerprint':fingerprint,'graph':graph})
    alignment=align(derived_transcript,slides,graph)
    if proofreading:
        raw_ids = {row['id']: row['transcript_id'] for row in derived_transcript}
        for row in alignment:
            row['proofreading_ids'] = list(row['transcript_ids'])
            row['transcript_ids'] = [raw_ids.get(ref, ref) for ref in row['transcript_ids']]
    speech_duration=union_duration((r['start_sec'],r['end_sec']) for r in transcript)
    slide_duration=union_duration((r['start_sec'],r['end_sec']) for r in slides)
    sections=[]
    for node in graph['nodes']:
        refs=node['properties']['source_refs']
        sections.append({'title':node['label'],'text':f'原始转写/课件中检出该词，共 {len(refs)} 条证据。点击引用回看并核对识别文本。',
                         'evidence_ids':refs})
    limits=['ASR/OCR为机器识别，可能包含错字和公式识别错误，尚未计算人工标注准确率。',
            '课件边界来自视频抽样，误差见各页 uncertainty_sec；时间重合不自动等于语义完全匹配。',
            '概念由模型提议并校验原文出现，尚未教师确认；同段出现不是先修、因果或掌握关系。',
            '未提供与本课确认同步的学生行为录像，未计算投入度、知识掌握或教学效果。']
    report={'title':title+' · 整课证据报告',
        'summary':f'完整源录像 {duration:.2f} 秒，产生 {len(transcript)} 条语音片段、{len(slides)} 个课件画面段、{len(graph["nodes"])} 个原文可核实概念。',
        'sections':sections,'limitations':limits,'generated_by':'deterministic_evidence_summary',
        'statistics':{'duration_sec':duration,'transcript_segments':len(transcript),'slide_segments':len(slides),
                      'speech_span_union_sec':round(speech_duration,3),'slide_span_union_sec':round(slide_duration,3),
                      'concepts':len(graph['nodes']),'relations':len(graph['edges']),
                      'alignment_segments':len(alignment)}}
    if not transcript:
        limits.append('本片段未识别到可靠语音内容；不补写教师台词。内容复盘依据实际课件画面文字与行为证据，不能当作语音转写。')
    bundle={'schema_version':'3.0','data_source':'lesson_review',
        'lesson':{'id':lesson_id,'title':title,'duration_sec':duration,'status':'complete',
            'video_url':f'/api/review/media/{lesson_id}/source.mp4',
            'source_video':{'filename':video_path.name,'sha256':video_hash},
            'modalities':{'audio':'available' if transcript else 'unrecognized','slides':'available','behavior':'unavailable'},
            'provenance':{'asr':{**asr['provenance'],'source_asr_sha256':sha256(asr_path)},'slides':pages['provenance'],
                          'graph':graph['provenance'],'time_base':'source_media_seconds','offset_sec':0,
                          'created_at':datetime.now(timezone.utc).isoformat()}},
        'transcript':transcript,'evidence_blocks':speech_blocks(transcript),
        'slides':slides,'alignment':alignment,'graph':graph,
        'report':report,'snapshots':[]}
    if proofreading:
        bundle=overlay_proofreading(bundle,proofreading)
        report=bundle['report'];sections=report['sections'];limits=report['limitations']
    activity_path=media_dir/'activity.json'
    activity=read_json(activity_path) if activity_path.is_file() else previous.get('activity_analysis')
    if activity:
        from .activity import overlay_activity
        bundle=overlay_activity(bundle,activity,media_dir)
        report=bundle['report'];sections=report['sections'];limits=report['limitations']
    from .correction_comparison import overlay_comparison
    bundle=overlay_comparison(bundle,media_dir)
    from .model_upgrade import overlay_model_upgrade
    bundle=overlay_model_upgrade(bundle,media_dir)
    if previous.get('courseware'):
        from .courseware_service import overlay_bundle
        raw_bundle=bundle
        bundle=overlay_bundle(raw_bundle,previous['courseware'])
        atomic_json(media_dir/'bundle.base.json',raw_bundle)
        report=bundle['report'];sections=report['sections'];limits=report['limitations']
    atomic_json(media_dir/'report.json',report)
    markdown=[f'# {report["title"]}',report['summary'],'']
    for section in sections:
        markdown.extend([f'## {section["title"]}',section['text'],'证据：'+', '.join(section['evidence_ids']),''])
    markdown.extend(['## 数据边界',*['- '+s for s in limits]])
    (media_dir/'report.md').write_text('\n\n'.join(markdown),encoding='utf-8')
    atomic_json(media_dir/'bundle.json',bundle)
    # Private registry is never served; source absolute paths stay server-side.
    registry_path=ROOT/'configs/lessons.json'
    registry=read_json(registry_path) if registry_path.exists() else {}
    registry[lesson_id]={'source_video':str(video_path),'directory':str(media_dir)}
    atomic_json(registry_path,registry)
    return bundle
