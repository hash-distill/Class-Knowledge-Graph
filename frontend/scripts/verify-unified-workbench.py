"""Real same-class multimodal flow, independent views, media and local-model QA."""
import argparse
import json
from pathlib import Path
import urllib.request
from playwright.sync_api import sync_playwright, expect

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:5175')
p.add_argument('--lesson',default='unified-chemistry-001-9')
p.add_argument('--output',type=Path,default=Path('output/playwright/unified-workbench-real'))
p.add_argument('--courseware',type=Path,default=Path(__file__).resolve().parents[2]/'backend/artifacts/unified-demo-courseware/amino-acids-observed-courseware.pdf')
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
def get_data():
    with urllib.request.urlopen(args.url+'/api/review/lessons/'+args.lesson) as response:return json.load(response)
data=get_data();analysis=data['activity_analysis'];source=data['lesson']['source_video']['sha256']
assert source==analysis['source_sha256']==data['lesson']['provenance']['asr']['source_video']['sha256']==data['lesson']['provenance']['slides']['source_video']['sha256']
assert data['transcript'] and data['slides'] and any(f['participation_index'] is not None for f in analysis['frames'])
assert len(data['transcript_unguided'])==36
checks=['Voice, observed courseware and actual behavior all belong to the same original Chinese video SHA-256']
errors=[];answers=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    def capture(response):
        if response.url.endswith('/api/review/chat'):answers.append({'status':response.status,'body':response.json()})
    page.on('response',capture)
    page.goto(args.url+'/?lesson='+args.lesson+'&view=review',wait_until='networkidle')
    expect(page.get_by_label('Current lesson',exact=True)).to_have_value(args.lesson)
    page.wait_for_function('document.querySelector("video").readyState>=2&&!document.querySelector("video").error')
    page.evaluate('window.__originalSharedVideo=document.querySelector("video");window.__originalSharedVideo.muted=true')
    def same_video():
        assert page.locator('video').count()==1
        assert page.evaluate('document.querySelector("video")===window.__originalSharedVideo')
        page.wait_for_function('!document.querySelector("video").seeking&&document.querySelector("video").readyState>=2&&!document.querySelector("video").error')
        assert page.locator('video').evaluate('v=>v.readyState>=2&&!v.error')
    def seek(at):
        page.get_by_label('Whole-lesson playback progress').evaluate('(el,t)=>{el.value=String(t);el.dispatchEvent(new Event("input",{bubbles:true}))}',at)
        page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.1',arg=at)
        page.wait_for_function('!document.querySelector("video").seeking&&document.querySelector("video").readyState>=2')
    def view(name):
        page.locator('nav [data-view="'+name+'"]').click()
        expect(page.locator('[data-active-view]')).to_have_attribute('data-active-view',name)
        expect(page.locator('[data-view-panel="'+name+'"]')).to_be_visible()
        assert page.locator('[data-view-panel]').count()==1
        same_video()
    seek(40)
    frame=next(f for f in analysis['frames'] if f['start_sec']<=40<f['end_sec'])
    current_index=frame.get('participation',{}).get('index',frame['participation_index'])
    if current_index is not None:expect(page.get_by_test_id('overview-behavior')).to_contain_text(f"{current_index:.1f}")
    active=[row for row in data['transcript'] if row['start_sec']<=40<row['end_sec']]
    proofread={row['transcript_id']:row for row in data.get('transcript_proofreading',{}).get('segments',[])}
    if active:expect(page.get_by_test_id('overview-speech')).to_contain_text(proofread.get(active[0]['id'],active[0])['text'])
    expect(page.get_by_test_id('overview-courseware')).to_contain_text('No aligned slide image')
    page.screenshot(path=str(args.output/'overview-students-40s.png'))
    seek(44.5)
    expect(page.get_by_test_id('overview-courseware')).to_contain_text('认识氨基酸')
    assert page.get_by_test_id('overview-courseware').locator('img').evaluate('im=>im.complete&&im.naturalWidth>0')
    expect(page.get_by_test_id('overview-behavior')).to_contain_text('Unknown')
    page.screenshot(path=str(args.output/'overview-courseware-44s.png'))
    seek(47.5)
    expect(page.get_by_test_id('overview-courseware')).to_contain_text('No aligned slide image')
    checks.append('Overview shares exact source time; actual projection only43.5–47.5s and no-people/insufficient evidence remain unknown')
    seek(40);view('evidence')
    expect(page.get_by_label('Whole-lesson playback progress')).to_have_value('40')
    expect(page.locator('#evidence')).to_contain_text('Correction' if proofread else 'Transcript prompted with courseware vocabulary')
    page.get_by_label('Search lesson transcript').fill('氨基酸')
    view('knowledge')
    expect(page.locator('.graph-node')).to_have_count(len(data['graph']['nodes']))
    page.go_back(wait_until='networkidle')
    expect(page.locator('[data-active-view]')).to_have_attribute('data-active-view','evidence')
    expect(page.get_by_label('Search lesson transcript')).to_have_value('氨基酸')
    same_video();page.go_forward(wait_until='networkidle')
    expect(page.locator('[data-active-view]')).to_have_attribute('data-active-view','knowledge')
    same_video()
    checks.append('Independent views, URL history, stable video, playback time and KeepAlive search state survive navigation')
    view('review');seek(40)
    page.get_by_role('button',name='Play lesson video',exact=True).click()
    page.wait_for_function('!document.querySelector("video").paused')
    view('behavior')
    assert not page.locator('video').evaluate('v=>v.paused')
    page.get_by_role('button',name='Pause lesson video',exact=True).click();seek(40)
    page.screenshot(path=str(args.output/'behavior-overview.png'))
    for tab in ['heatmap','frames','models']:
        page.locator('#behavior-'+tab+'-tab').click()
        expect(page.locator('#behavior-'+tab+'-tab')).to_have_attribute('aria-selected','true')
        assert page.locator('[role="tabpanel"]:visible').count()==1
    page.locator('#behavior-models-tab').click()
    expect(page.locator('#behavior')).to_contain_text('Model')
    checks.append('Playing video stays the same node and keeps playing across views; behavior details are separate tabs')
    view('review');seek(40)
    page.locator('.assistant-trigger').click()
    drawer=page.locator('#review-assistant-drawer');expect(drawer).to_be_visible()
    page.get_by_role('button',name='Whole lesson',exact=True).click()
    page.get_by_label('Ask the review assistant').fill('Review this lesson using the audio, courseware and student behavior: Describe the amino acids mentioned in the audio, the three examples listed on the projected slide. and the observed reading and writing in the video. Cite the corresponding evidence for each point; Do not treat rule indices as psychological engagement. State when observations are unknown because they are not visible.')
    page.get_by_role('button',name='Send lesson question',exact=True).click()
    page.get_by_role('button',name='Close review assistant',exact=True).click()
    view('evidence');page.locator('#evidence-courseware-tab').click()
    page.get_by_label('Import courseware',exact=True).set_input_files(str(args.courseware))
    expect(page.locator('.cw-message[role="status"]')).to_contain_text('Courseware imported',timeout=150000)
    same_video();expect(page.get_by_label('Whole-lesson playback progress')).to_have_value('40')
    current=get_data()
    assert current['activity_analysis']==analysis and current['transcript']==data['transcript'] and current['transcript_unguided']==data['transcript_unguided']
    assert len(current['graph']['nodes'])==len(data['graph']['nodes'])
    assert {node['id'] for node in current['graph']['nodes']}=={node['id'] for node in data['graph']['nodes']}
    if proofread:
        assert current['transcript_proofreading']==data['transcript_proofreading']
        assert current['proofread_evidence_blocks']==data['proofread_evidence_blocks']
    if data.get('correction_comparison'):
        assert current['correction_comparison']==data['correction_comparison']
    if data.get('model_upgrade_experiment'):
        assert current['model_upgrade_experiment']==data['model_upgrade_experiment']
    matches=[m for m in current['courseware']['matches'] if m['status'] in ('auto_matched','confirmed')]
    assert matches and all(m['start_sec']>=43.5 and m['end_sec']<=47.5 for m in matches)
    page_two=current['courseware']['pages'][1]['id']
    assert all(row['page_id']!=page_two for row in current['courseware_evidence'])
    seek(44.5);expect(page.locator('.cw-preview img')).to_be_visible()
    page.screenshot(path=str(args.output/'imported-courseware.png'))
    checks.append('Actual browser PDF import matches only observed projection, leaves source note untimed and preserves ASR/behavior/graph policy/time')
    page.locator('.assistant-trigger').click()
    expect(page.locator('.model-answer')).to_be_visible(timeout=190000)
    assert answers and answers[-1]['status']==200,answers
    answer=answers[-1]['body'];kinds={row['kind'] for row in answer['citations']}
    assert answer['model']=='classroom-local',answer
    assert all(name in answer['answer'] for name in ('甘氨酸','丙氨酸','谷氨酸')),answer
    if proofread:
        assert not any(term in answer['answer'] for term in ('二发氨基酸','谷氨酸纳','一个缩疾')),answer
    elif any(term in answer['answer'] for term in ('二发','苏基','缩疾')):
        assert any('Term transcription requires verification' in warning for warning in answer.get('warnings',[])), 'Suspect ASR terms require explicit source-verification notice'
        expect(page.locator('.answer-warning')).to_contain_text('not verified standard definitions')
    assert 'behavior' in kinds and kinds.intersection({'slide','courseware'}) and kinds.intersection({'transcript','transcript_revision','transcript_proofread'}),answer
    page.locator('.model-answer').screenshot(path=str(args.output/'three-source-answer.png'))
    citation=next(row for row in answer['citations'] if row['kind']=='behavior')
    page.locator('.answer-citations button').filter(has_text=citation['id']+' ·').click()
    expect(drawer).not_to_be_visible()
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.1',arg=citation['start_sec'])
    same_video()
    checks.append('Real local-model answer cites voice, courseware and behavior from this same lesson; citation returns to persistent source video')
    view('report')
    with page.expect_download() as download:page.get_by_role('button',name='Full data',exact=True).click()
    path=args.output/'lesson-export.json';download.value.save_as(path)
    assert json.loads(path.read_text(encoding='utf-8'))==get_data()
    checks.append('Full data export retains original ASR, guided ASR, same-source observations and imported courseware')
    view('behavior');page.reload(wait_until='networkidle')
    page.wait_for_function('document.querySelector("video").readyState>=2&&!document.querySelector("video").error')
    page.evaluate('window.__originalSharedVideo=document.querySelector("video")')
    expect(page.locator('[data-active-view]')).to_have_attribute('data-active-view','behavior')
    expect(page.get_by_label('Current lesson',exact=True)).to_have_value(args.lesson)
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(args.output/'behavior-mobile.png'))
    view('review')
    page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(args.output/'overview-mobile.png'))
    checks.append('View deep link survives refresh; 390px independent views have no horizontal overflow')
    assert not errors,errors
    browser.close()
report={'passed':True,'scope':'actual_same_video_multimodal_views_media_import_and_local_model_no_mocks','lesson_id':args.lesson,'source_sha256':source,'checks':checks,'page_errors':errors,'answers':answers}
(args.output/'browser-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
