"""Compare real rendered engagement scores and exports with an independent calculation."""
import argparse
import json
import math
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright, expect


def reference(frames):
    weights={'write':.85,'read':.8,'lookup':.85,'turn_head':.3,'raise_hand':.95,'stand':.9,'discuss':.75}
    rows=[]
    for i, frame in enumerate(frames):
        values=[weights[p['label']] for p in frame['participants'] if p.get('source') in ('trained_classifier','pretrained_detector')
                and p.get('student_state',{}).get('source')==p['source'] and p.get('student_state',{}).get('label')==p['label'] and p['label'] in weights]
        score=None
        if values:
            mean=sum(values)/len(values)
            deviation=math.sqrt(sum((v-mean)**2 for v in values)/len(values))
            score=round(mean*math.exp(-.6*deviation)*100,4)
        smoothed=score
        if i and score is not None and rows[-1]['smoothed'] is not None:
            old=frames[i-1]
            if (old['scene_id']==frame['scene_id'] and 0<frame['timestamp_sec']-old['timestamp_sec']<=1.2
                    and frame['start_sec']<=old['end_sec']+1e-6):
                smoothed=round(.3*rows[-1]['smoothed']+.7*score,4)
        rows.append({'frame_id':frame['id'],'score':score,'smoothed':smoothed,'scored_people':len(values)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:7032')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    lesson='unified-chemistry-001-9'
    with urlopen(args.url+f'/api/review/lessons/{lesson}') as response: package=json.load(response)
    frames=package['activity_analysis']['frames'];scores=reference(frames)
    valid=[(f,s) for f,s in zip(frames,scores) if s['smoothed'] is not None]
    average=sum(s['smoothed']*(f['end_sec']-f['start_sec']) for f,s in valid)/sum(f['end_sec']-f['start_sec'] for f,s in valid)
    checks=[];errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.emulate_media(reduced_motion='reduce')
        page.goto(args.url+f'/?view=review&lesson={lesson}',wait_until='networkidle')
        for seconds in [0,21,40,45,75,89.9]:
            page.get_by_role('slider',name='Whole-lesson playback progress').evaluate(
                '(el,seconds)=>{el.value=String(seconds);el.dispatchEvent(new Event("input",{bubbles:true}))}',seconds)
            selected=next((i for i,f in enumerate(frames) if f['start_sec']<=seconds<f['end_sec']),None)
            assert selected is not None
            expected=scores[selected]
            score=page.get_by_test_id('overall-engagement-index')
            expect(score).to_be_visible()
            if expected['smoothed'] is None: expect(score).to_have_text('No observations')
            else: assert abs(float(score.inner_text())-expected['smoothed'])<=.051,(seconds,score.inner_text(),expected)
            support=page.get_by_test_id('engagement-support').inner_text()
            assert str(expected['scored_people']) in support
            checks.append({'time':seconds,'displayed_score':score.inner_text(),'expected':expected})
            if seconds==21: page.get_by_test_id('overview-behavior').screenshot(path=str(args.output/'overview-21.png'))
        page.locator('.review-sidebar [data-view="behavior"]').click()
        expect(page.locator('[data-view-panel="behavior"]')).to_be_visible()
        page.get_by_role('slider',name='Whole-lesson playback progress').evaluate(
            '(el)=>{el.value="21";el.dispatchEvent(new Event("input",{bubbles:true}))}')
        expected21=next(s for f,s in zip(frames,scores) if f['start_sec']<=21<f['end_sec'])
        assert abs(float(page.get_by_test_id('overall-engagement-index').inner_text())-expected21['smoothed'])<=.051
        expect(page.locator('.participation-dot')).to_have_count(len(valid))
        expect(page.get_by_test_id('class-engagement-summary')).to_contain_text(f'{average:.1f}')
        page.get_by_text('Scoring rules and coverage',exact=True).click()
        expect(page.locator('.reference-policy')).to_contain_text('Standing cue 90 points')
        expect(page.locator('.reference-policy')).to_contain_text('Head-turn cue 30 points')
        page.screenshot(path=str(args.output/'behavior.png'),full_page=True)
        page.locator('.review-sidebar [data-view="report"]').click()
        expect(page.get_by_test_id('report-engagement')).to_contain_text(f'{average:.1f}')
        with page.expect_download() as result:
            page.get_by_role('button',name='Full data',exact=True).click()
        path=args.output/'export.json';result.value.save_as(path)
        exported=json.loads(path.read_text(encoding='utf-8'))
        derived=exported['derived_engagement']
        assert derived['method']=='overall_behavior_engagement_v1'
        assert abs(derived['summary']['average']-average)<.0002
        assert derived['summary']['scored_frames']==len(valid)
        assert derived['source_sha256']==package['lesson']['source_video']['sha256']
        for expected,actual in zip(scores,derived['frames']):
            assert actual['frame_id']==expected['frame_id'] and actual['scored_people']==expected['scored_people']
            assert actual['smoothed_index']==expected['smoothed'] or abs(actual['smoothed_index']-expected['smoothed'])<.0003
        assert exported['activity_analysis']==package['activity_analysis'],'Raw vision output must be preserved'
        with page.expect_download() as result:
            page.get_by_role('button',name='Review document',exact=True).click()
        path=args.output/'export.md';result.value.save_as(path)
        markdown=path.read_text(encoding='utf-8')
        assert f'Lesson average engagement: {average:.1f}/100' in markdown
        assert 'Overall student engagement 43.1/100' in markdown
        assert 'Limited-support estimate' in markdown
        assert errors==[],errors
        browser.close()
    report={'status':'passed','url':args.url,'read_only':True,'samples':checks,
            'scored_frames':len(valid),'total_frames':len(frames),'average':round(average,4),
            'exports_match_independent_reference':True,'raw_observations_preserved':True,'page_errors':errors}
    (args.output/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['status','scored_frames','total_frames','average','page_errors']},ensure_ascii=False))


if __name__=='__main__':main()
