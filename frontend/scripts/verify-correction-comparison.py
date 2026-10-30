"""Real API/media comparison acceptance. No intercepted routes or mock outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from playwright.sync_api import sync_playwright, expect

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:5175')
p.add_argument('--lesson',default='unified-chemistry-001-9')
p.add_argument('--output',type=Path,default=Path('output/playwright/cec4-comparison-real'))
args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
backend=Path(__file__).resolve().parents[2]/'backend'

def fetch(path):
    with urllib.request.urlopen(args.url+path,timeout=10) as response:
        return json.load(response)

data=fetch('/api/review/lessons/'+args.lesson)
proof=data['transcript_proofreading']
assert proof['model']=='classroom-corrector' and proof['status']=='complete'
comparison=data['correction_comparison']
assert comparison['source_video_sha256']==data['lesson']['source_video']['sha256']
assert comparison['source_asr_sha256']==proof['source_asr_sha256']
runs={row['id']:row for row in comparison['runs']}
assert set(runs)=={'qwen','cec4'}
assert runs['cec4']['model_accepted_edit_count']==0
assert runs['cec4']['source_rule_accepted_count']==13
assert all(len(row['segments'])==len(row['native_segments'])==28 for row in runs.values())
for filename,expected in json.loads((backend/'artifacts/verification/cec4-comparison/source-before.json').read_text(encoding='utf-8')).items():
    assert hashlib.sha256((backend/'artifacts/unified-chemistry-001-9'/filename).read_bytes()).hexdigest()==expected
managed=json.loads((backend/'artifacts/verification/cec4-comparison/production-managed-proofreading.json').read_text(encoding='utf-8'))
assert managed['status']=='complete' and managed['model']=='classroom-corrector'
assert [r['text'] for r in managed['segments']]==[r['text'] for r in proof['segments']]
with urllib.request.urlopen('http://127.0.0.1:18081/v1/models',timeout=5) as response:
    assert 'classroom-local' in [row['id'] for row in json.load(response)['data']]

errors=[]
checks=['Immutable ASR, audio, slides and behavior hashes match the frozen input',
        'Managed CEC4 generation completed and restored Qwen QA; published CEC4 text matches its result',
        'Comparison retains all28 source-timed segments in both lanes and separates model/rule contributions']
with sync_playwright() as pw:
    browser=pw.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(args.url+'/?lesson='+args.lesson+'&view=evidence',wait_until='networkidle')
    page.wait_for_function('document.querySelector("video").readyState>=2')
    page.evaluate('window.__comparisonSourceVideo=document.querySelector("video")')
    expect(page.get_by_test_id('active-correction-model').first).to_contain_text('ChineseErrorCorrector4-4B')
    panel=page.get_by_test_id('correction-comparison')
    assert panel.evaluate('el=>!el.open')
    page.get_by_test_id('correction-comparison-toggle').click()
    expect(page.get_by_test_id('correction-comparison-lane')).to_have_value('guarded')
    for name in runs:
        card=page.get_by_test_id('correction-comparison-run-'+name)
        expect(card.locator('.comparison-contributions dd').nth(0)).to_contain_text(str(runs[name]['model_accepted_edit_count']))
        expect(card.locator('.comparison-contributions dd').nth(1)).to_contain_text(str(runs[name]['source_rule_accepted_count']))
        expect(card).to_contain_text(format(int(runs[name]['memory']['peak_mib']),','))
    panel.screenshot(path=str(args.output/'controlled-comparison.png'))
    page.get_by_test_id('correction-comparison-lane').select_option('native')
    expect(page.get_by_test_id('correction-comparison-native-notice')).to_be_visible()
    row=page.get_by_test_id('comparison-segment-T0019')
    expect(row).to_contain_text('我说来了一个谷氨酸')
    expect(row).to_contain_text('我说来了一个咕噜声')
    target=next(r for r in data['transcript'] if r['id']=='T0019')['start_sec']
    row.locator('button').click()
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.1',arg=target)
    assert page.evaluate('document.querySelector("video")===window.__comparisonSourceVideo')
    page.get_by_label('Search correction model comparison').fill('咕噜声')
    assert panel.locator('.comparison-segment').count()==1
    page.screenshot(path=str(args.output/'native-risk-and-original.png'),full_page=True)
    page.get_by_label('Search correction model comparison').fill('')
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1')
    panel.screenshot(path=str(args.output/'comparison-mobile.png'))
    checks.append('Real comparison expands, shows measured memory/contributions, switches to unsafe native output and seeks exact original time;390px fits')
    page.locator('nav [data-view="knowledge"]').click()
    expect(page.locator('[data-active-view]')).to_have_attribute('data-active-view','knowledge')
    assert page.locator('.graph-node').filter(has_text='咕噜声').count()==0
    assert page.evaluate('document.querySelector("video")===window.__comparisonSourceVideo')
    page.locator('nav [data-view="report"]').click()
    with page.expect_download() as event:
        page.get_by_role('button',name='Full data',exact=True).click()
    destination=args.output/'lesson-export.json'; event.value.save_as(destination)
    exported=json.loads(destination.read_text(encoding='utf-8'))
    assert exported==fetch('/api/review/lessons/'+args.lesson)
    assert exported['correction_comparison']==comparison
    checks.append('Unsafe native hypotheses do not become graph nodes; complete JSON preserves both lanes and audit records')
    assert not errors,errors
    browser.close()
report={'passed':True,'scope':'real_api_media_models_no_mocks','lesson_id':args.lesson,
        'checks':checks,'page_errors':errors,'active_correction_model':proof['model']}
(args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
