"""Browser integration check against a running local Vite server and real inference JSON."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser()
parser.add_argument('--url', required=True)
parser.add_argument('--json', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
raw = json.loads(args.json.read_text(encoding='utf-8'))
last_seconds = int(raw[-1]['timestamp_sec'])
last_time = f'{last_seconds//60:02}:{last_seconds%60:02}'
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(args.url, wait_until='networkidle')
    expect(page.get_by_text('Actual inference records', exact=True)).to_be_visible()
    expect(page.get_by_label('Rule index variance: Unknown')).to_be_visible()
    expect(page.get_by_text('No lesson knowledge graph provided', exact=True)).to_be_visible()
    page.locator('summary').filter(has_text='Model provenance').click()
    expect(page.get_by_text('scb_yolov9e.pt', exact=True)).to_be_visible()
    assert any(item['sha256'] in page.locator('body').inner_text() for item in raw[0]['provenance']['models'])
    page.locator('summary').filter(has_text='Model provenance').click()
    expect(page.locator('img').first).to_be_visible()
    assert page.locator('img').first.evaluate('(el)=>el.complete && el.naturalWidth>0')
    page.get_by_label('Lesson playback progress').evaluate("el=>{el.value='100';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    expect(page.locator('.time-now')).to_have_text(last_time)
    with page.expect_download() as download_info:
        page.get_by_role('button', name='Export current snapshot', exact=True).click()
    exported = args.output/'browser-export.json'
    download_info.value.save_as(exported)
    record = json.loads(exported.read_text(encoding='utf-8'))
    assert record['snapshots'][0] == raw[-1], 'Browser export changed canonical source evidence'
    page.get_by_label('Choose inference snapshot JSON').set_input_files(exported)
    expect(page.get_by_text('Actual inference records', exact=True)).to_be_visible()
    conflict = {'schema_version':'2.0','data_source':'pretrained_inference',
                'snapshots':[{**raw[0],'data_source':'synthetic_demo'}]}
    page.get_by_label('Choose inference snapshot JSON').set_input_files({
        'name':'conflict.json','mimeType':'application/json','buffer':json.dumps(conflict).encode()})
    expect(page.get_by_role('alert')).to_contain_text('Inconsistent data sources')
    expect(page.get_by_role('alert')).to_contain_text('Previous data retained')
    expect(page.get_by_text('Actual inference records', exact=True)).to_be_visible()
    page.goto(args.url.split('?')[0] + '?demo=1', wait_until='networkidle')
    expect(page.get_by_text('Demo mode', exact=True)).to_be_visible(timeout=30000)
    page.goto(args.url, wait_until='networkidle')
    expect(page.get_by_label('Rule index variance: Unknown')).to_be_visible()
    page.screenshot(path=str(args.output/'Actual inference-Desktop.png'), full_page=True)
    page.get_by_label('Lesson playback progress').evaluate("el=>{el.value='100';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    expect(page.locator('.time-now')).to_have_text(last_time)
    page.screenshot(path=str(args.output/'Actual inference-Last frame.png'), full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth + 1')
    page.screenshot(path=str(args.output/'Actual inference-Mobile.png'), full_page=True)
    assert not errors, errors
    browser.close()
report={'passed':True,'source':str(args.json),'url':args.url,'frames':len(raw),
        'checks':['real source label','first frame unknown variance','empty knowledge graph',
                  'model provenance visible','same-origin frame image','seek to actual last timestamp',
                  'browser export deep-equals raw frame','reimport','conflicting demo rejection retains data',
                  'explicit demo URL only','390px layout without overflow'],'page_errors':errors}
(args.output/'browser-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
