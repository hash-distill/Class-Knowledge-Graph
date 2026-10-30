"""Real API, real media, real local LLM acceptance. No mocked routes."""
import argparse
import json
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:5173')
p.add_argument('--lesson', default='ts-chemistry-001-9')
p.add_argument('--output', type=Path, default=Path('output/playwright/trained-activity-real'))
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(args.url+'/api/review/lessons/'+args.lesson) as response:
    data = json.load(response)
a = data['activity_analysis']
assert a['source_sha256'] == data['lesson']['source_video']['sha256']
valid = [f for f in a['frames'] if f['participation_index'] is not None]
unknown = [f for f in a['frames'] if f['status'] == 'no_people']
assert valid and unknown, 'Acceptance lesson must exercise both actual valid and unknown states'
checks, errors, answers = [], [], []
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={'width':1440,'height':1000}, device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    def capture(response):
        if response.url.endswith('/api/review/chat'):
            answers.append({'status':response.status,'body':response.json()})
    page.on('response', capture)
    page.goto(args.url+'/?lesson='+args.lesson, wait_until='networkidle')
    expect(page.locator('.review-lesson-strip h2')).to_have_text(data['lesson']['title'])
    behavior = page.locator('#behavior')
    expect(behavior.get_by_role('heading', name='学生行为分析', exact=True)).to_be_visible()
    expect(behavior.locator('.participation-dot')).to_have_count(len(valid))
    checks.append('Actual source package and valid-score dots loaded without mock routes')
    target = min(valid, key=lambda f:abs(f['timestamp_sec']-40))
    behavior.get_by_role('button', name='定位行为帧 '+target['id']+' ', exact=False).click()
    expect(page.get_by_label('整堂课回放进度')).to_have_value(str(target['timestamp_sec']).removesuffix('.0'))
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.5', arg=target['timestamp_sec'])
    page.wait_for_function('document.querySelector("video").readyState>=2 && !document.querySelector("video").error')
    assert not page.get_by_text('课堂录像暂时无法访问。时间定位与文本证据仍可使用。',exact=True).is_visible()
    expect(behavior.locator('.current-observation')).to_contain_text(str(target['known_people']))
    assert behavior.locator('.track-table td button:not(.unknown)').count() > 0
    header=behavior.locator('th[data-frame-id="'+target['id']+'"]')
    assert header.evaluate('el=>{const c=el.closest(".track-scroll").getBoundingClientRect(),r=el.getBoundingClientRect();return r.left>=c.left&&r.right<=c.right}')
    checks.append('Valid-score time click seeks real video and updates scene-local track heatmap')
    behavior.scroll_into_view_if_needed()
    behavior.screenshot(path=str(args.output/'real-behavior.png'))
    gap = unknown[len(unknown)//2]
    behavior.get_by_role('button', name='定位行为帧 '+gap['id']+' ', exact=False).click()
    expect(behavior.locator('.current-observation')).to_contain_text('规则参与指数 未知')
    expect(behavior.locator('.current-observation')).to_contain_text('未知（无可用检测）')
    checks.append('Actual no-people camera interval remains unknown rather than zero')
    behavior.locator('.behavior-provenance summary').click()
    expect(behavior.locator('.behavior-provenance')).to_contain_text(a['source_sha256'])
    expect(behavior.locator('.behavior-provenance')).to_contain_text('仅限离线研究')
    expect(behavior.locator('.behavior-limits')).to_contain_text('训练组')
    checks.append('Training-source demonstration and offline-only ARIC provenance visible')
    source = behavior.locator('.source-frame').nth(2)
    source.click()
    frame = a['frames'][2]
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.5', arg=frame['timestamp_sec'])
    assert source.locator('img').evaluate('im=>im.complete&&im.naturalWidth>0')
    checks.append('Actual saved source image loads and clicking it seeks original media')
    with page.expect_download() as d:
        page.get_by_role('button',name='完整数据',exact=True).click()
    path=args.output/'lesson-export.json';d.value.save_as(path)
    assert json.loads(path.read_text(encoding='utf-8')) == data
    with page.expect_download() as d:
        page.get_by_role('button',name='复盘文稿',exact=True).click()
    path=args.output/'lesson-report.md';d.value.save_as(path)
    markdown=path.read_text(encoding='utf-8')
    assert a['source_sha256'] in markdown
    assert all(limit in markdown for limit in a['limitations'])
    checks.append('Browser JSON export preserves whole bundle; Markdown retains same-source hash and all activity limits')
    behavior.get_by_role('button',name='定位行为帧 '+target['id']+' ',exact=False).click()
    page.get_by_role('button',name='当前片段',exact=True).click()
    page.get_by_label('向课堂复盘助手提问').fill('当前画面有哪些可见行为？请引用行为证据，并说明未知对象与规则分数覆盖率；不要推断知识掌握或真实积极度。')
    page.get_by_role('button',name='发送课堂问题',exact=True).click()
    expect(page.locator('.model-answer')).to_be_visible(timeout=190000)
    assert answers and answers[-1]['status']==200, answers
    answer=answers[-1]['body']
    citations=[c for c in answer['citations'] if c['kind']=='behavior']
    assert citations, answer
    c=citations[0]
    page.locator('.answer-citations button').filter(has_text=c['id']+' ·').click()
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.5',arg=c['start_sec'])
    checks.append('Real local LLM returns behavior citations and citation click seeks original video')
    assert page.locator('video').evaluate('v=>v.readyState>=2&&!v.error')
    page.locator('.assistant-panel, .lesson-assistant').first.screenshot(path=str(args.output/'real-answer.png')) if page.locator('.assistant-panel, .lesson-assistant').count() else page.locator('.model-answer').screenshot(path=str(args.output/'real-answer.png'))
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(args.output/'real-desktop.png'))
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(args.output/'real-mobile.png'))
    checks.append('390px responsive page has no horizontal overflow')
    page.goto(args.url+'/?lesson=ts-math-006-32',wait_until='networkidle')
    expect(page.locator('#behavior .block-empty').filter(has_text='没有可计算规则指数')).to_be_visible()
    assert page.locator('#behavior .participation-dot').count()==0
    checks.append('Second real classroom with unknown temporal scores does not fabricate chart points')
    assert not errors, errors
    browser.close()
report={'passed':True,'scope':'actual_api_media_and_local_llm_no_mocks','lesson_id':args.lesson,'checks':checks,'page_errors':errors,'answers':answers}
(args.output/'browser-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
