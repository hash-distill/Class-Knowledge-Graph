"""Repeatable v3 browser verification. --fixture uses browser-only contract mocks; default uses actual API."""
import argparse
import json
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:5173')
parser.add_argument('--output', type=Path, default=Path('output/playwright/lesson-review'))
parser.add_argument('--fixture', type=Path)
parser.add_argument('--ask', action='store_true', help='Call the actual model during real API verification')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
mock = args.fixture is not None
if mock:
    data = json.loads(args.fixture.read_text(encoding='utf-8'))
else:
    with urllib.request.urlopen(args.url + '/api/review/lessons') as request:
        lessons = json.load(request)['lessons']
    assert lessons, 'Actual backend contains no lessons'
    with urllib.request.urlopen(args.url + '/api/review/lessons/' + lessons[0]['id']) as request:
        data = json.load(request)
errors, checks, chat_requests, model_answers = [], [], [], []

def fulfill(route, payload, status=200):
    route.fulfill(status=status, content_type='application/json', body=json.dumps(payload, ensure_ascii=False))

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1060}, device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    def record_model_response(response):
        if response.url.endswith('/api/review/chat') and response.ok:
            model_answers.append(response.json())
    page.on('response', record_model_response)
    if mock:
        page.route('**/api/review/lessons', lambda route: fulfill(route, {'lessons': [data['lesson']]}))
        page.route('**/api/review/lessons/' + data['lesson']['id'], lambda route: fulfill(route, data))
        page.route('**/api/review/media/**', lambda route: route.fulfill(status=404))
        def chat_route(route):
            chat_requests.append(route.request.post_data_json)
            if len(chat_requests) == 1:
                fulfill(route, {'error': {'code': 'MODEL_UNAVAILABLE', 'message': '合同测试：模型服务不可用'}}, 503)
            else:
                t = data['transcript'][0]
                fulfill(route, {'answer': '合同响应（非真实课堂分析）。', 'source': 'local_llm', 'model': 'contract-fixture-model', 'citations': [{**t, 'kind': 'transcript'}], 'evidence_ids': [t['id']], 'request_id': 'contract-request'})
        page.route('**/api/review/chat', chat_route)
    page.goto(args.url, wait_until='networkidle')
    expect(page.get_by_text('真实课堂', exact=True)).to_be_visible()
    expect(page.locator('.review-lesson-strip h2')).to_have_text(data['lesson']['title'])
    assert page.get_by_label('整堂课回放进度').get_attribute('max') == str(data['lesson']['duration_sec'])
    if not data['snapshots']:
        expect(page.get_by_text('本课没有已确认同步的行为录像', exact=True)).to_be_visible()
    checks.append('Actual duration and explicit missing behavior state independent of snapshots')
    segment = data['transcript'][len(data['transcript']) // 2]
    page.locator('.transcript-item').filter(has_text=segment['id']).click()
    expect(page.get_by_label('整堂课回放进度')).to_have_value(str(segment['start_sec']))
    expect(page.locator('.current-speech')).to_contain_text(segment['text'])
    active_slides = [s for s in data['slides'] if s['start_sec'] <= segment['start_sec'] < s['end_sec']]
    if active_slides:
        expect(page.locator('.slide-preview-caption strong')).to_have_text(active_slides[0]['title'] or active_slides[0]['id'])
    if not mock:
        video = page.locator('video')
        page.wait_for_function('document.querySelector("video").readyState >= 1')
        page.wait_for_function('(t) => Math.abs(document.querySelector("video").currentTime - t) < .5', arg=segment['start_sec'])
        assert video.evaluate('v => Number.isFinite(v.duration) && v.duration > 0')
        if active_slides and active_slides[0]['image_url']:
            assert page.locator('.slide-preview img').evaluate('image => image.complete && image.naturalWidth > 0')
        checks.append('Actual video metadata, media seek and actual slide image')
    checks.append('Transcript click synchronizes exact timestamp and current slide')
    if mock:
        page.get_by_label('整堂课回放进度').evaluate("el => { el.value = '200'; el.dispatchEvent(new Event('input', {bubbles:true})); }")
        expect(page.get_by_text('当前时刻没有对应的转写片段，可播放原录像核对。')).to_be_visible()
        assert page.locator('.graph-node.current').count() == 0
        page.get_by_role('button', name='知识节点 概念A，2 条来源，点击定位', exact=True).click()
        expect(page.locator('.concept-intervals button')).to_have_count(2)
        expect(page.get_by_label('整堂课回放进度')).to_have_value('3.21')
        page.locator('.concept-intervals button').filter(has_text='T0002').click()
        expect(page.get_by_label('整堂课回放进度')).to_have_value('400.35')
        checks.append('Repeated concept preserves two disjoint intervals and no evidence in middle gap')
    else:
        if data['graph']['nodes']:
            page.locator('.graph-node').first.click()
            expect(page.locator('.concept-evidence')).to_be_visible()
            if page.locator('.concept-intervals button').count():
                page.locator('.concept-intervals button').last.click()
                checks.append('Actual graph node and individual source interval seek')
    if mock or args.ask:
        page.get_by_role('button', name='当前片段', exact=True).click()
        page.get_by_label('向课堂复盘助手提问').fill('当前片段在讲什么？请引用课堂证据。')
        page.get_by_role('button', name='发送课堂问题', exact=True).click()
        if mock:
            expect(page.get_by_text('本次回答未生成', exact=True)).to_be_visible()
            assert page.locator('.model-answer').count() == 0
            expect(page.get_by_text('合同测试：模型服务不可用', exact=True)).to_be_visible()
            page.get_by_role('button', name='重试这条问题', exact=True).click()
            assert chat_requests[0]['scope'] == 'current'
            assert chat_requests[0]['lesson_id'] == data['lesson']['id']
            assert chat_requests[0]['at_sec'] == 400.35
            checks.append('503 model error has no fabricated answer; retry preserves question scope/time')
        expect(page.locator('.model-answer')).to_be_visible(timeout=190000)
        expect(page.locator('.answer-source b')).not_to_be_empty()
        citation = page.locator('.answer-citations button').first
        expect(citation).to_be_visible()
        citation.click()
        if mock:
            expect(page.get_by_label('整堂课回放进度')).to_have_value('3.21')
        else:
            page.wait_for_function('Math.abs(document.querySelector("video").currentTime - Number(document.querySelector("input[aria-label=整堂课回放进度]").value)) < .5')
        checks.append('Model source/model label visible and citation seeks video timeline')
    with page.expect_download() as download:
        page.get_by_role('button', name='完整数据', exact=True).click()
    export_path = args.output / 'lesson-export.json'
    download.value.save_as(export_path)
    assert json.loads(export_path.read_text(encoding='utf-8')) == data
    with page.expect_download() as download:
        page.get_by_role('button', name='复盘文稿', exact=True).click()
    markdown_path = args.output / 'lesson-report.md'
    download.value.save_as(markdown_path)
    markdown = markdown_path.read_text(encoding='utf-8')
    assert data['lesson']['source_video']['sha256'] in markdown
    assert all(limit in markdown for limit in data['report']['limitations'])
    checks.append('JSON preserves entire source package; Markdown preserves source hash and limitations')
    page.evaluate('window.scrollTo(0, 0)')
    page.screenshot(path=str(args.output / ('contract-desktop.png' if mock else 'real-desktop.png')), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth + 1')
    page.evaluate('window.scrollTo(0, 0)')
    page.screenshot(path=str(args.output / ('contract-mobile.png' if mock else 'real-mobile.png')), full_page=True)
    checks.append('390px responsive layout has no horizontal overflow')
    if mock:
        page.unroute('**/api/review/lessons')
        page.route('**/api/review/lessons', lambda route: fulfill(route, {'lessons': []}))
        page.reload(wait_until='networkidle')
        expect(page.get_by_text('还没有可复盘的真实课堂', exact=True)).to_be_visible()
        assert page.locator('video').count() == 0
        assert page.locator('.review-lesson-strip').count() == 0
        checks.append('Empty real lesson catalog never falls back to demo')
    assert not errors, errors
    browser.close()
report = {'passed': True, 'scope': 'browser_contract_mock_not_actual_model' if mock else 'actual_api_actual_model' if args.ask else 'actual_api_no_model_call', 'url': args.url, 'lesson_id': data['lesson']['id'], 'duration_sec': data['lesson']['duration_sec'], 'checks': checks, 'page_errors': errors, 'model_answers': model_answers}
(args.output / 'browser-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
