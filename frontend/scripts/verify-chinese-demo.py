"""Chinese-first demonstration acceptance using actual API, source media and LLM."""
import argparse
import json
import re
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:5175')
p.add_argument('--output',type=Path,default=Path('output/playwright/chinese-demo-real'))
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
def get(path):
    with urllib.request.urlopen(args.url+path) as response:return json.load(response)
catalog=get('/api/review/lessons')['lessons']
assert catalog[0]['id']=='lesson1-real'
assert all(row['id']!='mit-optimization-real' for row in catalog)
assert all(re.search(r'[\u4e00-\u9fff]',row['title']) for row in catalog)
data=get('/api/review/lessons/ts-chinese-001-2');a=data['activity_analysis']
assert not data['transcript'] and data['lesson']['modalities']['audio']=='unrecognized'
checks=['Chinese lesson is first; English archived lesson is absent from demonstration catalog']
errors=[];answers=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    def capture(response):
        if response.url.endswith('/api/review/chat'):answers.append({'status':response.status,'body':response.json()})
    page.on('response',capture)
    page.goto(args.url,wait_until='networkidle')
    expect(page.locator('.review-lesson-strip h2')).to_have_text('正弦定理 · 课堂复盘')
    assert page.locator('.transcript-item').count()>50
    page.wait_for_function('document.querySelector("video").readyState>=2&&!document.querySelector("video").error')
    page.get_by_label('向课堂复盘助手提问').fill('这堂中文数学课如何引入正弦定理？请用中文回答并引用课堂语音证据。')
    page.get_by_role('button',name='发送课堂问题',exact=True).click()
    expect(page.locator('.model-answer')).to_be_visible(timeout=190000)
    assert answers[-1]['status']==200 and answers[-1]['body']['citations'],answers
    assert any(c['kind']=='transcript' for c in answers[-1]['body']['citations'])
    page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(args.output/'chinese-main-desktop.png'))
    checks.append('Primary Chinese mathematics lesson has real source speech and actual cited Chinese model answers')
    page.get_by_label('当前课程',exact=True).select_option('ts-chinese-001-2')
    expect(page.locator('.review-lesson-strip h2')).to_have_text(data['lesson']['title'])
    expect(page.locator('#evidence')).to_contain_text('本片段未识别到可靠语音')
    expect(page.locator('.slide-list')).to_contain_text('北国的槐树')
    expect(page.get_by_role('tab',name=re.compile('课件页段'))).to_have_attribute('aria-selected','true')
    visible=page.locator('body').inner_text()
    for old in ['CLASSROOM INSIGHTS','WORKSPACE','OBSERVE. CONNECT. REFLECT.','THIS LESSON','LESSON REVIEW','MIT 优化问题',' · complete',' · video_ocr']:
        assert old not in visible,old
    assert not re.search(r'[A-Za-z]{2,}',page.locator('.slide-list').inner_text())
    checks.append('Chinese interface, clean Chinese display text and explicit unrecognized-speech notice')
    page.wait_for_function('document.querySelector("video").readyState>=2&&!document.querySelector("video").error')
    assert page.locator('.slide-preview img').evaluate('im=>im.complete&&im.naturalWidth>0')
    checks.append('Actual Chinese source video and matching courseware image decode')
    valid=[f for f in a['frames'] if f['participation_index'] is not None]
    target=min(valid,key=lambda f:abs(f['timestamp_sec']-40))
    behavior=page.locator('#behavior')
    behavior.get_by_role('button',name='定位行为帧 '+target['id']+' ',exact=False).click()
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.5',arg=target['timestamp_sec'])
    expect(behavior.locator('.current-observation')).to_contain_text('规则参与指数')
    behavior.screenshot(path=str(args.output/'chinese-behavior.png'))
    checks.append('Chinese classroom behavior is real; clicking an observation seeks original media')
    page.get_by_role('button',name='整堂课',exact=True).click()
    page.get_by_label('向课堂复盘助手提问').fill('这段课文通过哪些景物和感受写秋？请用中文回答并引用课件证据。没有可靠语音时请说明，不要补写教师台词。')
    page.get_by_role('button',name='发送课堂问题',exact=True).click()
    expect(page.locator('.model-answer')).to_be_visible(timeout=190000)
    assert answers and answers[-1]['status']==200,answers
    answer=answers[-1]['body']
    assert len(re.findall(r'[\u4e00-\u9fff]',answer['answer']))>20
    citations=[row for row in answer['citations'] if row['kind']=='slide']
    assert citations,answer
    citation=citations[0]
    page.locator('.answer-citations button').filter(has_text=citation['id']+' ·').click()
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.5',arg=citation['start_sec'])
    expect(page.locator('.answer-citations')).to_contain_text('课件画面')
    checks.append('Real local LLM answers in Chinese from courseware evidence and citations seek the original video')
    with page.expect_download() as download:page.get_by_role('button',name='完整数据',exact=True).click()
    path=args.output/'lesson-export.json';download.value.save_as(path)
    assert json.loads(path.read_text(encoding='utf-8'))==data
    checks.append('Chinese display does not rewrite raw recognized evidence or exported machine metadata')
    page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(args.output/'chinese-desktop.png'))
    page.locator('.model-answer').screenshot(path=str(args.output/'chinese-answer.png'))
    page.set_viewport_size({'width':390,'height':844});page.evaluate('window.scrollTo(0,0)')
    page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(args.output/'chinese-mobile.png'))
    checks.append('390px Chinese layout has no horizontal overflow')
    assert not errors,errors
    browser.close()
report={'passed':True,'scope':'actual_chinese_catalog_media_and_local_llm_no_mocks','url':args.url,'lesson_id':data['lesson']['id'],'checks':checks,'page_errors':errors,'answers':answers}
(args.output/'browser-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
