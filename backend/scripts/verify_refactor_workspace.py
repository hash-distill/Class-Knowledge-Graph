"""Browser acceptance on an isolated copy of real classroom evidence.

Teacher entries are explicitly labelled acceptance fixtures. Never point this
script at a teacher's production workspace database.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests
from playwright.sync_api import sync_playwright,expect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:18092')
    parser.add_argument('--lesson',default='unified-chemistry-001-9')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=args.url.rstrip('/');endpoint=f'/api/review/lessons/{args.lesson}/workspace'
    checks=[];errors=[]

    def workspace():
        response=requests.get(base+endpoint,timeout=10);response.raise_for_status();return response.json()

    with sync_playwright() as pw:
        browser=pw.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(base+f'/?lesson={args.lesson}&view=teaching',wait_until='networkidle')
        expect(page.get_by_test_id('teacher-workspace')).to_be_visible()
        expect(page.get_by_test_id('teacher-goals')).to_be_enabled()
        page.wait_for_function('document.querySelector("video").readyState>=2')
        page.evaluate('window.__refactorVideo=document.querySelector("video");window.__refactorVideo.currentTime=5')
        page.wait_for_function('Math.abs(document.querySelector("video").currentTime-5)<.2')

        def save(button,path,method='POST'):
            with page.expect_response(lambda r:urlsplit(r.url).path==endpoint+path and r.request.method==method) as result:
                page.get_by_test_id(button).click()
            response=result.value
            assert response.status==200,(path,response.status,response.text())
            state=response.json()
            expect(page.get_by_test_id('teacher-workspace')).to_contain_text('记录版本 '+str(state['version']))
            return state

        page.get_by_test_id('teacher-goals').fill('验收样例：核对来源并记录改进结果')
        save('save-teacher-goals','/goals')
        checks.append('Goals saved through browser to SQLite')
        page.get_by_test_id('teacher-transcript-select').select_option('T0001')
        page.get_by_test_id('teacher-correction-text').fill('验收样例：这是独立核对稿，不替换真实课堂原始转写。')
        save('save-teacher-correction','/corrections')
        expect(page.get_by_test_id('revert-teacher-correction')).to_be_visible()
        page.wait_for_function('document.querySelector("video")===window.__refactorVideo')
        assert abs(page.locator('video').evaluate('v=>v.currentTime')-5)<.2
        assert workspace()['corrections'][0]['replacement'].startswith('验收样例')
        checks.append('Teacher overlay saved without replacing player, raw transcript or playback time')

        title='验收样例：下一课核对术语 '+str(int(time.time()))
        page.locator('[data-teaching-tab="actions"]').click()
        page.get_by_test_id('teacher-action-title').fill(title)
        page.get_by_test_id('teacher-action-evidence').select_option('T0001')
        state=save('save-teacher-action','/actions')
        action=next(row for row in state['actions'] if row['title']==title)
        page.get_by_test_id('teacher-action-outcome-'+action['id']).fill('验收样例：实际结果录入测试，不作为教学效果结论。')
        with page.expect_response(lambda r:r.url.endswith('/actions/'+action['id']) and r.request.method=='PATCH') as result:
            page.get_by_test_id('complete-teacher-action-'+action['id']).click()
        assert result.value.status==200,result.value.text()
        expect(page.get_by_test_id('teacher-action-'+action['id'])).to_contain_text('已完成')
        checks.append('Follow-up action only completes with a recorded outcome')

        page.locator('[data-teaching-tab="practice"]').click()
        page.get_by_test_id('teacher-assessment-question').fill('验收样例：请写出课堂中的一个术语。')
        page.get_by_test_id('teacher-assessment-reference').fill('验收参考答案，供软件流程检查。')
        page.get_by_test_id('teacher-assessment-evidence').select_option('T0001')
        state=save('save-teacher-assessment','/assessments')
        assessment=state['assessments'][-1]
        page.get_by_test_id('teacher-attempt-answer-'+assessment['id']).fill('验收样例：实际作答录入测试。')
        page.get_by_test_id('teacher-attempt-score-'+assessment['id']).fill('0')
        with page.expect_response(lambda r:r.url.endswith('/assessments/'+assessment['id']+'/attempts') and r.request.method=='POST') as result:
            page.get_by_test_id('save-teacher-attempt-'+assessment['id']).click()
        assert result.value.status==200,result.value.text()
        assert result.value.json()['attempts'][-1]['score']==0
        expect(page.get_by_test_id('teacher-recorded-attempt').last).to_contain_text('实际作答录入测试')
        checks.append('Actual response saves numeric zero correctly; missing score remains a separate state')

        page.locator('[data-teaching-tab="review"]').click()
        save('revert-teacher-correction','/corrections/T0001',method='DELETE')
        assert not workspace()['corrections']
        expect(page.get_by_test_id('teacher-correction-text')).not_to_have_value('验收样例：这是独立核对稿，不替换真实课堂原始转写。')
        checks.append('Correction can be revoked while audit history remains')
        page.locator('nav [data-view="knowledge"]').click()
        page.locator('nav [data-view="teaching"]').click()
        assert page.locator('video').count()==1
        assert page.evaluate('document.querySelector("video")===window.__refactorVideo')
        page.reload(wait_until='networkidle')
        expect(page.get_by_test_id('teacher-goals')).to_have_value('验收样例：核对来源并记录改进结果')
        page.locator('[data-teaching-tab="practice"]').click()
        expect(page.get_by_test_id('teacher-recorded-attempt').last).to_contain_text('实际作答录入测试')
        checks.append('Saved records survive page reload and view navigation')
        page.screenshot(path=str(args.output/'teacher-workspace-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        page.screenshot(path=str(args.output/'teacher-workspace-mobile.png'),full_page=True)
        checks.append('390px view has no horizontal overflow')
        browser.close()
    assert not errors,errors
    report={'checks':checks,'page_errors':errors,'workspace':workspace(),
            'scope':'Isolated acceptance DB with explicitly labelled teacher fixtures; no model accuracy evaluation.'}
    (args.output/'browser-workspace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'checks':checks,'page_errors':errors},ensure_ascii=False))


if __name__=='__main__':main()
