"""Actual API/browser verification against the official MIT video/PDF pair.

This imports the given file and tests reversible manual mappings. It does not
mock the API/model, edit source media, or claim alignment accuracy.
"""
import argparse
import json
import time
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:5173')
parser.add_argument('--lesson', default='mit-optimization-real')
parser.add_argument('--file', required=True, type=Path)
parser.add_argument('--output', type=Path, default=Path('output/playwright/courseware-real'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
endpoint = args.url + '/api/review/lessons/' + args.lesson
errors, answers, checks = [], [], []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('response', lambda response: answers.append(response.json()) if response.url.endswith('/api/review/chat') and response.ok else None)
    page.goto(args.url + '/?lesson=' + args.lesson, wait_until='networkidle')
    expect(page.locator('.review-lesson-strip h2')).to_contain_text('MIT')
    expect(page.get_by_role('heading', name='Uploaded courseware alignment', exact=True)).to_be_visible()
    original = requests.get(endpoint, timeout=10).json()
    page.get_by_label('Import courseware', exact=True).set_input_files(args.file.resolve())
    expect(page.locator('.cw-message').filter(has_text='Courseware imported')).to_be_visible(timeout=350000)
    data = requests.get(endpoint, timeout=10).json()
    assert data['transcript'] == original['transcript'], 'Import must never overwrite native ASR'
    assert len(data['courseware']['pages']) == 31
    assert data['courseware']['sha256'] == 'cf9ac01c099a3d3062493c692efbfd75554c16357c240f3b4b4e644537c6a345'
    (args.output/'automatic-alignment.json').write_text(json.dumps(data['courseware'], ensure_ascii=False, indent=2), encoding='utf-8')
    checks.append('Real official PDF uploaded through browser; 31 rendered pages; raw ASR unchanged')
    unmatched = next(m for m in data['courseware']['matches'] if m['status'] == 'unmatched')
    page.get_by_label('Whole-lesson playback progress').evaluate('(el, t) => { el.value=String(t); el.dispatchEvent(new Event("input",{bubbles:true})); }', unmatched['start_sec'])
    expect(page.get_by_text('No uploaded courseware page is matched to the current time. Replay the video to confirm or browse the source pages manually.', exact=True)).to_be_visible()
    assert page.locator('.cw-preview img').count() == 0
    checks.append('Follow mode shows explicit absence at unmatched intervals, never the first page')
    page.get_by_label('Preview uploaded courseware pages', exact=True).select_option(data['courseware']['pages'][10]['id'])
    image = page.locator('.cw-preview img')
    expect(image).to_have_attribute('alt', 'Uploaded courseware page 11')
    page.wait_for_function('document.querySelector(".cw-preview img").naturalWidth > 0')
    match = next(m for m in data['courseware']['matches'] if m['start_sec'] <= 600 < m['end_sec'])
    index = data['courseware']['matches'].index(match)
    page.locator('.cw-match').nth(index).click()
    expect(page.get_by_label('Whole-lesson playback progress')).to_have_value(f"{match['start_sec']:g}")
    page.get_by_label('Assign a courseware page to an observed interval').select_option('')
    page.get_by_role('button', name='Save page confirmation', exact=True).click()
    expect(page.locator('.cw-message').filter(has_text='Saved the page confirmation for this observed interval')).to_be_visible(timeout=30000)
    rejected = requests.get(endpoint, timeout=10).json()
    assert next(m for m in rejected['courseware']['matches'] if m['id'] == match['id'])['status'] == 'rejected'
    assert not any(match['id'] in e.get('match_ids', [e.get('match_id')]) for e in rejected['courseware_evidence'])
    page.get_by_label('Assign a courseware page to an observed interval').select_option(data['courseware']['pages'][10]['id'])
    with page.expect_response(lambda response: response.url.endswith('/courseware/matches') and response.status == 200):
        page.get_by_role('button', name='Save page confirmation', exact=True).click()
    expect(page.get_by_role('button', name='Save page confirmation', exact=True)).to_be_enabled(timeout=30000)
    confirmed = requests.get(endpoint, timeout=10).json()
    assert next(m for m in confirmed['courseware']['matches'] if m['id'] == match['id'])['status'] == 'confirmed'
    checks.append('Observed interval seek, reject removes timed evidence, manual confirmed page recomputes overlay')
    page.get_by_role('button', name='Current segment', exact=True).click()
    page.get_by_label('Ask the review assistant').fill('In the current courseware, what is the knapsack problem  ?Explain using the courseware and audio. and cite evidence.')
    page.get_by_role('button', name='Send lesson question', exact=True).click()
    expect(page.locator('.model-answer')).to_be_visible(timeout=190000)
    assert answers and answers[-1]['source'] == 'local_llm'
    assert any(c['kind'] == 'courseware' for c in answers[-1]['citations']), answers[-1]
    citation = page.locator('.answer-citations button').filter(has_text='courseware').first
    citation.click()
    page.wait_for_function('Math.abs(document.querySelector("video").currentTime - Number(document.querySelector("input[type=range]").value)) < .5')
    checks.append('Actual local model references standalone courseware; citation seeks actual video')
    # The test exercised a manual state; re-import restores automatic decisions.
    page.get_by_role('button', name='Remove association', exact=True).click()
    expect(page.get_by_role('button', name='Import PPT / PDF', exact=True)).to_be_visible(timeout=30000)
    detached = requests.get(endpoint, timeout=10).json()
    assert 'courseware' not in detached and detached['transcript'] == original['transcript']
    page.get_by_label('Import courseware', exact=True).set_input_files(args.file.resolve())
    expect(page.locator('.cw-message').filter(has_text='Courseware imported')).to_be_visible(timeout=350000)
    final = requests.get(endpoint, timeout=10).json()
    assert all(m['status'] != 'confirmed' for m in final['courseware']['matches'])
    checks.append('Detach restores raw base; re-import restores automatic alignment without deleting originals')
    page.get_by_label('Preview uploaded courseware pages', exact=True).select_option(final['courseware']['pages'][10]['id'])
    page.get_by_label('Whole-lesson playback progress').evaluate("el => { el.value='600'; el.dispatchEvent(new Event('input',{bubbles:true})); }")
    with page.expect_download() as download:
        page.get_by_role('button', name='Full data', exact=True).click()
    download.value.save_as(args.output/'lesson-export.json')
    assert json.loads((args.output/'lesson-export.json').read_text(encoding='utf-8')) == final
    with page.expect_download() as download:
        page.get_by_role('button', name='Markdown', exact=True).click()
    download.value.save_as(args.output/'lesson-report.md')
    assert final['courseware']['sha256'] in (args.output/'lesson-report.md').read_text(encoding='utf-8')
    checks.append('Exports preserve full source, mapping status, original text and revision lineage')
    page.locator('#courseware').screenshot(path=str(args.output/'courseware-desktop.png'))
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile horizontal overflow'
    page.locator('#courseware').screenshot(path=str(args.output/'courseware-mobile.png'))
    checks.append('390px layout fits viewport')
    assert not errors, errors
    result = {'passed': True, 'scope': 'actual_official_video_pdf_actual_api_actual_model',
              'lesson_id': args.lesson, 'checks': checks, 'page_errors': errors, 'model_answers': answers,
              'automatic_matches': {status: sum(m['status'] == status for m in final['courseware']['matches'])
                                    for status in ('auto_matched','ambiguous','unmatched')},
              'transcript_revisions': len(final['transcript_revisions']), 'accuracy_evaluated': False}
    (args.output/'browser-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'model_answers'}, ensure_ascii=False))
    browser.close()
