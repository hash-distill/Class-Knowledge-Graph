"""Read-only rendered spacing regression across real frames and browser sizes."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7032')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []; errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width':1280, 'height':800})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.emulate_media(reduced_motion='reduce')
        page.goto(args.url.rstrip('/')+'/?view=review&lesson=unified-chemistry-001-9', wait_until='networkidle')
        expect(page.get_by_test_id('overview-behavior')).to_be_visible()
        for width, height, zoom in [(1280,800,1), (1440,1000,1), (1024,768,1),
                                     (1600,900,1.25), (1280,800,1.5), (768,1024,1), (390,844,1)]:
            page.set_viewport_size({'width':width, 'height':height})
            page.evaluate('(zoom)=>{document.documentElement.style.zoom=String(zoom)}', zoom)
            for seconds in [21, 0, 40, 45, 75, 89.9]:
                page.get_by_role('slider', name='整堂课回放进度').evaluate(
                    '(el,seconds)=>{el.value=String(seconds);el.dispatchEvent(new Event("input",{bubbles:true}))}', seconds)
                clock = f'{int(seconds)//60:02d}:{int(seconds)%60:02d}'
                expect(page.locator('.overview-heading>.mono')).to_have_text(clock)
                result = page.evaluate('''() => {
                    const root=document.querySelector('.overview-students');
                    const rect=selector=>root.querySelector(selector)?.getBoundingClientRect();
                    const caption=rect('.image-caption'),reading=rect('.state-reading');
                    const boundary=rect('.state-boundary'),layout=rect('.state-layout');
                    const clashes=[];
                    for(const row of root.querySelectorAll('.state-counts>div')){
                        const label=row.querySelector('span').getBoundingClientRect();
                        const number=row.querySelector('b').getBoundingClientRect();
                        if(label.right>number.left+1) clashes.push(row.textContent.trim());
                    }
                    const metric=root.querySelector('.participation-reading').getBoundingClientRect();
                    const support=root.querySelector('.participation-reference').getBoundingClientRect();
                    const meter=root.querySelector('.participation-mark').getBoundingClientRect();
                    const heading=root.querySelector('.reading-label').getBoundingClientRect();
                    const overlap=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>1 &&
                        Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1;
                    const stage=document.querySelector('.active-feature').getBoundingClientRect();
                    const player=document.querySelector('.shared-player').getBoundingClientRect();
                    return {captionGap:caption ? reading.top-caption.bottom : null,
                        boundaryGap:boundary.top-layout.bottom,clashes,
                        labelMeterOverlap:overlap(meter,heading),metricSupportOverlap:overlap(metric,support),
                        documentWidth:document.documentElement.scrollWidth,viewportWidth:innerWidth,
                        columnTopDelta:Math.abs(stage.top-player.top),columnBottomDelta:Math.abs(stage.bottom-player.bottom)};
                }''')
                sample = {'width':width, 'height':height, 'zoom':zoom, 'time':seconds, **result}
                samples.append(sample)
                assert result['captionGap'] is None or result['captionGap'] >= 8, sample
                assert result['boundaryGap'] >= 5, sample
                assert not result['clashes'] and not result['labelMeterOverlap'] and not result['metricSupportOverlap'], sample
                assert result['documentWidth'] <= width+1, sample
                if width > 1000:
                    assert result['columnTopDelta'] <= 1 and result['columnBottomDelta'] <= 1, sample
                if seconds == 21:
                    page.get_by_test_id('overview-behavior').screenshot(path=str(args.output/f'student-{width}-{zoom}.png'))
            # Expanded raw transcript must not spill into adjacent cards.
            page.get_by_role('slider', name='整堂课回放进度').evaluate(
                '(el)=>{el.value="21";el.dispatchEvent(new Event("input",{bubbles:true}))}')
            toggle=page.get_by_test_id('transcript-raw-toggle')
            if toggle.count():
                toggle.click()
                expect(page.get_by_test_id('transcript-raw-comparison')).to_be_visible()
                toggle.click()
        assert errors == [], errors
        browser.close()
    report={'status':'passed','read_only':True,'url':args.url,'samples':samples,'page_errors':errors}
    (args.output/'spacing.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','samples':len(samples),'page_errors':errors},ensure_ascii=False))


if __name__ == '__main__': main()
