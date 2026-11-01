"""Exercise the guide in Chromium against a running static server."""
from pathlib import Path
import argparse
import json
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("url")
parser.add_argument("--screenshots", type=Path)
args = parser.parse_args()
if args.screenshots:
    args.screenshots.mkdir(parents=True, exist_ok=True)

expected = {"supported": [True, True, True], "site_confounded": [False, False, False],
            "nonreplicating": [True, False, False]}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    response = page.goto(args.url)
    assert response.status == 200
    page.locator('[data-case="nonreplicating"]').wait_for()
    if args.screenshots:
        page.screenshot(path=str(args.screenshots / "desktop.png"), full_page=True)
    for case, decisions in expected.items():
        page.locator(f'[data-case="{case}"]').click()
        for step in range(5):
            page.locator(f'[data-step="{step}"]').click()
            assert page.locator('#stage-content').inner_text().strip()
        for standard, supported in zip(["screen", "replicate", "full"], decisions):
            page.locator(f'[data-standard="{standard}"]').click()
            assert page.locator('.decision-card h3').inner_text() == ("Supported" if supported else "Withheld")
        with page.expect_download() as download:
            page.locator('#record-link').click()
        record = json.loads(Path(download.value.path()).read_text())
        assert record["example"] == case
        page.locator('[data-step="3"]').click()
        assert page.locator('.gate-row').count() == 5
        page.locator('.gate-row summary').first.click()
    for step in range(4):
        page.locator(f'[data-replay="{step}"]').click()
        assert page.locator('#replay-content').inner_text().strip()
    assert page.locator('#replay-content .status').all_inner_texts() == ["Withheld", "Withheld"]
    page.locator('#replay-next').click()
    assert page.locator('[data-replay="0"]').get_attribute('aria-current') == 'step'
    for width in [1440, 1024, 768, 390, 360]:
        page.set_viewport_size({"width": width, "height": 900})
        for step in range(5):
            page.locator(f'[data-step="{step}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, step)
        if args.screenshots and width == 390:
            page.locator('[data-step="3"]').click()
            page.screenshot(path=str(args.screenshots / "mobile.png"), full_page=True)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.locator('[data-step="2"]').click()
    if args.screenshots:
        page.locator('#workspace').screenshot(path=str(args.screenshots / "results.png"))
    page.locator('[data-step="3"]').click()
    if args.screenshots:
        page.locator('#workspace').screenshot(path=str(args.screenshots / "checks.png"))
    page.reload()
    page.locator('[data-case="nonreplicating"]').wait_for()
    assert page.locator('[data-step="3"]').get_attribute('aria-current') == 'step'
    page.locator('[data-case="supported"]').focus()
    page.keyboard.press('Space')
    assert page.locator('[data-case="supported"]').get_attribute('aria-pressed') == 'true'
    page.locator('[data-step="4"]').focus()
    page.keyboard.press('Enter')
    assert page.locator('.decision-card h3').inner_text() == 'Supported'
    page.locator('#copy-command').focus()
    page.keyboard.press('Enter')
    page.wait_for_function("document.querySelector('#copy-status').textContent.startsWith('Command')")
    assert not errors, errors
    context.close()
    browser.close()
print('Browser checks passed: all example stages, nine reporting decisions, downloads, follow-up playback, URL state, and five viewport widths.')
