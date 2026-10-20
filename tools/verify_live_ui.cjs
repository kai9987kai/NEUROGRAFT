const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const fsSync = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const [url, executablePath, out] = process.argv.slice(2);
  fsSync.writeFileSync(path.join(out, 'browser-stages.log'), '');
  const log = (message) => {
    const line = `${new Date().toISOString()} ${message}`;
    console.log(line); fsSync.appendFileSync(path.join(out, 'browser-stages.log'), `${line}\n`);
  };
  log('Launching Chromium');
  const browser = await chromium.launch({ executablePath, headless: true,
    timeout: 20000, args: ['--enable-unsafe-swiftshader', '--disable-dev-shm-usage'] });
  log('Chromium launched; creating desktop page');
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, acceptDownloads: true });
  page.setDefaultTimeout(10000);
  page.setDefaultNavigationTimeout(15000);
  const errors = [];
  page.on('pageerror', error => { errors.push(error.message); log(`PAGE ERROR ${error.message}`); });
  page.on('console', message => { if (message.type() === 'error') { errors.push(message.text()); log(`CONSOLE ERROR ${message.text()}`); } });
  page.on('requestfailed', request => log(`REQUEST FAILED ${request.url()} ${request.failure()?.errorText}`));
  try {
    log('Navigating to isolated session');
    await page.goto(url + '/?test=1', { waitUntil: 'domcontentloaded' });
    log('Waiting for initialized live snapshot');
    await page.waitForFunction(() => typeof window.render_game_to_text === 'function'
      && JSON.parse(window.render_game_to_text()).steps >= 8, null, { timeout: 15000 });
    log('Startup passed; stepping once');
    assert.match(await page.locator('#experiment-scope').innerText(), /do not feed the policy/);
    await page.locator('#step-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).steps === 9);
    await page.locator('label[for="learning-toggle"]').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).learning === false);
    log('Step and learning toggle passed; changing preset');
    await page.locator('[data-world-preset="hierarchical"]').click();
    await page.waitForFunction(() => document.getElementById('world-preset-badge').textContent === 'hierarchical');
    await page.locator('#evaluation-seeds').fill('10000,10001');
    await page.locator('#evaluation-size').fill('5');
    await page.locator('#evaluation-preset').selectOption('concentric');
    await page.locator('#evaluation-steps').fill('16');
    log('Preset changed; starting bounded evaluation');
    const before = await (await page.request.get(url + '/api/state')).json();
    await page.locator('#evaluate-btn').click();
    await page.waitForFunction(() => document.querySelectorAll('#evaluation-table-body tr').length === 5,
                               null, { timeout: 45000 });
    log('Evaluation rows rendered; checking session and exporting JSON');
    const after = await (await page.request.get(url + '/api/state')).json();
    assert.equal(after.steps, before.steps);
    assert.equal(after.model_hash, before.model_hash);
    assert.equal(after.updates, before.updates);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#export-evaluation-btn').click();
    const download = await downloadEvent;
    const exportedPath = path.join(out, 'browser-evaluation.json');
    await download.saveAs(exportedPath);
    const report = JSON.parse(await fs.readFile(exportedPath, 'utf8'));
    assert.equal(report.variants.shortest_path.summary.success_rate, 1);
    assert.equal(report.variants.shortest_path.summary.spl, 1);
    assert.equal(report.model_hash, before.model_hash);
    log('Evaluation and JSON export passed; checking stale weight notice');
    await page.locator('label[for="learning-toggle"]').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).learning === true);
    await page.locator('#step-btn').click();
    await page.waitForFunction(() => document.getElementById('evaluation-status').textContent.includes('weights have changed'));
    await page.locator('#toast').waitFor({ state: 'hidden' });
    await page.locator('#evaluation-heading').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(out, 'live-evaluation-desktop.png') });
    await page.locator('.evaluation-panel').screenshot({ path: path.join(out, 'live-evaluation-panel.png') });
    log('Legacy checks passed; verifying observed-map controls and persistence');
    await page.locator('label[for="learning-toggle"]').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).learning === false);
    await page.locator('#navigation-mode').selectOption('observed_map');
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation?.mode === 'observed_map');
    for (let i = 0; i < 3; i++) {
      const steps = JSON.parse(await page.evaluate(() => window.render_game_to_text())).steps;
      await page.locator('#step-btn').click();
      await page.waitForFunction((value) => JSON.parse(window.render_game_to_text()).steps === value + 1, steps);
    }
    const mapped = (await (await page.request.get(url + '/api/state')).json()).navigation;
    assert.equal(mapped.teacher_probability, 0);
    assert(mapped.map.visited > 0);
    assert(await page.locator('#observed-map rect').count() > 0);
    await page.locator('#navigation-heading').scrollIntoViewIfNeeded();
    await page.locator('.navigation-panel').screenshot({ path: path.join(out, 'observed-map-desktop.png') });
    await page.locator('#save-name').fill('navigation-check');
    await page.locator('#save-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).checkpoints.includes('navigation-check'));
    await page.locator('#navigation-mode').selectOption('policy');
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'policy');
    await page.locator('#load-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'observed_map');
    const restoredMap = (await (await page.request.get(url + '/api/state')).json()).navigation;
    assert.deepEqual(restoredMap, mapped);
    await page.locator('#evaluation-controllers').check();
    await page.locator('#evaluate-btn').click();
    await page.waitForFunction(() => document.querySelectorAll('#evaluation-table-body tr').length === 8, null, { timeout: 45000 });
    const extended = await (await page.request.get(url + '/api/evaluation')).json();
    assert.equal(extended.variants.observed_map.summary.success_rate, 1);
    assert.equal(extended.variants.observed_map.summary.collision_rate, 0);
    assert.equal(extended.config.include_controllers, true);
    await page.locator('.evaluation-panel').screenshot({ path: path.join(out, 'controller-comparison.png') });
    log('Map mode, observed-only SVG, checkpoint restoration and eight-way comparison passed; checking mobile layout');
    const screenshotScope = await page.locator('#evaluation-results').boundingBox();
    assert(screenshotScope && screenshotScope.width > 0);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).steps >= 10
      && document.querySelectorAll('#evaluation-table-body tr').length === 8
      && JSON.parse(window.render_game_to_text()).navigation.mode === 'observed_map');
    await page.locator('#evaluation-heading').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(out, 'live-evaluation-mobile.png') });
    await page.locator('.evaluation-panel').screenshot({ path: path.join(out, 'live-evaluation-mobile-panel.png') });
    await page.locator('.navigation-panel').screenshot({ path: path.join(out, 'observed-map-mobile.png') });
    const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
    if (horizontalOverflow) {
      const overflow = await page.evaluate(() => ({ viewport: innerWidth, width: document.documentElement.scrollWidth,
        elements: [...document.querySelectorAll('body *')].map(el => ({ tag: el.tagName, id: el.id, class: el.className?.baseVal ?? el.className,
          x: el.getBoundingClientRect().x, right: el.getBoundingClientRect().right, width: el.getBoundingClientRect().width }))
          .filter(el => el.width && el.right > innerWidth + 2).slice(0, 40) }));
      await fs.writeFile(path.join(out, 'mobile-overflow.json'), JSON.stringify(overflow, null, 2));
      log(`Mobile overflow ${JSON.stringify(overflow)}`);
    }
    assert.equal(horizontalOverflow, false, 'mobile page must not overflow horizontally');
    assert.deepEqual(errors, []);
    const receipt = { ok: true, page_errors: errors, tested: ['startup', 'step', 'learning toggle',
      'hierarchical preset', 'frozen matched trials', 'JSON export', 'stale weight notice', 'observed-map mode',
      'observed-only SVG', 'exact map save/load', 'eight-way controller comparison', 'mobile startup', 'mobile layout'],
      oracle: report.variants.shortest_path.summary, evaluated_model: report.model_hash,
      protocol_id: report.protocol_id, renderer: JSON.parse(await page.evaluate(() => window.render_game_to_text())).renderer };
    assert.equal(receipt.renderer, 'three-webgl');
    await fs.writeFile(path.join(out, 'browser-verification.json'), JSON.stringify(receipt, null, 2));
    log('Desktop and mobile verification passed');
    console.log(JSON.stringify(receipt));
  } catch (error) {
    log(`FAILED ${error.stack}`);
    await fs.writeFile(path.join(out, 'browser-failure.html'), await page.content()).catch(() => {});
    await page.screenshot({ path: path.join(out, 'browser-failure.png'), timeout: 5000 }).catch(() => {});
    throw error;
  } finally {
    log('Closing isolated browser');
    let closeTimer;
    try {
      await Promise.race([browser.close(), new Promise((_, reject) => { closeTimer = setTimeout(() => reject(new Error('isolated browser close exceeded 60s')), 60000); })]);
    } finally { clearTimeout(closeTimer); }
    log('Browser closed');
  }
})().catch(error => { console.error(error.stack); process.exit(1); });
