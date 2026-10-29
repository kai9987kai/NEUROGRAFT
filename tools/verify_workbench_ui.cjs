const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const [url, executablePath, out] = process.argv.slice(2);
  const browser = await chromium.launch({ executablePath, headless: true, args: ['--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, acceptDownloads: true });
  const errors = [];
  const expectedHttpErrors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => {
    if (m.type() !== 'error') return;
    if (/Failed to load resource.*400 \(Bad Request\)/.test(m.text())) expectedHttpErrors.push(m.text());
    else errors.push(m.text());
  });
  page.on('dialog', d => d.accept());
  page.setDefaultTimeout(10000);
  const state = () => page.evaluate(() => JSON.parse(window.workbenchText()));
  const idle = () => page.waitForFunction(() => window.workbenchText && !JSON.parse(window.workbenchText()).busy);
  try {
    await page.goto(url + '/workbench', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.workbenchText && JSON.parse(window.workbenchText()).models);
    await idle();
    assert.equal((await state()).provenance.kind, 'synthetic-fixture');
    assert(await page.locator('#transition-graph g[data-position]').count() > 3);
    await page.locator('#fixture-seed').fill('42');
    await page.locator('#window-select').selectOption('8');
    await page.locator('#fixture-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).provenance.seed === 42);
    await page.locator('#step-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).operations === 1);
    assert.equal((await state()).arms.recent.none.total_backups, 0);
    await page.locator('#inspect-x').fill('3');
    await page.locator('#inspect-z').fill('1');
    await page.locator('#inspect-btn').click();
    await page.locator('#observation-action').selectOption('1');
    await page.locator('#observation-outcome').selectOption('collision');
    await page.locator('#observation-count').evaluate(el => { el.value = '8'; el.dispatchEvent(new Event('input', { bubbles: true })); });
    await page.locator('#observe-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).events === 8);
    let s = await state();
    assert.notEqual(s.models.cumulative.sha256, s.models.recent.sha256);
    assert.equal(s.arms.recent.prioritized.stale, true);
    await page.locator('#replay-budget').selectOption('8');
    await page.locator('#replay-btn').click();
    await page.waitForFunction(() => !JSON.parse(window.workbenchText()).arms.recent.prioritized.stale);
    assert.equal((await state()).arms.cumulative.none.total_backups, 0);
    await page.locator('#model-cumulative').click();
    assert.equal((await state()).selected_model, 'cumulative');
    await page.locator('#schedule-uniform').click();
    assert.equal((await state()).selected_schedule, 'uniform');
    await page.locator('#play-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).playing);
    await page.locator('#pause-btn').click();
    await page.waitForFunction(() => !JSON.parse(window.workbenchText()).playing && !JSON.parse(window.workbenchText()).busy);
    const stopped = (await state()).operations;
    await page.waitForTimeout(700);
    assert.equal((await state()).operations, stopped, 'pause must prevent later batches');
    const downloaded = page.waitForEvent('download');
    await page.locator('#export-btn').click();
    const receiptPath = path.join(out, 'workbench-receipt.json');
    await (await downloaded).saveAs(receiptPath);
    await idle();
    const receipt = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
    assert.equal(receipt.final_sha256.length, 64);
    const beforeImport = await state();
    const corrupted = { ...receipt, final_sha256: '0'.repeat(64) };
    const corruptPath = path.join(out, 'tampered-receipt.json');
    await fs.writeFile(corruptPath, JSON.stringify(corrupted));
    await page.locator('#import-file').setInputFiles(corruptPath);
    await page.waitForFunction(() => !document.getElementById('error-banner').hidden);
    assert.match(await page.locator('#error-message').innerText(), /digest|hash/i);
    assert.equal((await state()).operations, beforeImport.operations);
    await idle();
    await page.locator('#reset-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).operations === 0);
    await page.locator('#import-file').setInputFiles(receiptPath);
    await page.waitForFunction(n => JSON.parse(window.workbenchText()).operations === n, beforeImport.operations);
    s = await state();
    assert.deepEqual(s.arms, beforeImport.arms);
    await page.locator('#inspect-x').fill('1');
    await page.locator('#inspect-z').fill('1');
    await page.locator('#inspect-btn').click();
    await page.locator('#goal-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).goal.join(',') === '1,1');
    await page.screenshot({ path: path.join(out, 'workbench-desktop.png'), fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.locator('.card-metrics > span').first().evaluate(el => parseFloat(getComputedStyle(el).fontSize) >= 12),
      'mobile comparison labels must remain readable');
    await page.screenshot({ path: path.join(out, 'workbench-mobile.png'), fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    await page.locator('#capture-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).provenance.kind === 'live_capture');
    assert.equal((await state()).models.cumulative.total_transitions, 8);
    assert.equal((await state()).operations, 0);
    await page.locator('#fixture-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).provenance.kind === 'synthetic-fixture');
    await idle();
    assert.deepEqual(errors, []);
    assert.equal(expectedHttpErrors.length, 1, 'only the intentional tampered import should return 400');
    const result = { ok: true, page_errors: errors, expected_rejected_requests: expectedHttpErrors, checked: ['startup', 'SVG experience graph', 'seeded fixture',
      'single backup', 'disabled replay control', 'coordinate inspection', 'supplied collision events',
      'different transition models', 'stale values', 'bounded replay', 'model and schedule toggles',
      'play/pause without later requests', 'receipt download', 'tamper rejection without state changes',
      'reset', 'exact receipt restoration', 'goal change', 'desktop layout', 'mobile layout',
      'frozen live capture', 'fixture replacement'], final: { events: (await state()).events, operations: (await state()).operations } };
    await fs.writeFile(path.join(out, 'workbench-browser.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } catch (e) {
    await page.screenshot({ path: path.join(out, 'workbench-failure.png'), fullPage: true }).catch(() => {});
    await fs.writeFile(path.join(out, 'workbench-failure.html'), await page.content()).catch(() => {});
    throw e;
  } finally { await browser.close(); }
})().catch(e => { console.error(e.stack); process.exit(1); });
