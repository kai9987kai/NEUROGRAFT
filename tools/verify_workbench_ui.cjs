const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

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
    assert.equal(await page.locator('#model-adaptive').count(), 1, 'adaptive evidence control must be present');
    assert.equal((await state()).selected_model, 'adaptive');
    assert.deepEqual(Object.keys((await state()).models).sort(), ['adaptive', 'cumulative', 'recent']);
    assert.equal((await state()).adaptive_evidence.events, 0);
    assert.match(await page.locator('#evidence-score-count').innerText(), /no scored events/i);
    assert.deepEqual(await page.locator('#adaptive-score-table tbody td').allTextContents(), Array(8).fill('—'));
    assert.match(await page.locator('#play-state').innerText(), /9 planners/);
    assert(await page.locator('#transition-graph g[data-position]').count() > 3);
    await page.locator('#fixture-seed').fill('42');
    await page.locator('#window-select').selectOption('8');
    await page.locator('#fixture-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).provenance.seed === 42);
    const frozenDiagnostics = (await state()).adaptive_evidence;
    await page.locator('#step-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).operations === 1);
    assert.equal((await state()).arms.recent.none.total_backups, 0);
    assert.deepEqual((await state()).adaptive_evidence, frozenDiagnostics, 'replay cannot advance predictive scores');
    await page.locator('#inspect-x').fill('3');
    await page.locator('#inspect-z').fill('1');
    await page.locator('#inspect-btn').click();
    await page.locator('#observation-action').selectOption('1');
    await page.locator('#observation-outcome').selectOption('collision');
    const preEvent = await state();
    const beforeEdges = preEvent.models.cumulative.edges.filter(e => e.source.join(',') === '3,1' && e.action === 1);
    const total = beforeEdges.reduce((sum, edge) => sum + edge.count, 0);
    const moves = beforeEdges.filter(e => e.destination.join(',') !== '3,1').reduce((sum, edge) => sum + edge.count, 0);
    const probabilityMove = (moves + 0.5) / (total + 1);
    await page.locator('#observe-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).events === 1);
    const firstScores = (await state()).adaptive_evidence;
    assert(Math.abs(firstScores.losses.cumulative.brier - probabilityMove ** 2) < 1e-12, 'first outcome must be scored before its counts are updated');
    assert(Math.abs(firstScores.losses.cumulative.log_loss + Math.log(1 - probabilityMove)) < 1e-12);
    assert.equal(firstScores.pairs.find(pair => pair.source.join(',') === '3,1' && pair.action === 1).fresh_count, 1);
    await page.locator('#reset-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).events === 0);
    await page.locator('#inspect-x').fill('3');
    await page.locator('#inspect-z').fill('1');
    await page.locator('#inspect-btn').click();
    await page.locator('#observation-count').evaluate(el => { el.value = '8'; el.dispatchEvent(new Event('input', { bubbles: true })); });
    await page.locator('#observe-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).events === 8);
    let s = await state();
    assert.notEqual(s.models.cumulative.sha256, s.models.recent.sha256);
    assert.equal(s.arms.recent.prioritized.stale, true);
    assert.equal(s.adaptive_evidence.events, 8);
    assert.match(await page.locator('#adaptive-pairs').innerText(), /East/);
    assert.match(await page.locator('#adaptive-pairs').innerText(), /8 fresh/);
    const batchDiagnostics = s.adaptive_evidence;
    assert.deepEqual(await page.locator('#adaptive-score-table tbody td').allTextContents(),
      ['cumulative', 'recent', 'guarded_recent', 'adaptive'].flatMap(model => ['brier', 'log_loss'].map(field => (batchDiagnostics.losses[model][field] / 8).toFixed(4))),
      'displayed forecast averages must use the number of scored events');
    for (let read = 0; read < 2; read++) {
      const response = await page.request.get(url + '/api/workbench');
      assert(response.ok());
      assert.deepEqual((await response.json()).adaptive_evidence, batchDiagnostics, 'snapshot reads cannot update scores');
    }
    const resetResponse = await page.request.post(url + '/api/workbench', { data: { action: 'reset' } });
    assert(resetResponse.ok());
    for (let event = 0; event < 8; event++) {
      const response = await page.request.post(url + '/api/workbench', { data: { action: 'observe', source: [3, 1], action_index: 1, destination: [3, 1], count: 1 } });
      assert(response.ok());
      if (event === 7) assert.deepEqual((await response.json()).adaptive_evidence, batchDiagnostics, 'one batch and eight sequential observations must score identically');
    }
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.workbenchText && JSON.parse(window.workbenchText()).events === 8);
    await idle();
    await page.locator('#replay-budget').selectOption('8');
    await page.locator('#replay-btn').click();
    await page.waitForFunction(() => !JSON.parse(window.workbenchText()).arms.recent.prioritized.stale);
    assert.equal((await state()).arms.cumulative.none.total_backups, 0);
    assert.deepEqual((await state()).adaptive_evidence, batchDiagnostics);
    await page.locator('#model-adaptive').click();
    assert.equal((await state()).selected_model, 'adaptive');
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
    assert.deepEqual((await state()).adaptive_evidence, batchDiagnostics, 'read controls and replay leave prediction scores unchanged');
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
    assert.deepEqual(s.adaptive_evidence, beforeImport.adaptive_evidence);
    await page.locator('#inspect-x').fill('1');
    await page.locator('#inspect-z').fill('1');
    await page.locator('#inspect-btn').click();
    await page.locator('#goal-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).goal.join(',') === '1,1');
    assert.deepEqual((await state()).adaptive_evidence, batchDiagnostics, 'goal changes cannot update prediction scores');
    await page.locator('#model-adaptive').click();
    await page.locator('#inspect-x').fill('3');
    await page.locator('#inspect-z').fill('1');
    await page.locator('#inspect-btn').click();
    await page.screenshot({ path: path.join(out, 'workbench-desktop.png'), fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.locator('.card-metrics > span').first().evaluate(el => parseFloat(getComputedStyle(el).fontSize) >= 12),
      'mobile comparison labels must remain readable');
    assert(await page.locator('.adaptive-pair-detail').first().evaluate(el => parseFloat(getComputedStyle(el).fontSize) >= 12),
      'mobile adaptive diagnostics must remain readable');
    await page.screenshot({ path: path.join(out, 'workbench-mobile.png'), fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    const legacy = spawnSync(process.env.NEUROGRAFT_PYTHON || 'python', ['-c', 'import json; from neurograft.workbench import ReplayWorkbench; print(json.dumps(ReplayWorkbench(seed=42).export_receipt()))'], { cwd: path.resolve(__dirname, '..'), encoding: 'utf8' });
    assert.equal(legacy.status, 0, legacy.stderr);
    const legacyPath = path.join(out, 'legacy-v1-receipt.json');
    await fs.writeFile(legacyPath, legacy.stdout);
    await page.locator('#model-adaptive').click();
    await page.locator('#import-file').setInputFiles(legacyPath);
    await page.waitForFunction(() => !JSON.parse(window.workbenchText()).models.adaptive);
    await idle();
    assert.equal(await page.locator('#model-adaptive').isDisabled(), true);
    assert.equal((await state()).selected_model, 'recent');
    assert.match(await page.locator('#play-state').innerText(), /6 planners/);
    assert.match(await page.locator('#evidence-score-count').innerText(), /legacy/i);
    await page.locator('#model-cumulative').click();
    assert.equal((await state()).selected_model, 'cumulative');
    await page.locator('#step-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).operations === 1);
    await page.screenshot({ path: path.join(out, 'workbench-mobile-v1.png'), fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    await page.locator('#capture-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).provenance.kind === 'live_capture');
    assert.equal(await page.locator('#model-adaptive').isDisabled(), false);
    assert.equal((await state()).models.cumulative.total_transitions, 8);
    assert.equal((await state()).operations, 0);
    await page.locator('#fixture-btn').click();
    await page.waitForFunction(() => JSON.parse(window.workbenchText()).provenance.kind === 'synthetic-fixture');
    await idle();
    assert.deepEqual(errors, []);
    assert.equal(expectedHttpErrors.length, 1, 'only the intentional tampered import should return 400');
    const result = { ok: true, page_errors: errors, expected_rejected_requests: expectedHttpErrors, checked: ['startup', 'SVG experience graph', 'seeded fixture',
      'single backup', 'disabled replay control', 'coordinate inspection', 'supplied collision events',
      'three transition models', 'pre-update proper scores', 'sequential batch scoring', 'predictive read and replay purity',
      'adaptive per-action inspector', 'stale values', 'bounded replay', 'model and schedule toggles',
      'play/pause without later requests', 'receipt download', 'tamper rejection without state changes',
      'reset', 'exact receipt restoration', 'goal change', 'desktop layout', 'mobile layout', 'legacy v1 import and continued replay',
      'frozen live capture', 'fixture replacement'], final: { events: (await state()).events, operations: (await state()).operations } };
    await fs.writeFile(path.join(out, 'workbench-browser.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } catch (e) {
    await page.screenshot({ path: path.join(out, 'workbench-failure.png'), fullPage: true }).catch(() => {});
    await fs.writeFile(path.join(out, 'workbench-failure.html'), await page.content()).catch(() => {});
    throw e;
  } finally { await browser.close(); }
})().catch(e => { console.error(e.stack); process.exit(1); });
