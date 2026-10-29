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
    log('Original thirteen checks passed; verifying synthetic steering and continuation');
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.locator('#navigation-mode').selectOption('cx_steer');
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'cx_steer');
    for (let i = 0; i < 2; i++) {
      const steps = JSON.parse(await page.evaluate(() => window.render_game_to_text())).steps;
      await page.locator('#step-btn').click();
      await page.waitForFunction(value => JSON.parse(window.render_game_to_text()).steps === value + 1, steps);
    }
    const steeringState = await (await page.request.get(url + '/api/state')).json();
    assert.equal(steeringState.running, false);
    assert.equal(steeringState.learning, false);
    assert.equal(steeringState.navigation.teacher_probability, 0);
    assert.equal(steeringState.last_action_source, 'cx_steer');
    assert.match(await page.locator('#navigation-status').innerText(), /Synthetic goal\/heading steering/);
    assert.match(await page.locator('#navigation-map-note').innerText(), /engineered cosine heuristic/);
    await page.locator('#navigation-heading').scrollIntoViewIfNeeded();
    await page.locator('.navigation-panel').screenshot({ path: path.join(out, 'steering-desktop.png') });
    await page.locator('#save-name').fill('steering-check');
    await page.locator('#save-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).checkpoints.includes('steering-check'));
    await page.locator('#navigation-mode').selectOption('policy');
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'policy');
    await page.locator('#load-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'cx_steer');
    const restoredSteering = await (await page.request.get(url + '/api/state')).json();
    assert.deepEqual(restoredSteering.navigation, steeringState.navigation);
    assert.equal(restoredSteering.last_action_source, 'cx_steer');
    assert.equal(restoredSteering.steps, steeringState.steps);
    assert.equal(restoredSteering.model_hash, steeringState.model_hash);
    log('Steering action/source and named checkpoint continuation passed; starting ten-way comparison');
    await page.locator('#evaluation-controllers').check();
    await page.locator('#evaluation-steering').check();
    await page.locator('#evaluation-seeds').fill('10000,10001');
    await page.locator('#evaluation-size').fill('5');
    await page.locator('#evaluation-preset').selectOption('concentric');
    await page.locator('#evaluation-steps').fill('16');
    await page.locator('#evaluate-btn').click();
    await page.waitForFunction(() => document.querySelectorAll('#evaluation-table-body tr').length === 10, null, { timeout: 45000 });
    const steeringReport = await (await page.request.get(url + '/api/evaluation')).json();
    assert.equal(steeringReport.schema_version, 4);
    assert.equal(Object.keys(steeringReport.variants).length, 10);
    assert.equal(steeringReport.config.include_steering_controls, true);
    assert.equal(steeringReport.variants.cx_steer.summary.collision_rate, 0);
    assert.equal(steeringReport.variants.cx_without_policy.summary.collision_rate, 0);
    assert(steeringReport.variants.cx_without_policy.episodes.every(episode => episode.policy_prior_used === false));
    const frozenSteering = await (await page.request.get(url + '/api/state')).json();
    assert.equal(frozenSteering.steps, restoredSteering.steps);
    assert.equal(frozenSteering.updates, restoredSteering.updates);
    assert.equal(frozenSteering.model_hash, restoredSteering.model_hash);
    assert.equal(frozenSteering.running, false);
    assert.equal(frozenSteering.learning, false);
    await fs.writeFile(path.join(out, 'browser-steering-evaluation.json'), JSON.stringify(steeringReport, null, 2));
    await page.locator('.evaluation-panel').screenshot({ path: path.join(out, 'steering-comparison-desktop.png') });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false,
      'desktop steering controls must not overflow horizontally');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => typeof window.render_game_to_text === 'function'
      && JSON.parse(window.render_game_to_text()).navigation?.mode === 'cx_steer'
      && document.querySelectorAll('#evaluation-table-body tr').length === 10);
    assert.equal(await page.locator('#navigation-mode').inputValue(), 'cx_steer');
    assert.match(await page.locator('#navigation-status').innerText(), /Synthetic goal\/heading steering/);
    await page.locator('.navigation-panel').screenshot({ path: path.join(out, 'steering-mobile.png') });
    await page.locator('.evaluation-panel').screenshot({ path: path.join(out, 'steering-comparison-mobile.png') });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false,
      'mobile steering controls and evaluation must not overflow horizontally');
    const mobilePanel = await page.locator('.evaluation-panel').boundingBox();
    assert(mobilePanel && mobilePanel.x >= 0 && mobilePanel.x + mobilePanel.width <= 392);
    log('Checking connected empirical replay controls on mobile');
    await page.locator('#navigation-mode').selectOption('replay_steer');
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'replay_steer');
    await page.locator('#step-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).last_action_source === 'replay_steer');
    const replayBefore = await (await page.request.get(url + '/api/state')).json();
    assert.equal(replayBefore.navigation.teacher_probability, 0);
    assert.equal(replayBefore.navigation.replay.decision_budget, 32);
    const learnedGoal = replayBefore.world.agent;
    const goalResponse = await page.request.post(url + '/api/control', {
      data: { action: 'goal', x: learnedGoal.x, z: learnedGoal.z }, headers: { Origin: url }
    });
    assert.equal(goalResponse.status(), 200);
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.replay.goal_changed);
    assert.match(await page.locator('#replay-learning-status').innerText(), /Goal changed/);
    await page.locator('#replay-btn').click();
    await page.waitForFunction(() => {
      const replay = JSON.parse(window.render_game_to_text()).navigation.replay;
      return !replay.stale && replay.last_summary.goal_known && replay.total_backups > 0;
    });
    assert.match(await page.locator('#replay-learning-status').innerText(), /state backups/);
    const replaySaved = await (await page.request.get(url + '/api/state')).json();
    assert.equal(replaySaved.steps, replayBefore.steps);
    assert.equal(replaySaved.updates, replayBefore.updates);
    assert.equal(replaySaved.model_hash, replayBefore.model_hash);
    await page.locator('#save-name').fill('replay-check');
    await page.locator('#save-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).checkpoints.includes('replay-check'));
    await page.locator('#navigation-mode').selectOption('policy');
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'policy');
    await page.locator('#load-btn').click();
    await page.waitForFunction(() => JSON.parse(window.render_game_to_text()).navigation.mode === 'replay_steer');
    const replayRestored = await (await page.request.get(url + '/api/state')).json();
    assert.deepEqual(replayRestored.navigation.replay, replaySaved.navigation.replay);
    await page.locator('.navigation-panel').screenshot({ path: path.join(out, 'replay-learning-mobile.png') });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.locator('.navigation-panel').screenshot({ path: path.join(out, 'replay-learning-desktop.png') });
    const finalState = JSON.parse(await page.evaluate(() => window.render_game_to_text()));
    assert.equal(finalState.running, false);
    assert.equal(finalState.learning, false);
    assert.deepEqual(errors, []);
    const receipt = { ok: true, page_errors: errors, tested: ['startup', 'step', 'learning toggle',
      'hierarchical preset', 'frozen matched trials', 'JSON export', 'stale weight notice', 'observed-map mode',
      'observed-only SVG', 'exact map save/load', 'eight-way controller comparison', 'mobile startup', 'mobile layout',
      'synthetic steering mode', 'steering teacher bypass and action source', 'exact steering save/load',
      'ten-way steering comparison', 'desktop steering layout', 'mobile steering layout',
      'empirical replay control', 'replay teacher bypass', 'replay preserves neural weights and steps',
      'replay continuation save/load', 'mobile and desktop replay diagnostics',
      'goal change marks replay stale', 'experienced goal receives actual value backups'],
      oracle: report.variants.shortest_path.summary, evaluated_model: report.model_hash,
      protocol_id: report.protocol_id, steering_protocol_id: steeringReport.protocol_id,
      steering_evaluation: { schema_version: steeringReport.schema_version, variants: Object.keys(steeringReport.variants) },
      final_state: { running: finalState.running, learning: finalState.learning, mode: finalState.navigation.mode,
        last_action_source: finalState.last_action_source },
      renderer: finalState.renderer };
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
