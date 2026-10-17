const $ = (id) => document.getElementById(id);
const TEST_MODE = new URLSearchParams(location.search).get('test') === '1';
const SOURCE_KEYS = ['worm', 'fly_f', 'fly_m', 'human'];
const SOURCE_LABELS = ['Worm', 'Fly ♀', 'Fly ♂', 'Human'];
const SOURCE_COLORS = ['#5ee9ce', '#69b5ff', '#ad9af6', '#eda879'];
const BEACON_COLORS = { blue: '#69b5ff', amber: '#f5be66', rose: '#ed85ae' };
const DIRECTION_NAMES = ['north', 'east', 'south', 'west'];
let session = null;
let writeInFlight = false;
let pollInFlight = false;
let generation = 0;
let lastSuccessful = 0;
let networkError = false;
let lastChat = '';
let lastFacts = '';
let lastCheckpoints = '';
let toastTimer;
let THREE, renderer, scene, camera, worldGroup, wallGroup, beaconGroup, agentGroup, goalGroup, trailLine;
let floorMesh, raycaster, pointerVector;
let renderedWorldKey = '';
let webglReady = false;
let webglError = null;
let sceneDirty = true;
let worldSize = 11;
let lastWorldVisualKey = '';
const beaconPickables = [];
const agentTarget = { x: 0, z: 0, angle: 0 };
const orbit = { theta: 0.67, phi: 0.75, radius: 20.5, targetX: 0, targetZ: 0 };
const pointer = { down: false, x: 0, y: 0, startX: 0, startY: 0, dragged: false, id: null };
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const formatNumber = (value, decimals = 0) => Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals }) : '—';
const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const clip = (value, low, high) => Math.max(low, Math.min(high, value));

function showError(message) {
  $('error-message').textContent = String(message);
  $('error-banner').hidden = false;
}

function showToast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
}

class APIError extends Error {
  constructor(message, status = 0) { super(message); this.status = status; }
}

async function fetchJSON(path, payload) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), payload ? 20000 : 7000);
  try {
    const response = await fetch(path, {
      method: payload ? 'POST' : 'GET',
      headers: payload ? { 'Content-Type': 'application/json' } : { Accept: 'application/json' },
      body: payload ? JSON.stringify(payload) : undefined,
      signal: controller.signal,
      cache: 'no-store',
      credentials: 'same-origin',
    });
    let data;
    try { data = await response.json(); }
    catch { throw new APIError(`The local server returned an unreadable response (${response.status}).`, response.status); }
    if (!response.ok) throw new APIError(data.error || `Request failed (${response.status}).`, response.status);
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new APIError('The local server did not respond in time. The last received state is still shown.');
    if (error instanceof APIError) throw error;
    throw new APIError('Cannot reach the local session. Keep the Python live service running; this view will reconnect automatically.');
  } finally { clearTimeout(timeout); }
}

function isStale() { return !lastSuccessful || Date.now() - lastSuccessful > 8000; }

function updateConnectivity() {
  const stale = isStale();
  const indicator = $('connection-status');
  indicator.className = `connection ${networkError ? 'offline' : stale && lastSuccessful ? 'stale' : lastSuccessful ? 'online' : ''}`;
  $('connection-label').textContent = networkError ? 'Disconnected · retrying' : stale && lastSuccessful ? 'State is stale' : lastSuccessful ? 'Local session connected' : 'Connecting';
  if (session) {
    $('scene-state').textContent = networkError ? 'DISCONNECTED · LAST RECEIVED WORLD' : stale ? 'STALE · LAST RECEIVED WORLD' : session.running ? 'LIVE · SERVER CONTROLLED' : 'PAUSED · MANUAL CONTROL';
    $('run-state').textContent = writeInFlight ? 'Updating session…' : networkError ? 'Session disconnected' : stale ? 'Awaiting fresh state' : session.running ? '● Session running' : 'Ⅱ Session paused';
  }
  updateControls();
}

function updateControls() {
  const unavailable = writeInFlight || !session || networkError || isStale();
  document.querySelectorAll('[data-write]').forEach((el) => { el.disabled = unavailable; });
  $('chat-input').disabled = unavailable;
  $('step-btn').disabled = unavailable || Boolean(session?.running);
  $('pause-btn').hidden = Boolean(session && !session.running);
  $('resume-btn').hidden = !session || Boolean(session.running);
  $('chat-send').disabled = unavailable || !$('chat-input').value.trim();
  document.querySelectorAll('.example-chip').forEach((el) => { el.disabled = unavailable; });
  if (session && !writeInFlight) $('learning-toggle').checked = Boolean(session.learning);
}

function applySnapshot(incoming) {
  if (!incoming || typeof incoming !== 'object' || !incoming.world || !incoming.world.agent) throw new APIError('The local server returned an incomplete session snapshot.');
  session = { ...incoming, runtime: incoming.runtime ?? session?.runtime, checkpoints: incoming.checkpoints ?? session?.checkpoints ?? [] };
  lastSuccessful = Date.now();
  networkError = false;
  renderSession();
  updateConnectivity();
}

async function poll() {
  if (!pollInFlight && !writeInFlight) {
    pollInFlight = true;
    const startedAtGeneration = generation;
    try {
      const state = await fetchJSON('/api/state');
      // A read begun before a mutation must never replace its newer response.
      if (!writeInFlight && startedAtGeneration === generation) applySnapshot(state);
    } catch (error) {
      if (!writeInFlight && startedAtGeneration === generation) {
        networkError = true;
        if (!session) {
          $('scene-loading').querySelector('strong').textContent = 'Waiting for the local service';
          $('scene-loading').querySelector('span:last-child').textContent = error.message;
        }
        updateConnectivity();
      }
    } finally { pollInFlight = false; }
  }
  setTimeout(poll, 500);
}

async function mutate(path, payload) {
  if (writeInFlight) return null;
  const focused = document.activeElement;
  writeInFlight = true;
  generation += 1;
  updateConnectivity();
  try {
    const result = await fetchJSON(path, payload);
    applySnapshot(result.state || result);
    return result;
  } catch (error) {
    showError(error.message);
    if (!error.status) networkError = true;
    return null;
  } finally {
    writeInFlight = false;
    generation += 1;
    updateConnectivity();
    if (focused && focused !== document.body && focused.isConnected && !focused.disabled && !focused.hidden && document.activeElement === document.body) {
      focused.focus({ preventScroll: true });
    }
  }
}

function control(payload) { return mutate('/api/control', payload); }

function makeSourceRows() {
  SOURCE_KEYS.forEach((key, index) => {
    const row = document.createElement('div');
    row.className = 'source-row';
    row.style.setProperty('--source-color', SOURCE_COLORS[index]);
    const icon = document.createElement('span');
    icon.className = 'source-icon';
    icon.textContent = ['C', 'F', 'M', 'H'][index];
    const copy = document.createElement('div');
    const title = document.createElement('div');
    title.className = 'source-title';
    const name = document.createElement('span');
    name.textContent = SOURCE_LABELS[index];
    const badge = document.createElement('span');
    badge.id = `source-${key}-badge`;
    badge.className = 'source-badge';
    badge.textContent = 'pending';
    title.append(name, badge);
    const detail = document.createElement('div');
    detail.id = `source-${key}-detail`;
    detail.className = 'source-detail';
    detail.textContent = 'Awaiting graph';
    const nodes = document.createElement('div');
    nodes.id = `source-${key}-nodes`;
    nodes.className = 'source-nodes';
    nodes.textContent = 'Top firing: —';
    copy.append(title, detail, nodes);
    const activity = document.createElement('div');
    activity.className = 'activity-value';
    const value = document.createElement('span');
    value.id = `source-${key}-activity`;
    value.textContent = '—';
    const track = document.createElement('div');
    track.className = 'activity-track';
    const bar = document.createElement('div');
    bar.className = 'activity-bar';
    bar.id = `source-${key}-bar`;
    track.append(bar);
    activity.append(value, track);
    row.append(icon, copy, activity);
    $('source-list').append(row);
  });
  const empty = document.createElement('span');
  $('routing-matrix').append(empty);
  ['WORM', 'FLY ♀', 'FLY ♂', 'HUMAN', 'NULL'].forEach((label) => {
    const cell = document.createElement('span');
    cell.className = 'route-col'; cell.textContent = label;
    $('routing-matrix').append(cell);
  });
  SOURCE_LABELS.forEach((label, row) => {
    const name = document.createElement('span');
    name.className = 'route-label'; name.textContent = label;
    $('routing-matrix').append(name);
    for (let col = 0; col < 5; col += 1) {
      const cell = document.createElement('div');
      cell.id = `route-${row}-${col}`; cell.className = 'route-cell'; cell.textContent = '—';
      $('routing-matrix').append(cell);
    }
  });
}

function renderSession() {
  const world = session.world;
  const telemetry = session.telemetry || {};
  $('world-size').textContent = `${world.size} × ${world.size}`;
  $('agent-position').textContent = `${world.agent.x}, ${world.agent.z}`;
  $('scene-goal').textContent = `${world.goal.color ? `${world.goal.color[0].toUpperCase()}${world.goal.color.slice(1)} target` : 'Custom target'} · ${world.goal.x}, ${world.goal.z}`;
  $('scene-goal').style.color = BEACON_COLORS[world.goal.color] || '#b5c9d7';
  $('learning-label').textContent = session.learning ? 'ON' : 'OFF';
  $('metric-steps').textContent = formatNumber(session.steps);
  $('metric-updates').textContent = formatNumber(session.updates);
  $('metric-update-note').textContent = session.learning
    ? `Evolving · model ${(session.model_hash || '').slice(0, 8) || 'live'}`
    : 'Learning paused';
  $('metric-goals').textContent = formatNumber(session.goals_reached);
  $('metric-episode').textContent = `Episode ${formatNumber(session.episodes)}`;
  $('metric-accuracy').textContent = Number.isFinite(session.policy_accuracy) ? `${formatNumber(session.policy_accuracy * 100, 1)}%` : '—';
  $('metric-reward').textContent = formatNumber(world.total_reward, 2);
  const action = Number.isInteger(session.last_action) ? DIRECTION_NAMES[session.last_action] : '—';
  $('metric-action').textContent = `${action} · ${session.last_action_source || 'awaiting action'}`;
  $('metric-action').title = `Last action: ${action}. Selection source: ${session.last_action_source || 'none'}. Teacher probability: ${formatNumber(finite(session.teacher_probability) * 100, 0)}%.`;
  ['blue', 'amber', 'rose'].forEach((color) => {
    $(`goal-${color}`).classList.toggle('active', world.goal.color === color);
    $(`goal-${color}`).setAttribute('aria-pressed', String(world.goal.color === color));
  });
  SOURCE_KEYS.forEach((key, index) => {
    const source = session.sources?.[key];
    const badge = $(`source-${key}-badge`);
    badge.textContent = source ? source.degree_rewired_control ? 'control' : source.status || 'unknown' : 'missing';
    badge.classList.toggle('imported', Boolean(source?.authentic_ready));
    $(`source-${key}-detail`).textContent = source ? `${formatNumber(source.node_count)} ${source.scale === 'brain_region' ? 'regions' : 'nodes'} · ${formatNumber(source.edge_count)} observations` : 'No graph in bundle';
    const topNodes = telemetry.top_nodes?.[key];
    const gate = telemetry.graft_gates?.[index];
    const gateStr = Number.isFinite(gate) ? `Gate ${(gate * 100).toFixed(0)}%` : '';
    const nodeStr = Array.isArray(topNodes) && topNodes.length
      ? `Top: ${topNodes.slice(0, 3).map((n) => `${n.id}(${finite(n.activity).toFixed(2)})`).join(' ')}`
      : '';
    $(`source-${key}-nodes`).textContent = [gateStr, nodeStr].filter(Boolean).join(' · ') || 'Top firing: —';
    const activity = finite(session.activity?.[index]);
    $(`source-${key}-activity`).textContent = activity.toFixed(3);
    $(`source-${key}-bar`).style.width = `${clip(activity / (1 + Math.abs(activity)), 0, 1) * 100}%`;
  });
  if ($('telemetry-delta')) {
    $('telemetry-delta').textContent = Number.isFinite(telemetry.weight_delta_norm) ? telemetry.weight_delta_norm.toFixed(3) : '—';
    $('telemetry-graft').textContent = Number.isFinite(telemetry.graft_impact) ? telemetry.graft_impact.toFixed(3) : '—';
    $('telemetry-entropy').textContent = Number.isFinite(telemetry.route_entropy) ? `${(telemetry.route_entropy * 100).toFixed(0)}%` : '—';
    const probs = Array.isArray(telemetry.action_probs) ? telemetry.action_probs : [0.25, 0.25, 0.25, 0.25];
    for (let dir = 0; dir < 4; dir += 1) {
      const p = clip(finite(probs[dir], 0.25), 0, 1);
      if ($(`prob-${dir}`)) $(`prob-${dir}`).textContent = `${(p * 100).toFixed(0)}%`;
      if ($(`prob-bar-${dir}`)) $(`prob-bar-${dir}`).style.width = `${p * 100}%`;
      if ($(`prob-cell-${dir}`)) $(`prob-cell-${dir}`).classList.toggle('teacher', telemetry.expert_action === dir);
    }
  }
  const env = telemetry.environment || {};
  const activePreset = env.preset || 'labyrinth';
  if ($('world-preset-badge')) $('world-preset-badge').textContent = activePreset;
  if ($('env-hud-summary') && Number.isFinite(env.geodesic_distance)) {
    $('env-hud-summary').textContent = `Geo ${env.geodesic_distance} · Detour ${finite(env.detour_index, 1).toFixed(2)}× · Choke ${(env.choke_points || []).length}`;
  }
  document.querySelectorAll('.world-size-btn').forEach((btn) => {
    btn.classList.toggle('active', Number(btn.dataset.worldSize) === world.size);
  });
  document.querySelectorAll('.world-preset-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.worldPreset === activePreset);
  });
  const nm = telemetry.neuromodulators || {};
  if ($('mod-regime') && nm.regime) {
    $('mod-regime').textContent = nm.regime;
    $('mod-gain').textContent = `Gain ${finite(nm.plasticity_gain, 1).toFixed(2)}×`;
    const da = clip(finite(nm.dopamine, 0.5), 0, 1);
    const ht = clip(finite(nm.serotonin, 0.5), 0, 1);
    const oa = clip(finite(nm.octopamine, 0.5), 0, 1);
    $('mod-da').textContent = da.toFixed(2);
    $('mod-bar-da').style.width = `${da * 100}%`;
    $('mod-5ht').textContent = ht.toFixed(2);
    $('mod-bar-5ht').style.width = `${ht * 100}%`;
    $('mod-oa').textContent = oa.toFixed(2);
    $('mod-bar-oa').style.width = `${oa * 100}%`;
  }
  if ($('env-complexity-foot') && Number.isFinite(env.free_cells)) {
    $('env-complexity-foot').textContent = `Free ${env.free_cells}/${env.interior_cells} (${(finite(env.obstacle_density) * 100).toFixed(0)}% walls) · Corridors ${env.corridor_cells} · Junctions ${env.branch_junctions} · Plume ${(finite(env.chemoattractant) * 100).toFixed(0)}%`;
  }
  SOURCE_KEYS.forEach((key, idx) => {
    const ablEl = $(`ablation-${key}`);
    const tauEl = $(`tau-${key}`);
    if (ablEl) {
      const val = telemetry.source_ablation?.[key];
      ablEl.textContent = Number.isFinite(val) ? `Δ ${val.toFixed(3)}` : '—';
    }
    if (tauEl) {
      const tau = telemetry.liquid_tau?.[idx];
      const hubs = telemetry.connectome_motifs?.[key]?.rich_club_hubs || [];
      const hubStr = hubs.slice(0, 2).map((h) => `${h.id}(d=${h.degree})`).join(' ');
      tauEl.textContent = `τ=${Number.isFinite(tau) ? tau.toFixed(2) : '—'}s · ${hubStr || '—'}`;
    }
  });
  for (let row = 0; row < 4; row += 1) for (let col = 0; col < 5; col += 1) {
    const value = clip(finite(session.routes?.[row]?.[col]), 0, 1);
    const cell = $(`route-${row}-${col}`);
    cell.textContent = value.toFixed(2);
    cell.title = `${SOURCE_LABELS[row]} receives from ${col === 4 ? 'null route' : SOURCE_LABELS[col]}: ${value.toFixed(4)}`;
    cell.style.background = `rgba(71, 182, 161, ${0.06 + value * 0.54})`;
    cell.style.borderColor = `rgba(94, 233, 206, ${0.08 + value * 0.36})`;
    cell.style.color = value > 0.35 ? '#c3fff0' : '#8ca8b6';
  }
  renderLoss(); renderChat(); renderFacts(); renderPersistence();
  $('evidence-scope').textContent = session.evidence_scope || 'Experimental synthetic navigation. Online supervised adaptation; policy match is recent teacher agreement.';
  $('graph-id').textContent = [
    session.graph_hash ? `GRAPH ${session.graph_hash.slice(0, 8)}` : '',
    session.model_hash ? `MODEL ${session.model_hash.slice(0, 8)}` : '',
  ].filter(Boolean).join(' · ');
  $('graph-id').title = `Graph SHA-256: ${session.graph_hash || ''}\nModel SHA-256: ${session.model_hash || ''}`;
  if (webglReady) updateWorld(world);
}

function renderLoss() {
  const history = (session.loss_history || []).filter(Number.isFinite).slice(-64);
  $('loss-value').textContent = Number.isFinite(session.last_loss) ? session.last_loss.toFixed(4) : '—';
  $('loss-note').textContent = history.length ? `${history.length} recent updates · supervised action prediction` : 'Loss appears after the first learning update.';
  if (!history.length) { $('loss-line').setAttribute('d', ''); $('loss-area').setAttribute('d', ''); return; }
  const lo = Math.min(...history), hi = Math.max(...history), range = Math.max(hi - lo, 0.01);
  const points = history.map((value, index) => [history.length === 1 ? 280 : index / (history.length - 1) * 556 + 2, 74 - (value - lo) / range * 62]);
  const path = points.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  $('loss-line').setAttribute('d', history.length === 1 ? `${path} l1,0` : path);
  $('loss-area').setAttribute('d', `${path} L${points.at(-1)[0]},88 L${points[0][0]},88 Z`);
}

function renderChat() {
  const messages = Array.isArray(session.chat) ? session.chat : [];
  const signature = JSON.stringify(messages);
  if (signature === lastChat) return;
  lastChat = signature;
  if (!messages.length) return;
  const container = $('chat-messages');
  const keepAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 70;
  container.replaceChildren();
  for (const message of messages) {
    const entry = document.createElement('div');
    entry.className = `chat-message ${message.role === 'user' ? 'user' : 'assistant'}`;
    const role = document.createElement('div'); role.className = 'message-role'; role.textContent = message.role === 'user' ? 'You' : 'Neurograft · local';
    const content = document.createElement('div'); content.className = 'message-content'; content.textContent = String(message.content ?? '');
    entry.append(role, content); container.append(entry);
  }
  if (keepAtBottom || document.activeElement === $('chat-input')) container.scrollTop = container.scrollHeight;
}

function renderFacts() {
  const facts = session.facts || {};
  const signature = JSON.stringify(facts);
  if (signature === lastFacts) return;
  lastFacts = signature;
  const entries = Object.entries(facts);
  $('fact-count').textContent = entries.length;
  $('facts-list').replaceChildren();
  if (!entries.length) {
    const empty = document.createElement('p'); empty.className = 'empty-note'; empty.textContent = 'No facts yet. Use “remember key = value” in chat.'; $('facts-list').append(empty);
  }
  entries.forEach(([key, value]) => {
    const row = document.createElement('div'); row.className = 'fact-row';
    const label = document.createElement('span'); label.className = 'fact-key'; label.textContent = key;
    const detail = document.createElement('span'); detail.className = 'fact-value'; detail.textContent = String(value);
    row.append(label, detail); $('facts-list').append(row);
  });
}

function readableDate(value) {
  if (!value) return '';
  const date = new Date(typeof value === 'number' && value < 1e12 ? value * 1000 : value);
  return Number.isFinite(date.getTime()) ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
}

function renderPersistence() {
  const runtime = session.runtime || {};
  const autosave = finite(runtime.autosave_seconds);
  $('autosave-status').textContent = runtime.autosave_seconds === undefined ? 'Autosave status pending' : autosave > 0 ? `Autosave every ${formatNumber(autosave)} seconds` : 'Autosave disabled';
  if (runtime.last_error) {
    $('autosave-status').textContent = `Service notice: ${String(runtime.last_error)}`;
    $('autosave-status').title = String(runtime.last_error);
  } else $('autosave-status').title = '';
  const saved = session.last_saved;
  if (saved) {
    const path = typeof saved === 'string' ? saved : saved.path || saved.name || 'session';
    const name = String(path).split(/[\\/]/).at(-1);
    const time = typeof saved === 'object' ? readableDate(saved.time || saved.timestamp || saved.saved_at) : '';
    $('saved-status').textContent = `Last saved ${time ? `at ${time}` : ''} · ${name}`;
  } else $('saved-status').textContent = 'No save receipt yet.';
  const checkpoints = Array.isArray(session.checkpoints) ? session.checkpoints : [];
  const signature = JSON.stringify(checkpoints);
  if (signature === lastCheckpoints) return;
  lastCheckpoints = signature;
  $('checkpoint-count').textContent = checkpoints.length;
  $('checkpoint-options').replaceChildren();
  $('checkpoint-list').replaceChildren();
  if (!checkpoints.length) {
    const empty = document.createElement('span'); empty.className = 'empty-note'; empty.textContent = 'No checkpoints found.'; $('checkpoint-list').append(empty);
  }
  checkpoints.forEach((checkpoint) => {
    const option = document.createElement('option'); option.value = checkpoint.name; $('checkpoint-options').append(option);
    const item = document.createElement('button'); item.type = 'button'; item.className = 'checkpoint-item';
    const name = document.createElement('span'); name.textContent = checkpoint.name;
    const meta = document.createElement('small'); meta.textContent = `${formatNumber(finite(checkpoint.bytes) / 1024, 1)} KiB${checkpoint.modified ? ` · ${readableDate(checkpoint.modified)}` : ''}`;
    item.append(name, meta); item.addEventListener('click', () => { $('save-name').value = checkpoint.name; $('save-name').focus(); });
    $('checkpoint-list').append(item);
  });
}

function validCheckpointName() {
  const name = $('save-name').value.trim();
  if (!/^[A-Za-z0-9_-]{1,40}$/.test(name)) { showError('Choose a checkpoint name with 1–40 letters, numbers, underscores or hyphens.'); $('save-name').focus(); return null; }
  return name;
}

async function sendChat(event) {
  event?.preventDefault();
  const message = $('chat-input').value.trim();
  if (!message || writeInFlight || !session || networkError || isStale()) return;
  const result = await mutate('/api/chat', { message });
  if (result) { $('chat-input').value = ''; updateControls(); $('chat-messages').scrollTop = $('chat-messages').scrollHeight; $('chat-input').focus(); }
}

function bindControls() {
  $('pause-btn').addEventListener('click', () => control({ action: 'pause' }));
  $('resume-btn').addEventListener('click', () => control({ action: 'resume' }));
  $('step-btn').addEventListener('click', () => control({ action: 'step', steps: 1 }));
  if ($('boost-btn')) $('boost-btn').addEventListener('click', async () => {
    const res = await control({ action: 'boost', steps: 16 });
    if (res) showToast(`Executed 16-step evolution burst (step ${res.steps}, ${res.updates} updates).`);
  });
  $('learning-toggle').addEventListener('change', (event) => control({ action: 'learning', enabled: event.target.checked }));
  ['blue', 'amber', 'rose'].forEach((color) => $(`goal-${color}`).addEventListener('click', () => control({ action: 'goal', color })));
  document.querySelectorAll('.world-size-btn').forEach((btn) => btn.addEventListener('click', async () => {
    const size = Number(btn.dataset.worldSize);
    const preset = session?.telemetry?.environment?.preset || 'labyrinth';
    const res = await control({ action: 'world', size, preset });
    if (res) showToast(`Scaled 3D arena to ${size}×${size} (${preset} topology).`);
  }));
  document.querySelectorAll('.world-preset-btn').forEach((btn) => btn.addEventListener('click', async () => {
    const preset = btn.dataset.worldPreset;
    const size = session?.world?.size || 15;
    const res = await control({ action: 'world', size, preset });
    if (res) showToast(`Switched ${size}×${size} arena to ${preset} topology.`);
  }));
  $('dismiss-error').addEventListener('click', () => { $('error-banner').hidden = true; });
  $('chat-form').addEventListener('submit', sendChat);
  $('chat-input').addEventListener('input', updateControls);
  $('chat-input').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendChat(); } });
  document.querySelectorAll('[data-example]').forEach((button) => button.addEventListener('click', () => { $('chat-input').value = button.dataset.example; updateControls(); $('chat-input').focus(); }));
  $('save-btn').addEventListener('click', async () => {
    const name = validCheckpointName(); if (!name) return;
    const result = await mutate('/api/save', { name });
    if (result) showToast(`Checkpoint “${result.saved?.name || name}” saved.`);
  });
  $('load-btn').addEventListener('click', async () => {
    const name = validCheckpointName(); if (!name) return;
    const result = await mutate('/api/load', { name });
    if (result) showToast(`Restored checkpoint “${name}”.`);
  });
  $('camera-reset').addEventListener('click', resetCamera);
  $('fullscreen-btn').addEventListener('click', toggleFullscreen);
  document.addEventListener('keydown', (event) => {
    if (event.target.closest('input:not([type="checkbox"]), textarea, [contenteditable="true"]') || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key.toLowerCase() === 'f') { event.preventDefault(); toggleFullscreen(); return; }
    const keyActions = { ArrowUp: 0, w: 0, W: 0, ArrowRight: 1, d: 1, D: 1, ArrowDown: 2, s: 2, S: 2, ArrowLeft: 3, a: 3, A: 3 };
    const direction = keyActions[event.key];
    if (direction === undefined || !session) return;
    event.preventDefault();
    if (writeInFlight || networkError || isStale()) return;
    if (session.running) { if (!event.repeat) showToast('Pause the session to move with the arrow keys or WASD.'); return; }
    control({ action: 'move', direction });
  });
  document.addEventListener('fullscreenchange', resizeScene);
}

async function toggleFullscreen() {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('scene-wrap').requestFullscreen(); }
  catch { showError('Fullscreen is unavailable in this browser window.'); }
}

function resetCamera() {
  orbit.theta = 0.67; orbit.phi = 0.72; orbit.radius = worldSize * 1.65; orbit.targetX = 0; orbit.targetZ = 0;
  updateCamera();
}

function updateCamera() {
  if (!camera) return;
  camera.position.set(orbit.targetX + orbit.radius * Math.sin(orbit.phi) * Math.sin(orbit.theta), orbit.radius * Math.cos(orbit.phi), orbit.targetZ + orbit.radius * Math.sin(orbit.phi) * Math.cos(orbit.theta));
  camera.lookAt(orbit.targetX, 0, orbit.targetZ);
  sceneDirty = true;
}

function resizeScene() {
  if (!renderer || !camera) return;
  const rect = $('scene-wrap').getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  updateCamera();
}

function disposeGroup(group) {
  while (group.children.length) {
    const child = group.children[0];
    child.traverse((object) => {
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
      else object.material?.dispose();
    });
    group.remove(child);
  }
}

let pathLine = null;
let haloGroup = null;
let chokeGroup = null;
let lidarGroup = null;
const haloNodes = [];
const haloBridges = [];

function buildWorld(world) {
  disposeGroup(worldGroup);
  beaconPickables.length = 0;
  haloNodes.length = 0;
  haloBridges.length = 0;
  worldSize = world.size;
  if (scene?.fog) {
    scene.fog.near = Math.max(22, worldSize * 1.4);
    scene.fog.far = Math.max(58, worldSize * 3.8);
  }
  if (camera) {
    camera.far = Math.max(120, worldSize * 5.0);
    camera.updateProjectionMatrix();
  }
  const offset = (worldSize - 1) / 2;
  const material = (color, options = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.25, ...options });
  const floor = new THREE.Mesh(new THREE.BoxGeometry(worldSize + 0.25, 0.28, worldSize + 0.25), material('#152b3b'));
  floor.position.y = -0.2; floor.receiveShadow = true; worldGroup.add(floor);
  floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(worldSize, worldSize), material('#1b3545', { roughness: 0.88, metalness: 0.08 }));
  floorMesh.rotation.x = -Math.PI / 2; floorMesh.position.y = -0.048; floorMesh.receiveShadow = true; worldGroup.add(floorMesh);
  const grid = new THREE.GridHelper(worldSize, worldSize, '#365969', '#2c4a5d');
  grid.position.y = -0.04; grid.material.transparent = true; grid.material.opacity = 0.65; worldGroup.add(grid);
  const edge = new THREE.EdgesGeometry(floor.geometry);
  const outline = new THREE.LineSegments(edge, new THREE.LineBasicMaterial({ color: '#457281', transparent: true, opacity: 0.5 }));
  outline.position.copy(floor.position); worldGroup.add(outline);
  wallGroup = new THREE.Group(); worldGroup.add(wallGroup);
  for (const obstacle of world.obstacles || []) {
    const isBorder = obstacle.x === 0 || obstacle.z === 0 || obstacle.x === worldSize - 1 || obstacle.z === worldSize - 1;
    const h = isBorder ? 0.58 : 0.78;
    const wallColor = isBorder ? '#22384a' : '#2e4d66';
    const wall = new THREE.Mesh(new THREE.BoxGeometry(0.94, h, 0.94), material(wallColor, { roughness: 0.45, metalness: 0.32 }));
    wall.position.set(obstacle.x - offset, h / 2 - 0.05, obstacle.z - offset); wall.castShadow = true; wall.receiveShadow = true; wallGroup.add(wall);
    const cap = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.94, h, 0.94)), new THREE.LineBasicMaterial({ color: isBorder ? '#456478' : '#6393b0', transparent: true, opacity: 0.56 }));
    cap.position.copy(wall.position); wallGroup.add(cap);
  }
  beaconGroup = new THREE.Group(); worldGroup.add(beaconGroup);
  for (const beacon of world.beacons || []) {
    const color = BEACON_COLORS[beacon.color] || '#9fb6c9';
    const group = new THREE.Group(); group.position.set(beacon.x - offset, 0, beacon.z - offset); group.userData.beacon = beacon;
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.31, 0.1, 24), material(color, { emissive: color, emissiveIntensity: 0.12 }));
    base.position.y = 0.03; group.add(base);
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.5, 12), material(color, { emissive: color, emissiveIntensity: 0.3 }));
    post.position.y = 0.3; group.add(post);
    const crystal = new THREE.Mesh(new THREE.OctahedronGeometry(0.21), material(color, { emissive: color, emissiveIntensity: 0.55, metalness: 0.2, roughness: 0.25 }));
    crystal.position.y = 0.65; group.add(crystal);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.32, 0.36, 40), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.015; group.add(ring);
    const plume = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.95, 40), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, side: THREE.DoubleSide }));
    plume.rotation.x = -Math.PI / 2; plume.position.y = 0.01; group.add(plume);
    group.traverse((object) => { if (object.isMesh) { object.userData.beacon = beacon; beaconPickables.push(object); } });
    beaconGroup.add(group);
  }
  agentGroup = new THREE.Group(); worldGroup.add(agentGroup);
  const agentBody = new THREE.Mesh(new THREE.SphereGeometry(0.23, 24, 16), material('#6af4d2', { emissive: '#3ce4ba', emissiveIntensity: 0.35, metalness: 0.4, roughness: 0.25 }));
  agentBody.position.y = 0.25; agentBody.scale.y = 0.8; agentBody.castShadow = true; agentGroup.add(agentBody);
  const agentHalo = new THREE.Mesh(new THREE.RingGeometry(0.29, 0.33, 36), new THREE.MeshBasicMaterial({ color: '#5ee9ce', transparent: true, opacity: 0.65, side: THREE.DoubleSide }));
  agentHalo.rotation.x = -Math.PI / 2; agentHalo.position.y = 0.02; agentGroup.add(agentHalo);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.27, 3), material('#d9fff1', { emissive: '#82f7d0', emissiveIntensity: 0.25 }));
  nose.rotation.x = -Math.PI / 2; nose.position.set(0, 0.24, -0.31); agentGroup.add(nose);
  goalGroup = new THREE.Group(); worldGroup.add(goalGroup);
  const target = new THREE.Mesh(new THREE.RingGeometry(0.4, 0.44, 56), new THREE.MeshBasicMaterial({ color: '#69b5ff', transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
  target.rotation.x = -Math.PI / 2; target.position.y = 0.025; goalGroup.add(target);
  const outerTarget = new THREE.Mesh(new THREE.RingGeometry(0.48, 0.49, 56), new THREE.MeshBasicMaterial({ color: '#69b5ff', transparent: true, opacity: 0.32, side: THREE.DoubleSide }));
  outerTarget.rotation.x = -Math.PI / 2; outerTarget.position.y = 0.026; goalGroup.add(outerTarget);
  trailLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#5ee9ce', transparent: true, opacity: 0.35 }));
  worldGroup.add(trailLine);
  pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#f5be66', transparent: true, opacity: 0.68 }));
  worldGroup.add(pathLine);
  chokeGroup = new THREE.Group(); worldGroup.add(chokeGroup);
  lidarGroup = new THREE.Group(); worldGroup.add(lidarGroup);
  haloGroup = new THREE.Group(); haloGroup.position.y = Math.max(2.4, worldSize * 0.17); worldGroup.add(haloGroup);
  const haloRadius = Math.max(2.15, worldSize * 0.18);
  const positions = SOURCE_COLORS.map((_, idx) => {
    const angle = idx * Math.PI / 2 + Math.PI / 4;
    return new THREE.Vector3(Math.cos(angle) * haloRadius, 0, Math.sin(angle) * haloRadius);
  });
  SOURCE_COLORS.forEach((color, idx) => {
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), material(color, { emissive: color, emissiveIntensity: 0.58 }));
    sphere.position.copy(positions[idx]);
    haloGroup.add(sphere);
    haloNodes.push(sphere);
  });
  for (let dst = 0; dst < 4; dst += 1) {
    for (let src = 0; src < 4; src += 1) {
      if (src === dst) continue;
      const mid = positions[src].clone().add(positions[dst]).multiplyScalar(0.5);
      mid.y += 0.32;
      const curve = new THREE.QuadraticBezierCurve3(positions[src], mid, positions[dst]);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(curve.getPoints(16)),
        new THREE.LineBasicMaterial({ color: SOURCE_COLORS[src], transparent: true, opacity: 0.15 })
      );
      haloGroup.add(line);
      haloBridges.push({ dst, src, line });
    }
  }
  const shadowGround = new THREE.Mesh(new THREE.PlaneGeometry(140, 140), material('#0b1521', { roughness: 1, metalness: 0 }));
  shadowGround.rotation.x = -Math.PI / 2; shadowGround.position.y = -0.36; shadowGround.receiveShadow = true; worldGroup.add(shadowGround);
  agentGroup.position.set(world.agent.x - offset, 0, world.agent.z - offset);
  agentGroup.rotation.y = -finite(world.agent.direction) * Math.PI / 2;
  resetCamera();
}

function updateWorld(world) {
  const structuralKey = JSON.stringify([world.size, world.obstacles, world.beacons]);
  if (structuralKey !== renderedWorldKey) { renderedWorldKey = structuralKey; buildWorld(world); lastWorldVisualKey = ''; }
  const offset = (world.size - 1) / 2;
  agentTarget.x = world.agent.x - offset;
  agentTarget.z = world.agent.z - offset;
  agentTarget.angle = -finite(world.agent.direction) * Math.PI / 2;
  goalGroup.position.set(world.goal.x - offset, 0, world.goal.z - offset);
  const goalColor = BEACON_COLORS[world.goal.color] || '#b9dded';
  goalGroup.children.forEach((object) => object.material.color.set(goalColor));
  const expertPath = session.telemetry?.expert_path || [];
  const env = session.telemetry?.environment || {};
  const chokePoints = env.choke_points || [];
  const lidarRays = env.lidar_rays || [];
  const visualKey = JSON.stringify([world.trail, expertPath, chokePoints, lidarRays, session.routes, session.activity]);
  if (visualKey !== lastWorldVisualKey) {
    lastWorldVisualKey = visualKey;
    const points = (world.trail || []).slice(-64).map((point) => new THREE.Vector3(point.x - offset, 0.035, point.z - offset));
    trailLine.geometry.dispose(); trailLine.geometry = new THREE.BufferGeometry().setFromPoints(points);
    if (pathLine) {
      const routePts = expertPath.map((point) => new THREE.Vector3(point.x - offset, 0.065, point.z - offset));
      pathLine.geometry.dispose(); pathLine.geometry = new THREE.BufferGeometry().setFromPoints(routePts);
      pathLine.material.color.set(goalColor);
    }
    if (chokeGroup) {
      disposeGroup(chokeGroup);
      for (const cp of chokePoints.slice(0, 12)) {
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(0.18, 0.25, 24),
          new THREE.MeshBasicMaterial({ color: '#ad9af6', transparent: true, opacity: 0.55, side: THREE.DoubleSide })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(cp.x - offset, 0.045, cp.z - offset);
        chokeGroup.add(ring);
      }
    }
    if (lidarGroup) {
      disposeGroup(lidarGroup);
      const ax = world.agent.x - offset, az = world.agent.z - offset;
      for (const ray of lidarRays) {
        const rx = ray.free_x - offset, rz = ray.free_z - offset;
        const beam = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(ax, 0.18, az), new THREE.Vector3(rx, 0.12, rz)]),
          new THREE.LineBasicMaterial({ color: '#5ee9ce', transparent: true, opacity: 0.22 })
        );
        lidarGroup.add(beam);
      }
    }
    haloNodes.forEach((node, idx) => {
      const act = clip(finite(session.activity?.[idx]), 0, 2);
      const s = 0.85 + act * 0.65;
      node.scale.set(s, s, s);
    });
    haloBridges.forEach(({ dst, src, line }) => {
      const w = clip(finite(session.routes?.[dst]?.[src]), 0, 1);
      line.material.opacity = 0.06 + w * 0.78;
    });
  }
  sceneDirty = true;
  $('scene-loading').hidden = true;
}

function bindScenePointer() {
  const canvas = $('scene-canvas');
  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    pointer.down = true; pointer.id = event.pointerId; pointer.x = pointer.startX = event.clientX; pointer.y = pointer.startY = event.clientY; pointer.dragged = false;
    canvas.setPointerCapture(event.pointerId); canvas.focus({ preventScroll: true });
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!pointer.down || pointer.id !== event.pointerId) return;
    const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    if (Math.hypot(event.clientX - pointer.startX, event.clientY - pointer.startY) > 5) pointer.dragged = true;
    if (pointer.dragged) { orbit.theta -= dx * 0.007; orbit.phi = clip(orbit.phi + dy * 0.006, 0.22, 1.3); updateCamera(); }
    pointer.x = event.clientX; pointer.y = event.clientY;
  });
  canvas.addEventListener('pointerup', (event) => {
    if (!pointer.down || pointer.id !== event.pointerId) return;
    pointer.down = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (!pointer.dragged) selectGoalAt(event.clientX, event.clientY);
  });
  canvas.addEventListener('pointercancel', () => { pointer.down = false; });
  canvas.addEventListener('wheel', (event) => { event.preventDefault(); orbit.radius = clip(orbit.radius * Math.exp(event.deltaY * 0.001), worldSize * 0.8, worldSize * 3.3); updateCamera(); }, { passive: false });
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
}

function selectGoalAt(clientX, clientY) {
  if (!webglReady || !session || writeInFlight || networkError || isStale()) return;
  const rect = $('scene-canvas').getBoundingClientRect();
  pointerVector.set((clientX - rect.left) / rect.width * 2 - 1, -(clientY - rect.top) / rect.height * 2 + 1);
  raycaster.setFromCamera(pointerVector, camera);
  const beaconHits = raycaster.intersectObjects(beaconPickables);
  if (beaconHits.length) { control({ action: 'goal', color: beaconHits[0].object.userData.beacon.color }); return; }
  const hits = raycaster.intersectObject(floorMesh);
  if (!hits.length) return;
  const offset = (session.world.size - 1) / 2;
  const x = Math.round(hits[0].point.x + offset), z = Math.round(hits[0].point.z + offset);
  if (x < 0 || z < 0 || x >= session.world.size || z >= session.world.size || (session.world.obstacles || []).some((wall) => wall.x === x && wall.z === z)) { showToast('Choose a free floor cell. Walls cannot be targets.'); return; }
  const beacon = (session.world.beacons || []).find((entry) => entry.x === x && entry.z === z);
  control(beacon ? { action: 'goal', color: beacon.color } : { action: 'goal', x, z });
}

async function initScene() {
  try {
    THREE = await import('/vendor/three.module.js');
    renderer = new THREE.WebGLRenderer({ canvas: $('scene-canvas'), antialias: !TEST_MODE, alpha: false, preserveDrawingBuffer: true, powerPreference: 'low-power' });
    renderer.setPixelRatio(TEST_MODE ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.35;
    scene = new THREE.Scene(); scene.background = new THREE.Color('#0b1521'); scene.fog = new THREE.Fog('#0b1521', 24, 57);
    camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    scene.add(new THREE.HemisphereLight('#b9e4f7', '#132f39', 2.5));
    const keyLight = new THREE.DirectionalLight('#d3efff', 3.8); keyLight.position.set(-5, 11, 6); keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(512, 512); keyLight.shadow.camera.left = -9; keyLight.shadow.camera.right = 9; keyLight.shadow.camera.top = 9; keyLight.shadow.camera.bottom = -9; keyLight.shadow.normalBias = 0.03; scene.add(keyLight);
    const fill = new THREE.DirectionalLight('#57cbbd', 1.4); fill.position.set(6, 5, -8); scene.add(fill);
    worldGroup = new THREE.Group(); scene.add(worldGroup);
    raycaster = new THREE.Raycaster(); pointerVector = new THREE.Vector2();
    new ResizeObserver(resizeScene).observe($('scene-wrap'));
    window.addEventListener('resize', resizeScene); resizeScene(); bindScenePointer();
    $('scene-canvas').addEventListener('webglcontextlost', (event) => { event.preventDefault(); webglReady = false; webglError = 'WebGL context lost'; showError('The 3D renderer lost its graphics context. Reload to restore the view; the server session continues.'); });
    webglReady = true;
    if (session) updateWorld(session.world);
    let lastFrame = performance.now();
    function renderFrame(now) {
      requestAnimationFrame(renderFrame);
      if (!webglReady) return;
      const dt = Math.min((now - lastFrame) / 1000, 0.1); lastFrame = now;
      let moving = false;
      if (agentGroup) {
        const dx = agentTarget.x - agentGroup.position.x;
        const dz = agentTarget.z - agentGroup.position.z;
        let angle = agentTarget.angle - agentGroup.rotation.y;
        angle = Math.atan2(Math.sin(angle), Math.cos(angle));
        if (Math.hypot(dx, dz) > 1e-3 || Math.abs(angle) > 1e-3) {
          const easing = (reducedMotion || TEST_MODE) ? 1 : 1 - Math.exp(-dt * 15);
          agentGroup.position.x += dx * easing;
          agentGroup.position.z += dz * easing;
          agentGroup.rotation.y += angle * easing;
          moving = true;
        }
      }
      if (!sceneDirty && !moving) return;
      sceneDirty = false;
      // Visual interpolation only. Server inference/training never runs here.
      renderer.render(scene, camera);
    }
    requestAnimationFrame(renderFrame);
  } catch (error) {
    webglError = error.message || String(error);
    $('scene-loading').querySelector('strong').textContent = 'The 3D renderer is unavailable';
    $('scene-loading').querySelector('span:last-child').textContent = 'WebGL and the bundled local Three.js files are required. Session controls and chat remain available.';
    showError(`3D view: ${webglError}`);
  }
}

window.render_game_to_text = () => JSON.stringify({
  mode: !session ? 'connecting' : networkError ? 'disconnected' : isStale() ? 'stale' : session.running ? 'running' : 'paused',
  coordinates: 'integer cells; origin northwest; x east/right, z south/down; y vertical; actions 0 north, 1 east, 2 south, 3 west',
  connected: !networkError && !isStale(), renderer: webglReady ? 'three-webgl' : webglError ? 'unavailable' : 'loading', test_mode: TEST_MODE,
  running: session?.running ?? null, learning: session?.learning ?? null, busy: writeInFlight,
  world: session?.world ?? null, steps: session?.steps ?? null, updates: session?.updates ?? null,
  goals_reached: session?.goals_reached ?? null, episodes: session?.episodes ?? null,
  last_loss: session?.last_loss ?? null, policy_accuracy: session?.policy_accuracy ?? null,
  last_action: session?.last_action ?? null, last_action_source: session?.last_action_source ?? null,
  backend: session?.backend || 'local-memory-and-control', facts: session?.facts || {},
  latest_chat: session?.chat?.slice(-2) || [], checkpoints: session?.checkpoints?.map((entry) => entry.name) || [],
  last_saved: session?.last_saved ?? null, error: $('error-banner').hidden ? null : $('error-message').textContent,
});

let testAdvanceQueue = Promise.resolve();
let testAdvanceRemainderMs = 0;
window.advanceTime = (ms) => {
  if (!TEST_MODE) return Promise.resolve(window.render_game_to_text());
  // Test requests are serialized, bounded, and paused before explicit stepping.
  testAdvanceQueue = testAdvanceQueue.catch(() => {}).then(async () => {
    if (!Number.isFinite(ms) || ms < 0 || ms > 10000) throw new Error('advanceTime expects 0–10000 milliseconds.');
    while (writeInFlight) await new Promise((resolve) => setTimeout(resolve, 15));
    if (!session) {
      const state = await fetchJSON('/api/state'); applySnapshot(state);
    }
    if (session.running && !await control({ action: 'pause' })) throw new Error('Could not pause the test session.');
    const intervalMs = Math.max(1, finite(session.runtime?.interval_seconds, 0.25) * 1000);
    testAdvanceRemainderMs += ms;
    const steps = Math.min(32, Math.floor(testAdvanceRemainderMs / intervalMs));
    if (steps > 0) {
      testAdvanceRemainderMs -= steps * intervalMs;
      if (!await control({ action: 'step', steps })) throw new Error('Could not advance the test session.');
    }
    if (agentGroup) { agentGroup.position.x = agentTarget.x; agentGroup.position.z = agentTarget.z; agentGroup.rotation.y = agentTarget.angle; }
    if (renderer && webglReady) renderer.render(scene, camera);
    return window.render_game_to_text();
  });
  return testAdvanceQueue;
};

makeSourceRows();
bindControls();
updateControls();
initScene();
poll();
setInterval(updateConnectivity, 1000);
