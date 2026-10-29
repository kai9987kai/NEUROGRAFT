"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const schedules = ["prioritized", "uniform", "none"];
  const scheduleNames = { prioritized: "Prioritized", uniform: "Cyclic uniform", none: "No replay" };
  const directions = [{ name: "North", short: "N", delta: [0, -1] }, { name: "East", short: "E", delta: [1, 0] }, { name: "South", short: "S", delta: [0, 1] }, { name: "West", short: "W", delta: [-1, 0] }];
  const state = { snapshot: null, model: "recent", schedule: "prioritized", inspected: null, busy: false, playing: false, batchesRemaining: 0, playTimer: null, ready: false };
  const key = position => position.join(",");
  const positionText = position => `(${position[0]}, ${position[1]})`;
  const number = value => typeof value === "number" && Number.isFinite(value) ? value : null;
  const metric = value => number(value) === null ? "—" : Math.abs(value) > 0 && Math.abs(value) < 0.0001 ? value.toExponential(2) : value.toFixed(4);
  const count = value => number(value) === null ? "—" : value.toLocaleString("en-GB");
  const percentage = value => number(value) === null ? "N/A" : `${(value * 100).toFixed(1)}%`;
  const setText = (id, text) => { $(id).textContent = text; };
  const announce = message => setText("workbench-status", message);
  const svgElement = (tag, attributes = {}, text = null) => {
    const element = document.createElementNS(NS, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
    if (text !== null) element.textContent = text;
    return element;
  };

  function fail(error) {
    stopPlay(false);
    setText("error-message", error instanceof Error ? error.message : String(error));
    $("error-banner").hidden = false;
    announce("Command was not completed. Review the error above; the last received snapshot remains visible.");
  }

  function clearError() { $("error-banner").hidden = true; }

  function updateControls() {
    document.querySelectorAll("[data-mutation]").forEach(element => {
      element.disabled = state.busy || !state.ready || state.playing;
    });
    $("play-btn").hidden = state.playing;
    $("pause-btn").hidden = !state.playing;
    $("pause-btn").disabled = false;
    if (state.playing) {
      setText("play-state", `Playing · ${state.batchesRemaining} of 8 batches remaining${state.busy ? " · current request in flight" : ""}`);
    } else {
      setText("play-state", state.busy ? "Paused · current request is finishing; no next batch is scheduled" : "Paused · each command applies to all six planners");
    }
  }

  async function request(path, options = {}) {
    if (state.busy) throw new Error("Another workbench request is still in flight.");
    state.busy = true;
    updateControls();
    try {
      const response = await fetch(path, { cache: "no-store", ...options });
      const type = response.headers.get("content-type") || "";
      let body;
      if (type.includes("application/json")) body = await response.json();
      else body = await response.text();
      if (!response.ok) {
        const detail = typeof body === "string" ? body.slice(0, 400) : body.error || body.message || JSON.stringify(body);
        throw new Error(`${response.status}: ${detail}`);
      }
      return body;
    } finally {
      state.busy = false;
      updateControls();
    }
  }

  function acceptSnapshot(candidate) {
    // Mutations return the same public snapshot as GET; tolerate an explicit wrapper.
    const snapshot = candidate.snapshot || candidate.workbench || candidate;
    if (!snapshot.models || !snapshot.arms || !Array.isArray(snapshot.goal)) throw new Error("The server returned an unsupported workbench snapshot.");
    state.snapshot = snapshot;
    if (!state.ready) $("window-select").value = String(snapshot.window);
    state.ready = true;
    if (!state.inspected) {
      const first = snapshot.fixture?.start || snapshot.models[state.model].nodes.find(position => key(position) !== key(snapshot.goal)) || snapshot.models[state.model].nodes[0] || snapshot.goal;
      state.inspected = [...first];
    }
    $("connection-state").className = "connection ready";
    setText("connection-state", "Local workbench");
    render();
    updateControls();
  }

  async function command(payload, message) {
    clearError();
    try {
      const snapshot = await request("/api/workbench", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (["fixture", "capture", "import", "reset"].includes(payload.action)) state.inspected = null;
      acceptSnapshot(snapshot);
      if (["fixture", "capture", "import"].includes(payload.action)) $("window-select").value = String(state.snapshot.window);
      if (message) announce(message);
      return true;
    } catch (error) { fail(error); return false; }
  }

  function stopPlay(notify = true) {
    state.playing = false;
    state.batchesRemaining = 0;
    if (state.playTimer !== null) clearTimeout(state.playTimer);
    state.playTimer = null;
    updateControls();
    if (notify) announce(state.busy ? "Paused. The current bounded request may finish; no further batch will start." : "Replay paused. No batch is scheduled.");
  }

  async function playBatch() {
    state.playTimer = null;
    if (!state.playing || state.busy || state.batchesRemaining <= 0) return;
    const ok = await command({ action: "replay", budget: integerInput("replay-budget", 0, 256) }, "Replay batch applied to all six planners.");
    if (!ok || !state.playing) return;
    state.batchesRemaining -= 1;
    if (state.batchesRemaining <= 0) {
      stopPlay(false);
      announce("Bounded play complete: 8 batches applied. Inspect the actual backup counts and empirical errors.");
    } else {
      updateControls();
      state.playTimer = setTimeout(playBatch, 500);
    }
  }

  function integerInput(id, min, max) {
    const input = $(id);
    const value = Number(input.value);
    if (input.value.trim() === "" || !Number.isInteger(value) || value < min || value > max) throw new Error(`${input.labels?.[0]?.textContent?.trim() || id} must be an integer from ${min} to ${max}.`);
    return value;
  }

  function withInput(callback) {
    return () => { try { callback(); } catch (error) { fail(error); } };
  }

  function provenanceLabel(provenance) {
    if (typeof provenance === "string") return provenance;
    const serialized = JSON.stringify(provenance || {}).toLowerCase();
    const seed = provenance && typeof provenance === "object" ? provenance.seed : null;
    if (/synthetic|fixture/.test(serialized)) return `Synthetic fixture${seed === null || seed === undefined ? "" : ` · seed ${seed}`}`;
    if (/capture|live/.test(serialized)) return "Frozen live graph capture";
    return "Supplied frozen baseline";
  }

  function render() {
    const snapshot = state.snapshot;
    if (!snapshot) return;
    setText("provenance-label", provenanceLabel(snapshot.provenance));
    setText("experiment-counts", `${count(snapshot.models.cumulative.nodes.length)} cells at cumulative model · ${count(snapshot.events)} supplied events`);
    setText("operation-count", `${count(snapshot.operations)} / 128 operations`);
    setText("goal-badge", `Goal ${positionText(snapshot.goal)}${snapshot.models[state.model].nodes.some(position => key(position) === key(snapshot.goal)) ? "" : " · unknown"}`);
    setText("event-count", `${count(snapshot.events)} supplied events`);
    setText("selected-model-label", state.model === "recent" ? "Recent model" : "Cumulative model");
    setText("model-note", state.model === "recent" ? `First fresh outcome replaces that action’s frozen baseline. Thereafter, only its last ${snapshot.window} new outcomes count; untouched actions retain the aggregate.` : "All frozen baseline counts and all supplied outcomes accumulate together. Earlier outcomes remain in the model after an action changes.");
    for (const model of ["cumulative", "recent"]) {
      $("model-" + model).setAttribute("aria-pressed", String(state.model === model));
      $("model-" + model).classList.toggle("active", state.model === model);
    }
    for (const schedule of schedules) {
      const arm = snapshot.arms[state.model][schedule];
      const metrics = arm.metrics || {};
      $("schedule-" + schedule).classList.toggle("selected", state.schedule === schedule);
      $("schedule-" + schedule).setAttribute("aria-pressed", String(state.schedule === schedule));
      setText("mae-" + schedule, metric(metrics.value_mae));
      setText("residual-" + schedule, metric(metrics.bellman_residual));
      setText("agreement-" + schedule, percentage(metrics.action_agreement));
      setText("backups-" + schedule, count(arm.total_backups));
      setText("state-" + schedule, arm.stale ? "Stale values" : "Current model");
      $("state-" + schedule).classList.toggle("stale", Boolean(arm.stale));
      setText("detail-" + schedule, `${count(metrics.decisive_states)} decisive states${state.schedule === schedule ? " · heatmap selected" : " · view heatmap"}`);
    }
    const scope = snapshot.evidence_scope;
    if (typeof scope === "string") setText("evidence-scope", scope);
    else if (Array.isArray(scope)) setText("evidence-scope", scope.join(" "));
    else if (scope) setText("evidence-scope", Object.values(scope).map(value => typeof value === "string" ? value : JSON.stringify(value)).join(" "));
    setText("provenance-detail", JSON.stringify({ provenance: snapshot.provenance, cumulative_sha256: snapshot.models.cumulative.sha256, recent_sha256: snapshot.models.recent.sha256 }, null, 2));
    renderGraph();
    renderInspector();
    renderHistory();
  }

  function heatColor(value, maximum) {
    const ratio = Math.max(0, Math.min(1, maximum > 0 ? value / maximum : 0));
    const start = [20, 43, 62], end = [47, 142, 123];
    return `rgb(${start.map((component, index) => Math.round(component + (end[index] - component) * ratio)).join(",")})`;
  }

  function renderGraph() {
    const snapshot = state.snapshot, model = snapshot.models[state.model], arm = snapshot.arms[state.model][state.schedule];
    const graph = $("transition-graph");
    const focusedPosition = document.activeElement?.getAttribute("data-position");
    graph.replaceChildren();
    graph.append(svgElement("title", { id: "graph-title" }, `${state.model} model, ${scheduleNames[state.schedule]} replay values`), svgElement("desc", { id: "graph-description" }, "Observed cells are colored by retained value. Arrows show empirical direction and outcome counts. Click a cell or press Enter on a focused cell to inspect it. Arrow keys inspect coordinates. The goal button sets the inspected cell as the target."));
    const nodes = model.nodes, positions = [...nodes];
    if (!positions.some(position => key(position) === key(snapshot.goal))) positions.push(snapshot.goal);
    if (state.inspected && !positions.some(position => key(position) === key(state.inspected))) positions.push(state.inspected);
    $("graph-empty").hidden = nodes.length > 0;
    if (!positions.length) return;
    const xs = [...new Set(positions.map(position => position[0]))].sort((a, b) => a - b), zs = [...new Set(positions.map(position => position[1]))].sort((a, b) => a - b);
    const compressed = [xs, zs].some(axis => axis.some((coordinate, index) => index > 0 && coordinate - axis[index - 1] > 1));
    const columns = xs.length, rows = zs.length;
    const spacing = 76, cellSize = 49, padding = 48;
    const width = Math.max(380, columns * spacing + padding * 2 - spacing), height = Math.max(290, rows * spacing + padding * 2 - spacing);
    graph.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const offsetX = (width - (columns - 1) * spacing) / 2, offsetZ = (height - (rows - 1) * spacing) / 2;
    const center = position => [offsetX + xs.indexOf(position[0]) * spacing, offsetZ + zs.indexOf(position[1]) * spacing];
    const defs = svgElement("defs");
    for (const [id, fill] of [["edge-arrow", "#668b9c"], ["edge-arrow-selected", "#e2a669"]]) {
      const marker = svgElement("marker", { id, viewBox: "0 0 10 10", refX: "8", refY: "5", markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse" });
      marker.append(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill }));
      defs.append(marker);
    }
    graph.append(defs);
    const edgesGroup = svgElement("g", { "aria-hidden": "true" });
    const labelsGroup = svgElement("g", { "aria-hidden": "true" });
    for (const edge of model.edges) {
      const source = center(edge.source), target = center(edge.destination);
      const isInspected = key(edge.source) === key(state.inspected);
      const collision = key(edge.source) === key(edge.destination);
      const direction = directions[edge.action] || { short: String(edge.action) };
      let path, labelX, labelZ;
      if (collision) {
        const actionDelta = directions[edge.action]?.delta || [1, 0];
        const ax = actionDelta[0], az = actionDelta[1], crossX = -az, crossZ = ax;
        const startX = source[0] + ax * 18 + crossX * 10, startZ = source[1] + az * 18 + crossZ * 10;
        const endX = source[0] + ax * 18 - crossX * 10, endZ = source[1] + az * 18 - crossZ * 10;
        path = `M ${startX} ${startZ} C ${source[0] + ax * 46 + crossX * 24} ${source[1] + az * 46 + crossZ * 24}, ${source[0] + ax * 46 - crossX * 24} ${source[1] + az * 46 - crossZ * 24}, ${endX} ${endZ}`;
        labelX = source[0] + ax * 48; labelZ = source[1] + az * 48 + 3;
      } else {
        const dx = target[0] - source[0], dz = target[1] - source[1], length = Math.hypot(dx, dz) || 1;
        const ux = dx / length, uz = dz / length, shiftX = -uz * 5, shiftZ = ux * 5;
        path = `M ${source[0] + ux * 27 + shiftX} ${source[1] + uz * 27 + shiftZ} L ${target[0] - ux * 29 + shiftX} ${target[1] - uz * 29 + shiftZ}`;
        labelX = (source[0] + target[0]) / 2 - uz * 13; labelZ = (source[1] + target[1]) / 2 + ux * 13 + 3;
      }
      edgesGroup.append(svgElement("path", { d: path, class: `edge${isInspected ? " selected-edge" : ""}`, "marker-end": `url(#${isInspected ? "edge-arrow-selected" : "edge-arrow"})` }));
      labelsGroup.append(svgElement("text", { x: labelX, y: labelZ, "text-anchor": "middle", class: `edge-label${isInspected ? " selected-label" : ""}` }, `${direction.short}·${edge.count}`));
    }
    graph.append(edgesGroup, labelsGroup);
    const values = new Map(arm.values.map(item => [key(item.position), item.value]));
    const maximum = Math.max(0, ...arm.values.map(item => number(item.value) || 0));
    setText("heatmap-range", `Value 0 — ${maximum.toFixed(3)} · ${scheduleNames[state.schedule]}`);
    for (const position of positions) {
      const known = nodes.some(node => key(node) === key(position));
      const isGoal = key(position) === key(snapshot.goal), selected = key(position) === key(state.inspected);
      const value = values.get(key(position));
      const [x, z] = center(position);
      const cell = svgElement("g", { class: `node-cell${selected ? " node-selected" : ""}${isGoal ? " node-goal" : ""}`, tabindex: "0", role: "button", "data-position": key(position), "aria-label": `Inspect ${positionText(position)}${isGoal ? ", goal" : ""}, ${known ? `value ${metric(value)}` : "unobserved cell"}` });
      cell.append(svgElement("rect", { x: x - cellSize / 2, y: z - cellSize / 2, width: cellSize, height: cellSize, rx: 7, fill: heatColor(number(value) || 0, maximum), class: known ? "cell-background" : "cell-background unobserved-goal" }));
      cell.append(svgElement("text", { x, y: z - 7, "text-anchor": "middle", class: "cell-coordinate" }, `${position[0]}, ${position[1]}`));
      cell.append(svgElement("text", { x, y: z + 10, "text-anchor": "middle", class: isGoal ? "goal-label" : "cell-value" }, isGoal ? `GOAL${known ? "" : " ?"}` : known ? (number(value) === null ? "—" : value.toFixed(2)) : "?"));
      cell.addEventListener("click", () => inspect(position));
      cell.addEventListener("keydown", event => {
        if (event.key === " " || event.key === "Enter") { event.preventDefault(); event.stopPropagation(); inspect(position); }
      });
      graph.append(cell);
    }
    if (focusedPosition) {
      const focused = [...graph.querySelectorAll("[data-position]")].find(element => element.getAttribute("data-position") === focusedPosition);
      focused?.focus({ preventScroll: true });
    }
    setText("graph-counts", `${count(nodes.length)} observed cells · ${count(model.total_transitions)} counted outcomes · ${count(model.edges.length)} edges${compressed ? " · coordinate gaps compressed" : ""}`);
  }

  function inspect(position) {
    state.inspected = position.map(coordinate => Math.max(0, Math.min(30, coordinate)));
    clearError();
    if (state.snapshot) { renderGraph(); renderInspector(); }
  }

  function renderInspector() {
    const position = state.inspected, snapshot = state.snapshot;
    if (!position || !snapshot) return;
    const model = snapshot.models[state.model], arm = snapshot.arms[state.model][state.schedule];
    const known = model.nodes.some(node => key(node) === key(position));
    setText("inspected-position", positionText(position));
    setText("inspected-known", known ? "Observed" : "Unknown");
    $("inspect-x").value = position[0]; $("inspect-z").value = position[1];
    setText("inspected-schedule", `${scheduleNames[state.schedule]} / ${state.model}`);
    const value = arm.values.find(item => key(item.position) === key(position));
    setText("inspected-value", value ? metric(value.value) : "—");
    const outcomes = model.edges.filter(edge => key(edge.source) === key(position));
    setText("inspect-note", outcomes.length ? "Counted action outcomes in this empirical model:" : known ? "No outgoing outcomes for this cell. No observed action can be inferred." : "This coordinate has no empirical evidence. Setting it as a goal does not establish a route.");
    const list = $("inspected-outcomes"); list.replaceChildren();
    for (const direction of directions) {
      const action = directions.indexOf(direction), edges = outcomes.filter(edge => edge.action === action);
      if (!edges.length) continue;
      const row = document.createElement("div"); row.className = "outcome-row";
      const name = document.createElement("strong"); name.textContent = direction.name;
      const detail = document.createElement("span"); detail.textContent = edges.map(edge => `${positionText(edge.destination)} ×${edge.count}${key(edge.destination) === key(position) ? " collision" : ""}`).join(" · ");
      row.append(name, detail); list.append(row);
    }
    updateObservationPreview();
  }

  function updateObservationPreview() {
    const source = state.inspected || [0, 0], action = Number($("observation-action").value), delta = directions[action].delta;
    const collision = $("observation-outcome").value === "collision";
    const destination = collision ? [...source] : [source[0] + delta[0], source[1] + delta[1]];
    setText("observation-preview", `${positionText(source)} → ${directions[action].short} → ${positionText(destination)}${collision ? " · stay" : ""}`);
    const repeats = Number($("observation-count").value);
    setText("observation-count-label", `${repeats} ${repeats === 1 ? "time" : "times"}`);
    return { source: [...source], action_index: action, destination, count: repeats };
  }

  function historyDescription(item) {
    const operation = item.operation && typeof item.operation === "object" ? item.operation : item;
    const action = operation.action || operation.kind || operation.type;
    if (action === "observe") return ["Supplied observation", `${positionText(operation.source)} · ${directions[operation.action_index]?.name || operation.action_index} → ${positionText(operation.destination)} · ${operation.count} event${operation.count === 1 ? "" : "s"}`];
    if (action === "replay") return ["Replay", `${operation.budget} requested backups per arm · actual backups appear in each schedule`];
    if (action === "goal") return ["Goal changed", `${positionText(operation.goal || [operation.x, operation.z])} · values cleared without backups`];
    if (action === "reset") return ["Experiment reset", "Returned to the frozen baseline"];
    return [action ? String(action).replaceAll("_", " ") : "Operation", JSON.stringify(operation)];
  }

  function renderHistory() {
    const list = $("operation-history"); list.replaceChildren();
    const history = state.snapshot.history || [];
    if (!history.length) {
      const item = document.createElement("li"); item.className = "muted"; item.textContent = "Frozen baseline ready. No new operations yet."; list.append(item); return;
    }
    history.forEach((operation, index) => {
      const item = document.createElement("li"), numberElement = document.createElement("span"), copy = document.createElement("div"), title = document.createElement("strong"), detail = document.createElement("span");
      numberElement.className = "history-index"; numberElement.textContent = String(index + 1).padStart(2, "0"); copy.className = "history-copy";
      try { const description = historyDescription(operation); title.textContent = description[0]; detail.textContent = description[1]; }
      catch { title.textContent = "Recorded operation"; detail.textContent = JSON.stringify(operation); }
      copy.append(title, detail); item.append(numberElement, copy); list.append(item);
    });
  }

  for (const model of ["cumulative", "recent"]) $("model-" + model).addEventListener("click", () => { state.model = model; render(); });
  for (const schedule of schedules) $("schedule-" + schedule).addEventListener("click", () => { state.schedule = schedule; render(); });
  $("inspect-btn").addEventListener("click", withInput(() => inspect([integerInput("inspect-x", 0, 30), integerInput("inspect-z", 0, 30)])));
  for (const id of ["inspect-x", "inspect-z"]) $(id).addEventListener("keydown", event => { if (event.key === "Enter") $("inspect-btn").click(); });
  directions.forEach(direction => $("inspect-" + direction.name.toLowerCase()).addEventListener("click", () => { const current = state.inspected || [0, 0]; inspect([current[0] + direction.delta[0], current[1] + direction.delta[1]]); }));
  $("transition-graph").addEventListener("keydown", event => {
    const direction = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3 }[event.key];
    if (direction !== undefined) { event.preventDefault(); const current = state.inspected || [0, 0], delta = directions[direction].delta; inspect([current[0] + delta[0], current[1] + delta[1]]); }
    else if (event.key === "Enter" && event.target === $("transition-graph")) { event.preventDefault(); $("goal-btn").click(); }
  });
  $("goal-btn").addEventListener("click", () => { if (state.inspected) command({ action: "goal", x: state.inspected[0], z: state.inspected[1] }, `Goal set to ${positionText(state.inspected)}. Values cleared with no backups.`); });
  $("observation-action").addEventListener("change", updateObservationPreview);
  $("observation-outcome").addEventListener("change", updateObservationPreview);
  $("observation-count").addEventListener("input", updateObservationPreview);
  $("observe-btn").addEventListener("click", withInput(() => command({ action: "observe", ...updateObservationPreview() }, "Supplied outcomes applied to both empirical models and all six arms. Retained values may now be stale.")));
  $("step-btn").addEventListener("click", () => command({ action: "replay", budget: 1 }, "One backup requested for each planner; disabled replay performs no backups."));
  $("replay-btn").addEventListener("click", withInput(() => command({ action: "replay", budget: integerInput("replay-budget", 0, 256) }, "Replay batch applied. Compare actual backups and empirical errors across all schedules.")));
  $("play-btn").addEventListener("click", () => { if (state.busy || !state.ready) return; state.playing = true; state.batchesRemaining = 8; clearError(); updateControls(); playBatch(); });
  $("pause-btn").addEventListener("click", () => stopPlay());
  $("fixture-btn").addEventListener("click", withInput(() => command({ action: "fixture", seed: integerInput("fixture-seed", 0, 2147483647), window: integerInput("window-select", 4, 16) }, "New seeded synthetic fixture installed. The experiment starts from its aggregate baseline.")));
  $("window-select").addEventListener("change", withInput(() => {
    const value = integerInput("window-select", 4, 16);
    announce(`Window ${value} selected for the next new fixture or live capture. The current experiment still uses W=${state.snapshot.window}.`);
  }));
  $("capture-btn").addEventListener("click", withInput(() => command({ action: "capture", window: integerInput("window-select", 4, 16) }, "Frozen live graph captured. The live session was copied; supplied events affect only this workbench.")));
  $("reset-btn").addEventListener("click", () => command({ action: "reset" }, "Experiment reset to its frozen baseline. Supplied observations and replay operations cleared."));
  $("dismiss-error").addEventListener("click", clearError);
  $("export-btn").addEventListener("click", async () => {
    clearError();
    try {
      const receipt = await request("/api/workbench/receipt");
      const blob = new Blob([JSON.stringify(receipt, null, 2) + "\n"], { type: "application/json" });
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = `neurograft-workbench-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      announce("Receipt exported as JSON. Import it to validate and replay the ordered experiment.");
    } catch (error) { fail(error); }
  });
  $("import-btn").addEventListener("click", () => { $("import-file").value = ""; $("import-file").click(); });
  $("import-file").addEventListener("change", async event => {
    const file = event.target.files[0]; if (!file) return;
    if (state.busy || state.playing) { fail(new Error("Pause replay and wait for the current command before importing a receipt.")); return; }
    if (file.size > 2 * 1024 * 1024) { fail(new Error("Receipt file exceeds the 2 MiB import limit.")); return; }
    try {
      const receipt = JSON.parse(await file.text());
      await command({ action: "import", receipt }, "Receipt verified and imported. Ordered operations reproduced the recorded final state.");
    } catch (error) { fail(error); }
  });
  window.addEventListener("pagehide", () => stopPlay(false));
  window.workbenchText = () => JSON.stringify({ ...state.snapshot, selected_model: state.model, selected_schedule: state.schedule, inspected: state.inspected, playing: state.playing, busy: state.busy, batches_remaining: state.batchesRemaining });
  updateControls();
  request("/api/workbench").then(snapshot => { acceptSnapshot(snapshot); announce("Frozen baseline ready. Inspect an action, supply an outcome, then compare replay schedules."); }).catch(error => { $("connection-state").className = "connection error"; setText("connection-state", "Connection failed"); fail(error); });
})();
