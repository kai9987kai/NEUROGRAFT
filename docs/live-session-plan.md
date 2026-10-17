# NEUROGRAFT v0.2 live-session implementation

User request: make it continuous and learning, save its state, and add chat and a 3D environment interface.

## Design

A localhost service owns one CPU-small live session. A recurrent graph circuit acts in a deterministic discrete navigation world; the browser renders that same world in 3D. Online imitation updates use shortest-path route guidance and bounded replay. This is supervised online adaptation, not a claim of autonomous reinforcement learning. Chat controls the session and stores/retrieves explicit facts; its active backend is labelled local command/memory rather than implying an uninstalled language model.

Full saves include model/optimizer, recurrent inference state, world, private PRNG state, replay, counters, configuration and chat memory. Atomic, checksummed saves support exact continuation in the same runtime. The source graphs stay immutable. Autoresume and autosave are enabled; pause, step, learning toggle, named save and restore are explicit controls.

## Boundaries and ownership

- `neurograft/environment.py`, `tests/test_environment.py`: deterministic world, no torch dependency.
- `neurograft/live.py`, `tests/test_live.py`: online trainer, persistent session, chat control/memory.
- `neurograft/web/index.html`, `app.js`, `style.css`: offline UI with local Three.js; root supplies vendor files.
- Root: `neurograft/server.py`, API tests, CLI/launcher, vendor acquisition, integration, browser verification, documentation, packaging.

## World interface

`NavigationWorld(seed=0,size=11)`; actions 0=north(z-1),1=east(x+1),2=south(z+1),3=west(x-1). Coordinates x,z are integer cells, origin northwest; y is vertical in the renderer.

`.observe()` returns a finite Python nested list `[4][4]`: worm agent position, female fly goal position, male fly local obstacles/context, human reward/clock; normalize features. `.expert_action()->int` uses shortest accessible route. `.step(action)->dict` returns `reward`, `goal_reached`, `episode_done`, `collision`; selecting a new target automatically on completion. `.set_goal(color=None,x=None,z=None)` allows blue/amber/rose beacon or free interior cell; no blocked/outside goals. `.snapshot()->dict`, `.to_dict()->dict`, `NavigationWorld.from_dict(data)`. Save PRNG state and all episode/counter fields; reject malformed state.

Snapshot fields: `size`, `agent:{x,z,direction}`, `goal:{x,z,color}`, `beacons:[{x,z,color}]`, `obstacles:[{x,z}]`, `trail:[{x,z}]` bounded64, `steps`, `episode`, `goals_reached`, `total_reward`, `last_reward`. Every generated beacon is reachable. Max episode length bounded; explicit reset retains lifetime counters. `seed`/world serialization version included.

## Live-session interface

`LiveConfig(seed=0,hidden_dim=8,learning_rate=0.003,batch_size=4,train_every=4,unroll=4,replay_capacity=128,teacher_probability=0.35,threads=2)`; validate bounded values. `LiveSession(bundle,config=None)` creates graph circuit output_dim=4 and world. RLock protects all public operations. Own PRNG / torch initialization, no dependency on global mutable random state during ticks.

`.running=True`, `.learning=True`; `.tick(steps=1,manual_action=None)` executes steps even while paused when explicitly invoked; background loop checks running. Per step infer using `.step` and detached external recurrent state; record expert-labelled observation window; bounded replay training every train_every steps if learning enabled. Selection is policy plus explicitly recorded route-guidance assistance/exploration. Graph parameter updates must be real and finite. Recurrent state resets at episode boundaries. `.snapshot()` returns UI-ready JSON, no tensors. `.chat(message)->dict` returns `reply`, `backend:'local-memory-and-control'`, optional `action` such as `save`; bounded history/facts. Support pause/resume, status/help, go to blue/amber/rose or `go to x,z`, remember `key = value` (also `remember key is value`), recall `what is key` / `recall key`. Unknown messages explain available commands honestly. Do not claim generative-model learning.

`.save(path)` atomically persists complete state with integrity hash and returns receipt; `LiveSession.load(path,bundle)` validates entire candidate before returning; mismatch/corruption fail without mutating live state. Checkpoint format primitives/tensors only, weights_only=True load, size cap. Model topology buffers must agree with declared source graph, not merely self-consistent stored hashes. Optimizer state finite/bounded, config/world/replay/facts validate. Exact continuation test must compare subsequent ticks, model weights, optimizer, counters, recurrent state, world and chat memory after restore.

Live snapshot exact top-level fields: `version`, `running`, `learning`, `steps`, `updates`, `episodes`, `goals_reached`, `last_loss`, `loss_history` (max64 numeric values), `reward_history` (max64), `policy_accuracy` (recent expert-match fraction, not held-out generalization), `last_action`, `last_action_source`, `teacher_probability`, `world`, `sources` (bundle.stats()['sources']), `routes` ([4][5], null last), `activity` (4 mean node-state magnitudes), `facts` mapping, `chat` list `{role,content}`, `backend`, `graph_hash`, `model_hash`, `last_saved` (path/time nullable), `evidence_scope` concise string. `episodes` equals world episode count. Routes indicate synthetic latent edges.

## HTTP and browser contract

CLI: `python -m neurograft live --bundle data/imported/mixed.bundle.json --state data/live/session.pt --port 8768 --interval 0.25 --autosave 15`. Default bundle uses local mixed graph if available, otherwise explicit fixture bundle; saved graph hash must match. Bind 127.0.0.1. Single-writer session lock; orderly Ctrl+C saves. No external service or heartbeat automation needed: the running program maintains the learning loop independently of the browser tab.

GET `/api/state` returns snapshot plus `checkpoints:[{name,bytes,modified}]` and `runtime:{autosave_seconds,interval_seconds,last_error}`. GET `/api/checkpoints` returns list. GET `/api/health` returns `{ok:true}`. Static UI whitelist only, including `/vendor/three.module.js`, `/vendor/three.core.js`, `/vendor/three.LICENSE.txt`.

POST `/api/control` JSON `{action:'pause'|'resume'|'learning'|'step'|'goal'|'move', enabled?:bool, steps?:int, color?:str,x?:int,z?:int,direction?:int}` returns snapshot. Step max32; manual moves allowed only paused (clear UI error otherwise). Goal changes reset inference state/observation history; reject unsafe values before mutation.

POST `/api/chat` `{message}` returns `{reply,backend,state:snapshot}`. `action:'save'` from chat asks server to save default session. POST `/api/save` `{name:'default'|safe_slot_name}` returns `{saved:{name,...},state:snapshot}`; POST `/api/load` `{name}` validates named checkpoint and replaces session atomically under app lock. Default slot is CLI state file; others live in sibling `checkpoints/`. Allow only 1-40 alnum/underscore/hyphen slot names; never accept arbitrary paths. Cap JSON body64KiB. Require application/json and same-origin loopback Host/Origin on mutations, no CORS. Return JSON `{error}` with appropriate 4xx for user mistakes. Autosave default session, named saves preserve earlier snapshots.

UI: clean responsive desktop workspace, large actual 3D navigation canvas, camera orbit/zoom/reset, beacon/floor click to set goal, paused arrow/WASD manual moves, fullscreen f. Top live/pause/learning controls, metrics, loss sparkline, source status, synthetic routing activity, right-hand chat/fact memory, named save/load and autosave status. No runtime CDN/font dependencies. Display supervised guidance and local chat backend plainly. Clear stale/disconnected state, disable overlapping writes, catch all API errors. Poll state at ~500ms. `window.render_game_to_text()` serializes current visible world and session counters; `window.advanceTime(ms)` requests bounded explicit steps only in test mode `?test=1`, pauses first to prevent double stepping. Normal runtime authoritative server ticks, never browser frame rate.

## Verification

Test first: deterministic world and exact state roundtrip; actual parameter updates; pause/learning-off immutability; exact optimizer+world+memory resume; corrupt save rejection; concurrent save/tick; HTTP validation and no path traversal; chat fact persistence; browser start/pause/step/goals/chat/save/load/learning-toggle and reload, 3D rendering and console errors. Run bundled web-game Playwright client and inspect screenshots, then full Python regression suite. Build wheel with static assets and prove source/package parity. Launch live service and open verified UI.
