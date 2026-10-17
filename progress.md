Original prompt: make it continuous and learning, save its state, and add chat and a 3D environment interface.

Implemented and verified NEUROGRAFT v0.2 against docs/live-session-plan.md while preserving all v0.1 reports and acquired source graphs:
- `neurograft/environment.py`: deterministic 9x9 multi-beacon navigation world with shortest-path teacher, per-source sensory observations, and JSON state serialization.
- `neurograft/live.py`: `LiveSession` wrapping `GraphCircuit` + 5-way navigation head, online AdamW (`lr=3e-3`) supervised route imitation, persistent recurrent states, local command & fact memory chat (`local-memory-and-control`), and atomic `weights_only=True` checkpointing with bundle SHA-256, ZIP CRC32/header/padding, and tensor finiteness verification.
- `neurograft/service.py` & `neurograft/server.py`: thread-safe background loop, periodic autosave, named checkpoints (`data/live/checkpoints/<name>.pt`), HTTP JSON API, and static asset serving.
- `neurograft/web/`: interactive Three.js 3D navigation arena, cross-source route matrix, source activity gauges, loss/accuracy sparkline, checkpoint & goal controls, chat drawer, and Playwright hooks (`window.render_game_to_text`, `window.advanceTime`).
- `neurograft/cli.py` & `Run-NEUROGRAFT.ps1`: added `python -m neurograft live` and `-Mode Live` launcher.
- Verified with `pytest` and headless Chromium Playwright (`verify_ui.mjs` and `web_game_playwright_client.js`) with zero console/page errors.

