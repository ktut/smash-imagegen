# CLAUDE.md

Instructions for working on this repo with Claude Code.

## What this repo is

A local image generation server for the smash game project. Runs on a Windows
PC with an RTX 4080 Super. Replaces nano banana / Gemini image generation calls
with a self-hosted SDXL + ControlNet + IP-Adapter pipeline.

This is a **separate process** from the main game repos (`smash-client`,
`smash-server`). The game does not depend on it at runtime — it's used at
content creation time to generate sprites, projectiles, and items.

## Architecture in one paragraph

A FastAPI server holds an SDXL pipeline in VRAM. Requests come in over HTTP
with a prompt and optionally a reference image (identity, via IP-Adapter)
and/or a pose image (structure, via ControlNet OpenPose). Output is a PNG
saved to `outputs/` with a JSON sidecar, plus base64 returned in the response.

## Priorities

1. **Quality over speed.** This pipeline generates content for a game; users
   will inspect every image. Speed of the generation loop is not a goal.
2. **Reproducibility.** Every generation records its seed and parameters.
   Re-running with the same seed must produce the same image.
3. **Reference-image fidelity.** The point of this whole project is to match
   nano-banana's "use this as a reference" behavior. IP-Adapter tuning is the
   most important knob.

## How to make changes

- **All configuration is in `config.yaml`.** Don't hardcode model names in
  Python. If you find yourself wanting to change a model, change it there.
- **`pipeline.py` is the hot path.** Be careful with VRAM. Always test changes
  against the actual hardware before committing — diffusers behavior differs
  between CPU and CUDA paths.
- **`schemas.py` is the public API.** A change here is a breaking change for
  any caller (the CLI, the game's build scripts, future tools). Treat it
  accordingly: additive changes are fine, renames need migration.
- **Outputs are precious.** Never delete files in `outputs/`. The user
  cherry-picks from generations; older ones may still be in use.

## Running locally (PC)

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` only reloads the server code, not the pipeline weights. The
pipeline is loaded once at startup (`lifespan` in `main.py`). To pick up
config.yaml changes you must restart the server.

## Network setup

The server runs on the Windows PC at `192.168.1.103:8000`. On the Mac,
`gaming-pc` is mapped to that IP in `/etc/hosts`, so all commands use
`http://gaming-pc:8000`.

## Testing changes

There's no test suite yet (this is a small project). The minimum smoke test
after any change (run from the Mac):

```bash
curl http://gaming-pc:8000/health
curl http://gaming-pc:8000/config
curl -s -X POST http://gaming-pc:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "pixel-art fighter, idle pose, side view, 2D sprite, plain background", "seed": 1}' \
  | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('/tmp/test.png','wb').write(base64.b64decode(d['image_base64'])); print(d['seed_used'])"
open /tmp/test.png
```

Then visually inspect the output.

## What NOT to do

- Don't add a web UI. The user explicitly does not want one.
- Don't add a database. Filesystem + JSON sidecars is the contract.
- Don't refactor `pipeline.py` for "elegance" — diffusers' API is the thing
  driving its shape. Match diffusers' patterns even when they're awkward.
- Don't switch the base model to Flux without an explicit ask. SDXL is the
  choice because IP-Adapter / ControlNet / LoRA ecosystems are mature there.
