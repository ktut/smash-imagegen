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

## Generating animations or multi-view sets — two-stage rule

Any time you generate **multiple frames of the same subject** — an
animation, a rotation, a pose set, a sprite sheet — you MUST use a
two-stage workflow:

1. **Stage 1: produce one canonical reference image** of the subject
   (the gem at its default orientation, the character standing, etc.).
   Get this single image looking right before generating any frames.
2. **Stage 2: generate every frame using that canonical image as the
   IP-Adapter `reference_image`**. This locks the subject's identity
   across frames so it looks like the same gem rotating, not eight
   unrelated gems.

If you skip stage 1 and just generate N frames with N different seeds,
each frame will be a separately-imagined subject — the user gets eight
different gems instead of one gem rotating. Do not ship that result.

For 3D rotation specifically: use a moderate `reference_weight`
(~0.6-0.7, not 1.0) so the model can interpret "rotated 90 degrees"
while keeping identity. Higher weights lock the view angle too tightly
and the rotation prompt is ignored.

This mirrors the Gemini two-stage flow documented in the smash-client
CLAUDE.md (real photo → stand sprite → pose sprites). The same pattern
applies to SDXL via the harness: call the preset once with no reference
to get a canonical, then call N more times with the canonical passed as
`--reference`.

## Delivering results to the user — required output format

**Whenever the user asks for multiple items or an animation, the response
MUST include all three of:**

1. The composite artefact (animated WebP for animations, contact sheet for
   batches) hosted at a temporary public URL.
2. A labelled contact sheet — every frame in a grid, each annotated with
   its frame number, varied parameter (angle / pose / seed / etc.), and
   anything else relevant to giving feedback.
3. Per-frame hosted URLs in a labelled list, so the user can reference a
   single frame by name when giving feedback (e.g. "frame 3 has a green
   bleed at the bottom").

Use catbox.moe (`https://catbox.moe/user/api.php`) as the host — it's
been verified working and gives stable direct-image URLs. Do NOT just
paste image content into the response or save only locally; the user
is frequently on a remote control session and cannot see local files.

This rule applies even for ad-hoc test generations and one-off
experiments — anywhere multiple frames would benefit from per-frame
feedback. Single one-off images don't need the contact-sheet treatment.

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
