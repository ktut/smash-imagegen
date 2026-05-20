---
name: sdxl-generation-guide
description: When and how to use this SDXL server effectively — what it's good at, what it's bad at, what params actually matter, and what to use instead when it's the wrong tool.
---

# SDXL generation guide

What this server (SDXL + IP-Adapter + ControlNet on `gaming-pc:8000`) is
actually good at, where it falls down, and the param sweet spots
discovered through real game-asset work.

## TL;DR — what this server is for

**Use it for:** sprite generation (characters, NPCs, projectiles, small
items) where you have a reference image to feed IP-Adapter. The model
locks onto reference style and produces clean, repeatable sprite frames.

**Don't use it for:** illustrated 2D game backgrounds, anything
requiring readable text in the image, complex multi-element scenes with
specific spatial composition. Use Gemini or DALL-E for those — they
follow prompts much better and handle illustration styles without
fighting their training.

## API surface

```bash
POST /generate
{
  "prompt": "...",                  # main prompt
  "negative_prompt": "...",         # optional, very useful
  "reference_image": "<base64>",    # optional, enables IP-Adapter
  "reference_weight": 0.5,          # optional, see cheat sheet below
  "num_inference_steps": 30,        # 30 = fast, 50 = balanced, 80 = max quality
  "guidance_scale": 7.5,            # 7-9 typical, higher = follows prompt more
  "width": 1024,
  "height": 1024,
  "seed": 42                        # for reproducibility
}
→ { "image_base64": "...", "seed_used": N, "saved_path": "..." }
```

Health check: `GET /health` → `{"status":"ok","device":"cuda"}`

## IP-Adapter weight cheat sheet

| Weight    | Effect                                                              |
| --------- | ------------------------------------------------------------------- |
| 0.7+      | **Copies reference content.** Output ≈ reference image, prompt mostly ignored. Use only if you literally want a variation of the reference. |
| 0.5–0.6   | **Strong style + most content.** Good for sprite variations on the same character base. |
| 0.3–0.45  | **Pulls aesthetic, content from prompt.** Sweet spot for "draw X in the style of this reference." |
| 0.2–0.3   | **Subtle style nudge.** Reference barely visible; prompt drives almost everything. |
| 0 / unset | **No reference influence.** Pure prompt-driven gen, fully exposes SDXL's photoreal bias. |

**Empirical finding:** for "illustrated game backgrounds" specifically,
no IP-Adapter weight works well. The base model wants to produce photos.
You can't reliably style-transfer it into flat illustration without a
LoRA fine-tune on that style.

## Steps + guidance scale

- **30 steps, guidance 7.5** — fast iteration / sketching. ~10–15s/gen.
- **50 steps, guidance 8.0–8.5** — production sprite work. ~20–25s/gen.
- **80 steps, guidance 8.5–9.0** — final polish. Diminishing returns past 80.

Steps > 100 rarely helps; usually a sign the prompt is the problem.

## Prompt structure that works

Order matters — SDXL weighs early tokens heavier:

1. **Subject / setting first.** "Interior of a dimly-lit upscale steakhouse
   cocktail bar..." NOT "16-bit pixel art game background of..."
2. **Specific concrete details next.** Materials, lighting, props.
3. **Style anchor at the end** — and pick ONE. Stacking "pixel art +
   Ghibli + Castle Crashers + Streets of Rage + JRPG" makes the model
   pick the worst-fit average.

Negative prompts pull a lot of weight. Always include the failure modes
you're seeing in current outputs.

## Known failure modes

1. **Text rendering is broken.** SDXL physically cannot reliably spell
   short text like "LADIES NIGHT" on a sign. Don't try. Composite text
   overlays separately (SVG + PIL/sharp).
2. **Illustrated/flat styles drift to photoreal or to rough sketch.**
   The middle ground (clean detailed illustration) isn't reliably
   reachable from prompts alone. Needs a fine-tuned checkpoint
   (PixelWaveXL, etc).
3. **Complex composition is unreliable.** "Bar in upper half, floor in
   lower half" gets respected maybe 30% of the time. For reliable
   composition use ControlNet (depth or seg maps) or img2img with a
   layout reference.
4. **Style anchor conflicts cause weird drift.** "Side-scrolling neon
   sign in a tavern" → cyberpunk street scene because "side-scrolling
   + neon" triggers cyberpunk training data. Be careful about
   accidental theme collisions.
5. **The server appears to cache by (prompt + seed).** If you see
   identical file sizes across reruns with the same seeds, change a
   seed to force a fresh gen.

## What to use instead when SDXL is wrong

- **Illustrated game backgrounds** → Gemini (`gemini-2.5-flash-image`)
  or DALL-E 3. Both render illustrated styles natively. Gemini outputs
  1024² fixed — stitch two vertically for landscape aspects (see
  smash-client's `generate-level-background` skill).
- **Anything with required readable text** → render text as SVG and
  composite onto the gen.
- **Need precise composition** → ControlNet (this server supports it
  but most existing scripts don't use it; check `server/` source).
- **Specific art style with high consistency** → fine-tune a LoRA on
  that style and load it (LoRAs live in `loras/`).

## Reference files

- `server/` — API implementation, check here for actual supported params
- `config.yaml` — base model + checkpoint config
- `loras/` — fine-tuned LoRAs available for loading
- smash-client's `scripts/lib/image-gen.js` — Gemini wrapper for the
  cases where SDXL is wrong tool
- smash-client's `scripts/generate-tavern-bg-gemini-stitch.js` —
  worked example of "give up on SDXL, use Gemini stitching" pattern
