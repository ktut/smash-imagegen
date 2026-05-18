# smash-imagegen

Local image generation server for the `smash` game project. Runs on a Windows PC
with an NVIDIA GPU and exposes a simple HTTP API for character sprites,
projectiles, and items.

The intent is to replace per-image API costs (nano banana / Gemini) with
unlimited local generation, while keeping reference-image and pose-control
parity through **IP-Adapter** and **ControlNet**.

## Architecture

```
  Mac (or PC) ──HTTP──►  Windows PC (RTX 4080 Super)
                         ├── FastAPI server (this repo)
                         ├── diffusers pipeline
                         │     ├── SDXL base
                         │     ├── ControlNet OpenPose (pose)
                         │     └── IP-Adapter Plus (identity)
                         └── outputs/  (PNG + JSON sidecar per image)
```

Single process, single GPU, single model loaded once at startup and held in VRAM.
All configuration is in `config.yaml` — there is no UI.

---

## One-time PC setup

Run this entire block in **PowerShell as Administrator** from the `smash-imagegen`
directory. It is safe to re-run — every step is idempotent.

```powershell
# ── 1. SSH Server ─────────────────────────────────────────────────────────────
# Lets the Mac run commands and edit files on this PC without copy-pasting.

Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

# Allow SSH through the firewall
New-NetFirewallRule -DisplayName "OpenSSH Server" -Direction Inbound `
    -Protocol TCP -LocalPort 22 -Action Allow -ErrorAction SilentlyContinue

Write-Host "SSH server running. Connect from Mac with: ssh $env:USERNAME@gaming-pc"


# ── 2. Firewall rule for imagegen ─────────────────────────────────────────────
New-NetFirewallRule -DisplayName "smash-imagegen" -Direction Inbound `
    -Protocol TCP -LocalPort 8000 -Action Allow -ErrorAction SilentlyContinue


# ── 3. Prevent sleep while server is running ──────────────────────────────────
# Patches server/main.py to hold a Windows power request (same API video
# players use). The PC won't sleep while the server process is alive.
# Releasing the lock on shutdown is handled automatically via atexit.

python -c "
import pathlib
p = pathlib.Path('server/main.py')
src = p.read_text()
snippet = '''import ctypes, atexit
# Prevent system and display sleep while the server is running
_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002
ctypes.windll.kernel32.SetThreadExecutionState(
    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
)
atexit.register(ctypes.windll.kernel32.SetThreadExecutionState, _ES_CONTINUOUS)
'''
if 'SetThreadExecutionState' not in src:
    p.write_text(snippet + src)
    print('server/main.py patched — sleep prevention added.')
else:
    print('server/main.py already patched, skipping.')
"


# ── 4. Kill any stale server process on port 8000 ─────────────────────────────
$conn = netstat -ano | Select-String ':8000\s.*LISTENING'
if ($conn) {
    $stalePid = ($conn -split '\s+')[-1]
    Stop-Process -Id $stalePid -Force
    Write-Host "Killed stale process PID $stalePid on port 8000."
}


# ── 5. Start the server ───────────────────────────────────────────────────────
.\.venv\Scripts\Activate.ps1
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Once you see `Pipeline ready. Server listening.` the setup is complete.

---

## One-time Mac setup

```bash
# 1. Add the PC to /etc/hosts so scripts can use the hostname "gaming-pc"
#    (skip if already present)
echo "$(ping -c1 gaming-pc 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1) gaming-pc" \
  | sudo tee -a /etc/hosts

# 2. Copy your SSH key so you never need a password
ssh-copy-id $USER@gaming-pc

# 3. Verify
ssh gaming-pc "echo connected as \$env:USERNAME"
```

From here you can run anything on the PC directly:

```bash
# Single command
ssh gaming-pc "cd smash-imagegen && netstat -ano | findstr :8000"

# Interactive shell
ssh gaming-pc
```

---

## Daily workflow

The server **does not auto-start** after a reboot. To start it from the Mac
without touching the PC:

```bash
ssh gaming-pc "cd smash-imagegen && .\.venv\Scripts\Activate.ps1 && python -m uvicorn server.main:app --host 0.0.0.0 --port 8000"
```

Or add a helper alias to your Mac's `~/.zshrc`:

```bash
alias imagegen-start='ssh gaming-pc "cd smash-imagegen && .\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000"'
alias imagegen-restart='ssh gaming-pc "Stop-Process -Id \$(netstat -ano | Select-String \":8000.*LISTENING\" | ForEach-Object { (\$_ -split \"\s+\")[-1] }) -Force -ErrorAction SilentlyContinue; cd smash-imagegen; .\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000"'
```

---

## Prerequisites (PC side)

- Windows 10/11
- NVIDIA GPU with CUDA 12.x driver (RTX 30-series or newer recommended; 16GB+ VRAM)
- Python 3.11
- ~30GB free disk for model cache
- Hugging Face account (free) — needed to accept the SDXL license once

## Setup

```powershell
git clone https://github.com/<you>/smash-imagegen.git
cd smash-imagegen

# Accept the SDXL license once. Opens a browser.
# Visit https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 and click "Agree".
# Then in PowerShell:
pip install huggingface_hub
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens

# Install everything else
.\scripts\setup-windows.ps1
```

---

## Usage

### From the CLI (Mac or PC)

```bash
# Pure text-to-image
python scripts/cli.py --server http://gaming-pc:8000 \
    --prompt "pixel-art fighter, idle pose, side view, 2D sprite"

# With an identity reference (the nano-banana analog)
python scripts/cli.py --server http://gaming-pc:8000 \
    --prompt "same fighter, mid-air kick, side view" \
    --reference assets/hero-ref.png \
    --reference-weight 0.7

# With a pose target — extracts OpenPose from any image
python scripts/cli.py --server http://gaming-pc:8000 \
    --prompt "fighter, side view, 2D sprite" \
    --pose assets/kick-pose-source.jpg \
    --pose-weight 0.9

# Full combo: identity reference + pose + seed for reproducibility
python scripts/cli.py --server http://gaming-pc:8000 \
    --prompt "pixel-art fighter, mid-punch, side view, 2D sprite" \
    --reference assets/hero-ref.png --reference-weight 0.7 \
    --pose assets/punch-pose.png --pose-weight 0.9 \
    --seed 42 \
    --output ../smash-client/public/sprites/hero-punch.png
```

### From curl

```bash
REF=$(base64 -i assets/hero-ref.png)

curl -X POST http://gaming-pc:8000/generate \
    -H "Content-Type: application/json" \
    -d "{
      \"prompt\": \"pixel-art fighter, idle pose, side view\",
      \"reference_image\": \"$REF\",
      \"reference_weight\": 0.7,
      \"seed\": 42
    }"
```

### From Node (game build scripts)

```javascript
import fs from 'node:fs/promises';

const ref = await fs.readFile('assets/hero-ref.png', { encoding: 'base64' });

const res = await fetch('http://gaming-pc:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'pixel-art fighter, idle pose, side view, 2D sprite',
    reference_image: ref,
    reference_weight: 0.7,
    seed: 42,
  }),
});
const { image_base64, seed_used, saved_path } = await res.json();
await fs.writeFile('out.png', Buffer.from(image_base64, 'base64'));
```

---

## Tuning reference fidelity

| Knob | Range | Effect |
|---|---|---|
| `reference_weight` | 0.0–1.5 | How strongly the reference image controls identity. 0.5 = loose, 0.7 = balanced (default), 1.0+ = strict (may overpower prompt) |
| `pose_weight` | 0.0–2.0 | How strictly the output matches the pose skeleton. 0.9 default. Above 1.2 starts to look stiff |
| `guidance_scale` | 0.0–20.0 | Prompt adherence. 7.5 default. Higher = follows prompt harder but can look fried |
| `steps` | 1–150 | Quality vs speed. 30 default. Diminishing returns past 50 |
| `seed` | int or -1 | Fixed seed = reproducible. Use the same seed across an animation cycle for frame consistency |

---

## Troubleshooting

**CUDA error after PC wakes from sleep.** The GPU context is broken. Restart the
server: `Ctrl-C`, then re-run `python -m uvicorn ...`. The sleep prevention patch
(step 3 of one-time setup) stops this happening going forward.

**Black images out of the pipeline.** You're using the wrong VAE — the default
SDXL VAE is broken in fp16. Make sure `config.yaml` has
`madebyollin/sdxl-vae-fp16-fix`.

**CUDA out of memory.** `enable_model_cpu_offload()` is already on. Drop
resolution to 768×768, or comment out IP-Adapter loading temporarily.

**`torch.cuda.is_available()` returns False.** The CPU-only PyTorch was
installed. Reinstall with the CUDA index URL in `setup-windows.ps1`.

**Pose extraction looks wrong.** OpenPose works best on full-body human-like
references. For non-human characters, set `pose_extract: false` and provide a
hand-drawn skeleton directly.

**Server unreachable from Mac.** Check (a) firewall rule, (b) both machines on
same network/VLAN, (c) the PC isn't on a "Public" network profile in Windows
(switch to Private).

**SSH connection refused.** Run `Get-Service sshd` on the PC — if it's stopped,
run `Start-Service sshd`. Check the firewall rule for port 22 exists.

---

## LoRA training (later)

For peak character consistency, train a LoRA on 15–30 generations of your
character, then use it in subsequent requests:

```bash
python scripts/cli.py --server http://gaming-pc:8000 \
    --prompt "hero, jumping side kick" \
    --lora hero-v1 \
    --lora-weight 0.8 \
    --pose assets/kick.png
```

LoRA training script isn't included yet — diffusers ships with one, see
`examples/dreambooth/train_dreambooth_lora_sdxl.py` in the diffusers repo.

---

## Layout

```
smash-imagegen/
├── server/
│   ├── main.py        # FastAPI endpoints
│   ├── pipeline.py    # diffusers pipeline + inference
│   ├── schemas.py     # request/response models
│   ├── config.py      # config loader
│   └── utils.py       # image i/o helpers
├── scripts/
│   ├── cli.py         # client CLI (works from any machine)
│   └── setup-windows.ps1
├── loras/             # drop .safetensors files here
├── outputs/           # PNGs + JSON sidecars saved per generation
├── assets/            # reference images, pose templates
├── config.yaml
└── requirements.txt
```
