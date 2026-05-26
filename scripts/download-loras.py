"""Download pixel-art and other SDXL LoRAs into loras/.

Run on the PC after `git pull`:
    .\.venv\Scripts\Activate.ps1
    python scripts/download-loras.py

LoRAs are gitignored (large binary files) so each machine downloads its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
LORAS_DIR = ROOT / "loras"
LORAS_DIR.mkdir(exist_ok=True)

# (repo_id, filename, local_name.safetensors)
LORAS = [
    # nerijs/pixel-art-xl — the most widely-used SDXL pixel-art LoRA.
    # Trigger word: "pixel" (include in prompt). Recommended weight 1.0-1.2.
    ("nerijs/pixel-art-xl", "pixel-art-xl.safetensors", "pixel-art-xl.safetensors"),
]


def main() -> None:
    for repo_id, filename, local_name in LORAS:
        dst = LORAS_DIR / local_name
        if dst.exists():
            print(f"[skip] {local_name} already present ({dst.stat().st_size // 1024} KB)")
            continue
        print(f"[download] {repo_id}/{filename} -> {dst}")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        # hf_hub_download symlinks/caches; copy bytes so loras/ is self-contained
        dst.write_bytes(Path(path).read_bytes())
        print(f"           done ({dst.stat().st_size // 1024} KB)")
    print("All LoRAs ready in:", LORAS_DIR)


if __name__ == "__main__":
    main()
