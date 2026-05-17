"""Output file saving with metadata sidecar."""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from .schemas import GenerateRequest


def save_output(
    image: Image.Image,
    *,
    outputs_dir: Path,
    request: GenerateRequest,
    seed_used: int,
) -> Path:
    """Save the generated image alongside a .json metadata sidecar.

    Returns the image path.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{ts}-seed{seed_used}"
    img_path = outputs_dir / f"{base}.png"
    meta_path = outputs_dir / f"{base}.json"

    image.save(img_path, format="PNG")

    # Strip base64 blobs from metadata to keep it lightweight; record only
    # presence + a short hash for traceability.
    meta = request.model_dump()
    for key in ("reference_image", "pose_image"):
        val = meta.get(key)
        if val:
            meta[key] = f"<base64 omitted, len={len(val)}>"
    meta["seed_used"] = seed_used
    meta["timestamp"] = ts

    meta_path.write_text(json.dumps(meta, indent=2))
    return img_path
