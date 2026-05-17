#!/usr/bin/env python3
"""CLI client for smash-imagegen.

Works from either machine — just point it at the server.

Examples:
    # From the PC itself
    python scripts/cli.py --prompt "pixel-art fighter, idle pose" \\
                         --output outputs/hero-idle.png

    # From the Mac, with a reference + pose
    python scripts/cli.py --server http://gaming-pc.local:8000 \\
                         --prompt "pixel-art fighter, mid-punch" \\
                         --reference assets/hero-ref.png \\
                         --pose assets/punch-pose.png \\
                         --reference-weight 0.8 \\
                         --pose-weight 0.9 \\
                         --seed 42 \\
                         --output outputs/hero-punch.png
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import requests


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=None)

    p.add_argument("--reference", type=Path, help="Identity reference image")
    p.add_argument("--reference-weight", type=float, default=0.7)

    p.add_argument("--pose", type=Path, help="Pose reference (skeleton or raw image)")
    p.add_argument(
        "--no-pose-extract",
        action="store_true",
        help="Treat --pose as a pre-made OpenPose skeleton (skip extraction)",
    )
    p.add_argument("--pose-weight", type=float, default=0.9)

    p.add_argument("--lora", default=None, help="LoRA name (without .safetensors)")
    p.add_argument("--lora-weight", type=float, default=0.8)

    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)

    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Local path to also save the image (server saves its own copy regardless)")
    args = p.parse_args()

    payload: dict = {
        "prompt": args.prompt,
        "reference_weight": args.reference_weight,
        "pose_extract": not args.no_pose_extract,
        "pose_weight": args.pose_weight,
        "lora_weight": args.lora_weight,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance,
        "width": args.width,
        "height": args.height,
    }
    if args.negative_prompt:
        payload["negative_prompt"] = args.negative_prompt
    if args.reference:
        payload["reference_image"] = b64(args.reference)
    if args.pose:
        payload["pose_image"] = b64(args.pose)
    if args.lora:
        payload["lora_name"] = args.lora

    url = f"{args.server.rstrip('/')}/generate"
    print(f"POST {url} (prompt={args.prompt!r})", file=sys.stderr)
    resp = requests.post(url, json=payload, timeout=600)

    if not resp.ok:
        print(f"ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1

    data = resp.json()
    print(f"seed_used={data['seed_used']}", file=sys.stderr)
    print(f"server saved: {data['saved_path']}", file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(base64.b64decode(data["image_base64"]))
        print(f"local saved: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
