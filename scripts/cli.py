#!/usr/bin/env python3
"""smash-imagegen CLI — works from any machine, just point at the server.

Three subcommands:

    list                                — show available presets
    show  <preset>                      — print full preset definition
    run   <preset> [opts]               — generate using a preset (recommended)
    raw                                 — direct /generate call (old behavior,
                                          for when you need fine-grained control)

Typical usage (preset-based, recommended):

    cli.py run pixel-art-character \\
        --reference  photo.jpg \\
        --pose       walk-pose.png \\
        --canny      walk-pose.png \\
        --var character_description="bald man with leather jacket, beard" \\
        --candidates 4 \\
        --out        ./out/

The server is at http://gaming-pc:8000 by default; override with --server.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


DEFAULT_SERVER = "http://gaming-pc:8000"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args) -> int:
    resp = requests.get(f"{args.server.rstrip('/')}/presets", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data["presets"]:
        print("(no presets — add a YAML file under presets/ on the server)")
        return 0
    for p in data["presets"]:
        desc = (p["description"] or "").splitlines()[0] if p["description"] else ""
        print(f"  {p['name']:<28} {desc}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------

def cmd_show(args) -> int:
    resp = requests.get(f"{args.server.rstrip('/')}/presets/{args.preset}", timeout=10)
    if resp.status_code == 404:
        print(f"unknown preset: {args.preset}", file=sys.stderr)
        return 1
    resp.raise_for_status()
    p = resp.json()

    print(f"# {p['name']}")
    if p["description"]:
        print(p["description"])
    print()
    print("Required vars:")
    if not p["vars"]:
        print("  (none)")
    for v in p["vars"]:
        marker = "*" if v["required"] and v["default"] is None else " "
        line = f"  {marker} {v['name']}"
        if v["default"] is not None:
            line += f" (default: {v['default']!r})"
        print(line)
        for ln in (v["description"] or "").splitlines():
            print(f"      {ln}")
    print()
    print("Requires LoRAs:", ", ".join(p["requires_loras"]) or "(none)")
    print()
    print("Prompt template:")
    for ln in p["prompt_template"].splitlines():
        print(f"  {ln}")
    print()
    print("Defaults:")
    for k, v in p["defaults"].items():
        print(f"  {k}: {v!r}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def _parse_var_assignments(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            print(f"--var expects key=value, got: {item!r}", file=sys.stderr)
            sys.exit(2)
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


def _parse_override_assignments(items: list[str]) -> dict:
    """--override key=value with simple type coercion (int/float/bool/str/null)."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            print(f"--override expects key=value, got: {item!r}", file=sys.stderr)
            sys.exit(2)
        k, raw = item.split("=", 1)
        k = k.strip()
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            out[k] = lowered == "true"
        elif lowered in ("null", "none"):
            out[k] = None
        else:
            try:
                out[k] = int(raw)
            except ValueError:
                try:
                    out[k] = float(raw)
                except ValueError:
                    out[k] = raw
    return out


def cmd_run(args) -> int:
    payload: dict = {
        "preset": args.preset,
        "vars": _parse_var_assignments(args.var),
        "overrides": _parse_override_assignments(args.override),
    }
    if args.reference:
        payload["reference_image"] = b64(args.reference)
    if args.pose:
        payload["pose_image"] = b64(args.pose)
    if args.canny:
        payload["canny_image"] = b64(args.canny)

    out_dir: Path | None = args.out
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{args.server.rstrip('/')}/generate-from-preset"
    base_seed = args.seed if args.seed is not None else -1
    print(f"POST {url}  preset={args.preset}  candidates={args.candidates}", file=sys.stderr)

    failures = 0
    for i in range(args.candidates):
        seed = base_seed if base_seed == -1 else (base_seed + i * 1009)
        payload["seed"] = seed
        resp = requests.post(url, json=payload, timeout=600)
        if not resp.ok:
            failures += 1
            print(f"  cand {i+1}/{args.candidates}: ERROR {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            continue
        data = resp.json()
        line = f"  cand {i+1}/{args.candidates}: seed={data['seed_used']}  server={data['saved_path']}"
        if out_dir:
            local = out_dir / f"{args.preset}-{data['seed_used']}.png"
            local.write_bytes(base64.b64decode(data["image_base64"]))
            line += f"  local={local}"
        print(line, file=sys.stderr)

    return 1 if failures == args.candidates else 0


# ---------------------------------------------------------------------------
# Subcommand: raw (old behavior)
# ---------------------------------------------------------------------------

def cmd_raw(args) -> int:
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
    if args.canny:
        payload["canny_image"] = b64(args.canny)
        payload["canny_weight"] = args.canny_weight
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


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli.py")
    p.add_argument("--server", default=DEFAULT_SERVER, help=f"Server URL (default: {DEFAULT_SERVER})")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List presets available on the server")

    s = sub.add_parser("show", help="Print full definition of a preset")
    s.add_argument("preset")

    r = sub.add_parser("run", help="Generate one or more images from a preset")
    r.add_argument("preset")
    r.add_argument("--reference", type=Path, help="Identity reference image (photo)")
    r.add_argument("--pose", type=Path, help="Pose image (OpenPose source)")
    r.add_argument("--canny", type=Path, help="Canny edge source (often same as --pose)")
    r.add_argument("--var", action="append", help="Template variable: --var key=value (repeatable)")
    r.add_argument("--override", action="append",
                   help="Override a GenerateRequest field: --override steps=200 (repeatable)")
    r.add_argument("--candidates", type=int, default=1, help="Number of seeds to generate (default 1)")
    r.add_argument("--seed", type=int, default=None, help="Base seed (random if omitted)")
    r.add_argument("--out", type=Path, help="Local directory to save images to (server saves regardless)")

    rw = sub.add_parser("raw", help="Direct /generate call with explicit flags (no preset)")
    rw.add_argument("--prompt", required=True)
    rw.add_argument("--negative-prompt", default=None)
    rw.add_argument("--reference", type=Path)
    rw.add_argument("--reference-weight", type=float, default=0.7)
    rw.add_argument("--pose", type=Path)
    rw.add_argument("--no-pose-extract", action="store_true")
    rw.add_argument("--pose-weight", type=float, default=0.9)
    rw.add_argument("--canny", type=Path)
    rw.add_argument("--canny-weight", type=float, default=0.5)
    rw.add_argument("--lora", default=None)
    rw.add_argument("--lora-weight", type=float, default=0.8)
    rw.add_argument("--seed", type=int, default=-1)
    rw.add_argument("--steps", type=int, default=30)
    rw.add_argument("--guidance", type=float, default=7.5)
    rw.add_argument("--width", type=int, default=1024)
    rw.add_argument("--height", type=int, default=1024)
    rw.add_argument("--output", "-o", type=Path, default=None)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "list": cmd_list,
        "show": cmd_show,
        "run":  cmd_run,
        "raw":  cmd_raw,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
