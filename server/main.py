"""FastAPI server for SDXL image generation with IP-Adapter + ControlNet + LoRA."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .config import load_config
from .pipeline import ImageGenPipeline
from .presets import PresetRegistry
from .schemas import (
    ConfigResponse,
    GenerateFromPresetRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    PresetInfo,
    PresetListResponse,
    PresetVarInfo,
)
from .utils import save_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smash-imagegen")

# Loaded at startup, used by all endpoints
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline once at startup. Keep it warm in VRAM."""
    config = load_config(Path("config.yaml"))
    log.info("Loading pipeline (this takes ~30s)...")
    pipeline = ImageGenPipeline(config)
    presets = PresetRegistry(Path("presets"))
    state["config"] = config
    state["pipeline"] = pipeline
    state["presets"] = presets
    log.info(
        "Pipeline ready. %d preset(s) loaded. Server listening.",
        len(presets.list()),
    )
    yield
    log.info("Shutting down.")


app = FastAPI(title="smash-imagegen", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if "pipeline" in state else "loading",
        device=state["pipeline"].device if "pipeline" in state else None,
    )


@app.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    cfg = state["config"]
    return ConfigResponse(
        base_model=cfg.base_model,
        controlnet_model=cfg.controlnet_openpose_model,
        vae_model=cfg.vae_model,
        ip_adapter_repo=cfg.ip_adapter_repo,
        ip_adapter_weight_name=cfg.ip_adapter_weight_name,
        available_loras=list_loras(cfg.loras_dir),
        outputs_dir=str(cfg.outputs_dir),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        result = pipeline.generate(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    saved_path = save_output(
        result.image,
        outputs_dir=state["config"].outputs_dir,
        request=req,
        seed_used=result.seed_used,
    )
    log.info("Saved %s (seed=%s)", saved_path, result.seed_used)

    return GenerateResponse(
        image_base64=result.image_base64,
        seed_used=result.seed_used,
        saved_path=str(saved_path),
    )


def list_loras(loras_dir: Path) -> list[str]:
    if not loras_dir.exists():
        return []
    return sorted(p.stem for p in loras_dir.glob("*.safetensors"))


# ---------------------------------------------------------------------------
# Preset endpoints
# ---------------------------------------------------------------------------


def _preset_to_info(preset) -> PresetInfo:
    return PresetInfo(
        name=preset.name,
        description=preset.description,
        vars=[
            PresetVarInfo(
                name=v.name,
                description=v.description,
                required=v.required,
                default=v.default,
            )
            for v in preset.vars
        ],
        requires_loras=preset.requires_loras,
        defaults=preset.defaults,
        prompt_template=preset.prompt_template,
        negative_prompt=preset.negative_prompt,
    )


@app.get("/presets", response_model=PresetListResponse)
def list_presets() -> PresetListResponse:
    presets = state["presets"].list()
    return PresetListResponse(presets=[_preset_to_info(p) for p in presets])


@app.get("/presets/{name}", response_model=PresetInfo)
def get_preset(name: str) -> PresetInfo:
    try:
        return _preset_to_info(state["presets"].get(name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown preset: {name}")


@app.post("/generate-from-preset", response_model=GenerateResponse)
def generate_from_preset(req: GenerateFromPresetRequest) -> GenerateResponse:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        preset = state["presets"].get(req.preset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown preset: {req.preset}")

    try:
        body = preset.resolve(vars=req.vars, overrides=req.overrides)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Images and per-call seed always come from the request envelope, not
    # the preset itself.
    if req.reference_image is not None:
        body["reference_image"] = req.reference_image
    if req.pose_image is not None:
        body["pose_image"] = req.pose_image
    if req.canny_image is not None:
        body["canny_image"] = req.canny_image
    if req.seed != -1 or "seed" not in body:
        body["seed"] = req.seed

    try:
        generate_req = GenerateRequest(**body)
    except Exception as e:
        log.exception("Preset %s resolved to invalid GenerateRequest", req.preset)
        raise HTTPException(status_code=400, detail=f"invalid resolved request: {e}")

    return generate(generate_req)
