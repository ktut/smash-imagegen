"""FastAPI server for SDXL image generation with IP-Adapter + ControlNet + LoRA."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .config import load_config
from .pipeline import ImageGenPipeline
from .schemas import (
    ConfigResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
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
    state["config"] = config
    state["pipeline"] = pipeline
    log.info("Pipeline ready. Server listening.")
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
