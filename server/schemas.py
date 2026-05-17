"""Pydantic schemas for HTTP API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """A single generation request.

    Three orthogonal controls:
      * Prompt — text describing what you want
      * Reference image (IP-Adapter) — what should the character look like
      * Pose image (ControlNet OpenPose) — what pose should they be in

    Any subset can be omitted.
    """

    prompt: str
    negative_prompt: str = (
        "blurry, low quality, extra limbs, deformed, watermark, text, signature"
    )

    # ---- Identity reference (IP-Adapter) ----
    reference_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded reference image for character identity",
    )
    reference_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.5,
        description="How strongly the reference influences the output (0=ignore, ~0.7=balanced, 1.0+=strong)",
    )

    # ---- Pose control (ControlNet OpenPose) ----
    pose_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded image. Either a pose skeleton, or any image to extract pose from (see pose_extract)",
    )
    pose_extract: bool = Field(
        default=True,
        description="If True, run OpenPose on pose_image first. If False, treat pose_image as a pre-made skeleton",
    )
    pose_weight: float = Field(default=0.9, ge=0.0, le=2.0)

    # ---- LoRA ----
    lora_name: Optional[str] = Field(
        default=None,
        description="Filename (without .safetensors) of a LoRA in the loras/ directory",
    )
    lora_weight: float = Field(default=0.8, ge=0.0, le=2.0)

    # ---- Generation params ----
    seed: int = Field(default=-1, description="-1 for random")
    steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)


class GenerateResponse(BaseModel):
    image_base64: str
    seed_used: int
    saved_path: str


class HealthResponse(BaseModel):
    status: str
    device: Optional[str] = None


class ConfigResponse(BaseModel):
    base_model: str
    controlnet_model: str
    vae_model: str
    ip_adapter_repo: str
    ip_adapter_weight_name: str
    available_loras: list[str]
    outputs_dir: str
