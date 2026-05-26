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

    # ---- Canny edge control (second ControlNet, optional) ----
    # When supplied, this lets the caller lock the silhouette/edges of a
    # reference image (e.g. an existing pixel-art sprite) on top of the
    # OpenPose skeleton. The two control signals are combined via a
    # MultiControlNetModel inside the pipeline.
    canny_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded image to use as a Canny edge reference, or a pre-made edge map.",
    )
    canny_extract: bool = Field(
        default=True,
        description="If True, run a Canny edge detector on canny_image first.",
    )
    canny_weight: float = Field(default=0.5, ge=0.0, le=2.0)

    # ---- LoRA ----
    lora_name: Optional[str] = Field(
        default=None,
        description="Filename (without .safetensors) of a LoRA in the loras/ directory",
    )
    lora_weight: float = Field(default=0.8, ge=0.0, le=2.0)

    # ---- Post-process ----
    force_background_color: Optional[str] = Field(
        default=None,
        description=(
            "If set (e.g. '#00FF00'), the server runs a BFS flood-fill from the "
            "image edges and replaces every pixel connected to the corners (i.e. "
            "the background region) with this exact color. Guarantees a uniform, "
            "chroma-key-friendly background regardless of model bias."
        ),
    )
    background_tolerance: int = Field(
        default=60,
        ge=0,
        le=255,
        description="Per-channel tolerance for the flood-fill match against the sampled corner colour.",
    )
    outline_color: Optional[str] = Field(
        default=None,
        description=(
            "If set (e.g. '#000000'), the server strokes a solid border in this "
            "colour around the character silhouette after the bg fill. The border "
            "is `outline_width` pixels of character that touch the bg region. "
            "Requires force_background_color to be set as well."
        ),
    )
    outline_width: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Thickness (in pixels) of the silhouette outline stroke.",
    )

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
