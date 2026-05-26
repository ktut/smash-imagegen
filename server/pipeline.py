"""Diffusers pipeline: SDXL + ControlNet OpenPose + IP-Adapter + optional LoRA."""

from __future__ import annotations

import base64
import io
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .config import Config
from .schemas import GenerateRequest

log = logging.getLogger("smash-imagegen.pipeline")


@dataclass
class GenerationResult:
    image: Image.Image
    image_base64: str
    seed_used: int


def _decode_b64(data: str) -> Image.Image:
    """Decode a base64 data URL or raw base64 string into a PIL image."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ImageGenPipeline:
    def __init__(self, config: Config):
        # Lazy heavy imports so module-level errors are easier to read
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
        )

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            log.warning("CUDA not available — falling back to CPU. This will be very slow.")

        log.info("Loading VAE: %s", config.vae_model)
        vae = AutoencoderKL.from_pretrained(config.vae_model, torch_dtype=torch.float16)

        log.info("Loading ControlNet (OpenPose): %s", config.controlnet_openpose_model)
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_openpose_model, torch_dtype=torch.float16
        )

        log.info("Loading base model: %s", config.base_model)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            config.base_model,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )

        log.info(
            "Loading IP-Adapter: %s / %s",
            config.ip_adapter_repo,
            config.ip_adapter_weight_name,
        )
        self.pipe.load_ip_adapter(
            config.ip_adapter_repo,
            subfolder=config.ip_adapter_subfolder,
            weight_name=config.ip_adapter_weight_name,
        )

        # Memory optimizations — important for 16GB VRAM
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            log.info("xformers enabled")
        except Exception:
            log.info("xformers not available, using default attention")

        # Pose detector loaded lazily on first pose-extraction request
        self._pose_detector = None

        # Track currently-loaded LoRA so we don't reload unnecessarily
        self._current_lora: Optional[tuple[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, req: GenerateRequest) -> GenerationResult:
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # ---- LoRA ----
        self._ensure_lora(req.lora_name, req.lora_weight)

        # ---- IP-Adapter (identity reference) ----
        if req.reference_image:
            ref_img = _decode_b64(req.reference_image)
            self.pipe.set_ip_adapter_scale(req.reference_weight)
            ip_adapter_image = ref_img
        else:
            self.pipe.set_ip_adapter_scale(0.0)
            ip_adapter_image = None

        # ---- ControlNet (pose) ----
        # The SDXL ControlNet pipeline requires a control image. If the request
        # has no pose, we pass a blank image with controlnet_conditioning_scale=0
        # to effectively disable it.
        if req.pose_image:
            pose_input = _decode_b64(req.pose_image)
            if req.pose_extract:
                pose_input = self._extract_pose(pose_input)
            control_image = pose_input.resize((req.width, req.height))
            controlnet_scale = req.pose_weight
        else:
            control_image = Image.new("RGB", (req.width, req.height), color="black")
            controlnet_scale = 0.0

        log.info(
            "Generating: seed=%d, steps=%d, ip_weight=%.2f, pose_weight=%.2f, lora=%s",
            seed, req.steps,
            req.reference_weight if req.reference_image else 0.0,
            controlnet_scale,
            req.lora_name or "<none>",
        )

        output = self.pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=control_image,
            controlnet_conditioning_scale=controlnet_scale,
            ip_adapter_image=ip_adapter_image,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            generator=generator,
        )
        image = output.images[0]

        return GenerationResult(
            image=image,
            image_base64=_encode_b64(image),
            seed_used=seed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_pose(self, image: Image.Image) -> Image.Image:
        """Run OpenPose detector on an image to produce a pose skeleton."""
        if self._pose_detector is None:
            from controlnet_aux import OpenposeDetector

            log.info("Loading OpenPose detector (first request)")
            self._pose_detector = OpenposeDetector.from_pretrained(
                self.config.pose_detector_repo
            )
        return self._pose_detector(image)

    def _ensure_lora(self, name: Optional[str], weight: float) -> None:
        """Load (or swap, or unload) a LoRA on demand."""
        target = (name, weight) if name else None
        if target == self._current_lora:
            return

        # Unload any current LoRA
        if self._current_lora is not None:
            try:
                self.pipe.unfuse_lora()
            except Exception:
                pass
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass

        # Load the new one if requested
        if name:
            lora_path = self.config.loras_dir / f"{name}.safetensors"
            if not lora_path.exists():
                self._current_lora = None
                raise FileNotFoundError(f"LoRA not found: {lora_path}")
            log.info("Loading LoRA: %s (scale=%.2f)", name, weight)
            # Pass parent directory + explicit weight_name. Passing a full .safetensors
            # path makes diffusers fall through to its HF-repo path which errors with
            # "When using the offline mode, you must specify a `weight_name`."
            self.pipe.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
            )
            self.pipe.fuse_lora(lora_scale=weight)

        self._current_lora = target
