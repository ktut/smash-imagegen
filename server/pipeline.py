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


def _stroke_silhouette(
    img: Image.Image, bg_mask, hex_color: str, width: int
) -> Image.Image:
    """Paint a `width`-pixel-thick border in `hex_color` along the inner edge
    of the character silhouette (character pixels that border the bg region).

    Relies on `bg_mask` being the actual bg — if bg-fill bled into the
    character, this border will bleed too. The generation side must produce
    a hard-outlined character (LoRA at 0.95+) for the bg-mask to be clean.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    arr = np.array(img.convert("RGB"))
    char_mask = ~bg_mask
    edge = binary_dilation(bg_mask, iterations=width) & char_mask
    colour = np.array(
        [int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8
    )
    arr[edge] = colour
    return Image.fromarray(arr)


def _force_background_color(
    img: Image.Image, hex_color: str, tolerance: int
):
    """Flood-fill the background region from the image edges and replace it
    with `hex_color`. The 'background region' is every pixel connected to an
    edge whose colour is within `tolerance` (per-channel L1) of the sampled
    average corner colour.

    Pure threshold replacement would clobber any character pixel that happens
    to match the bg colour; restricting to edge-connected components avoids
    that. Implemented with scipy.ndimage.label for O(n) component finding.
    """
    import numpy as np
    from scipy.ndimage import label

    arr = np.array(img.convert("RGB")).astype(np.int16)
    h, w, _ = arr.shape

    corners = np.stack([arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]])
    seed = corners.mean(axis=0)

    diff = np.abs(arr - seed).sum(axis=2)
    similar = diff < tolerance

    # Simple connected-components flood-fill. Relies on the *generation* side
    # producing hard black outlines around the character (LoRA at 0.95+ does
    # this) — black is L1 ~360 from pure green, so the flood-fill cannot
    # cross an outlined boundary even at high tolerance. If you find the
    # post-process eating character pixels, the fix is in generation
    # (restore the black outline), not here.
    labelled, _ = label(similar)
    if labelled.max() == 0:
        return img, None

    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    max_count = int(counts.max())
    if max_count == 0:
        return img, None

    # Take edge-connected components large enough to plausibly *be* the bg
    # (handles bg-split-by-character as well as the single-blob case).
    edge_labels = set()
    edge_labels.update(np.unique(labelled[0, :]).tolist())
    edge_labels.update(np.unique(labelled[-1, :]).tolist())
    edge_labels.update(np.unique(labelled[:, 0]).tolist())
    edge_labels.update(np.unique(labelled[:, -1]).tolist())
    edge_labels.discard(0)

    bg_labels = [
        lbl for lbl in edge_labels if counts[lbl] >= max_count * 0.25
    ]
    bg_mask = np.isin(labelled, bg_labels)

    target = np.array(
        [int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8
    )
    out = arr.astype(np.uint8)
    out[bg_mask] = target
    return Image.fromarray(out), bg_mask


class ImageGenPipeline:
    def __init__(self, config: Config):
        # Lazy heavy imports so module-level errors are easier to read
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            MultiControlNetModel,
            StableDiffusionXLControlNetPipeline,
        )

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            log.warning("CUDA not available — falling back to CPU. This will be very slow.")

        log.info("Loading VAE: %s", config.vae_model)
        vae = AutoencoderKL.from_pretrained(config.vae_model, torch_dtype=torch.float16)

        log.info("Loading ControlNet (OpenPose): %s", config.controlnet_openpose_model)
        controlnet_openpose = ControlNetModel.from_pretrained(
            config.controlnet_openpose_model, torch_dtype=torch.float16
        )

        log.info("Loading ControlNet (Canny): %s", config.controlnet_canny_model)
        controlnet_canny = ControlNetModel.from_pretrained(
            config.controlnet_canny_model, torch_dtype=torch.float16
        )

        # MultiControlNet — at inference we pass [openpose_img, canny_img] and
        # [openpose_scale, canny_scale]. A scale of 0 effectively disables one
        # branch (we still need to pass a placeholder image in that slot).
        controlnet = MultiControlNetModel([controlnet_openpose, controlnet_canny])

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

        # Detectors loaded lazily on first request that needs them
        self._pose_detector = None
        self._canny_detector = None

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

        # ---- ControlNets (pose + canny) ----
        # MultiControlNet expects parallel lists of images and scales. A scale
        # of 0 disables that branch, but we still pass a placeholder image.
        blank = Image.new("RGB", (req.width, req.height), color="black")

        if req.pose_image:
            pose_input = _decode_b64(req.pose_image)
            if req.pose_extract:
                pose_input = self._extract_pose(pose_input)
            pose_control = pose_input.resize((req.width, req.height))
            pose_scale = req.pose_weight
        else:
            pose_control = blank
            pose_scale = 0.0

        if req.canny_image:
            canny_input = _decode_b64(req.canny_image)
            if req.canny_extract:
                canny_input = self._extract_canny(canny_input)
            canny_control = canny_input.resize((req.width, req.height))
            canny_scale = req.canny_weight
        else:
            canny_control = blank
            canny_scale = 0.0

        log.info(
            "Generating: seed=%d, steps=%d, ip_weight=%.2f, pose_weight=%.2f, canny_weight=%.2f, lora=%s",
            seed, req.steps,
            req.reference_weight if req.reference_image else 0.0,
            pose_scale, canny_scale,
            req.lora_name or "<none>",
        )

        output = self.pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=[pose_control, canny_control],
            controlnet_conditioning_scale=[pose_scale, canny_scale],
            ip_adapter_image=ip_adapter_image,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            generator=generator,
        )
        image = output.images[0]

        if req.force_background_color:
            image, bg_mask = _force_background_color(
                image, req.force_background_color, req.background_tolerance
            )
            if req.outline_color and bg_mask is not None:
                image = _stroke_silhouette(
                    image, bg_mask, req.outline_color, req.outline_width
                )

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

    def _extract_canny(self, image: Image.Image) -> Image.Image:
        """Run Canny edge detector on an image to produce an edge map."""
        if self._canny_detector is None:
            from controlnet_aux import CannyDetector

            log.info("Loading Canny detector (first request)")
            self._canny_detector = CannyDetector()
        return self._canny_detector(image)

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
