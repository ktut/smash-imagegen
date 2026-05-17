"""Config loaded from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    base_model: str
    vae_model: str
    controlnet_openpose_model: str
    ip_adapter_repo: str
    ip_adapter_subfolder: str
    ip_adapter_weight_name: str
    pose_detector_repo: str
    loras_dir: Path
    outputs_dir: Path
    host: str
    port: int


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text())

    loras_dir = Path(data.get("loras_dir", "loras")).resolve()
    outputs_dir = Path(data.get("outputs_dir", "outputs")).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    loras_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        base_model=data["base_model"],
        vae_model=data["vae_model"],
        controlnet_openpose_model=data["controlnet_openpose_model"],
        ip_adapter_repo=data["ip_adapter"]["repo"],
        ip_adapter_subfolder=data["ip_adapter"]["subfolder"],
        ip_adapter_weight_name=data["ip_adapter"]["weight_name"],
        pose_detector_repo=data.get("pose_detector_repo", "lllyasviel/Annotators"),
        loras_dir=loras_dir,
        outputs_dir=outputs_dir,
        host=data.get("host", "0.0.0.0"),
        port=data.get("port", 8000),
    )
