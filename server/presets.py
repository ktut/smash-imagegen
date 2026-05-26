"""Preset loading and resolution.

A preset is a YAML file in `presets/` that bundles a curated recipe — a prompt
template, a negative prompt, sensible defaults for every GenerateRequest knob,
and a list of named template variables the caller must fill in. It exists so
you don't have to remember 14 parameters every time you want to generate
something stylistically similar to last time.

The loader is intentionally simple: read YAML, validate required fields,
substitute `{var}` tokens in the prompt template, merge user overrides on top
of defaults, hand a fully-formed GenerateRequest dict back to the API layer.
No Jinja, no inheritance, no environment — keep it inspectable.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class PresetVar:
    name: str
    description: str = ""
    required: bool = True
    default: Optional[str] = None


@dataclass
class Preset:
    name: str
    description: str
    prompt_template: str
    negative_prompt: str
    defaults: dict[str, Any]
    vars: list[PresetVar] = field(default_factory=list)
    requires_loras: list[str] = field(default_factory=list)

    # The path the preset was loaded from — handy for error messages and
    # for /presets/{name} responses.
    source: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Preset":
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping")
        for required in ("name", "prompt_template", "defaults"):
            if required not in data:
                raise ValueError(f"{path}: missing required key '{required}'")

        vars_ = [
            PresetVar(
                name=v["name"],
                description=v.get("description", ""),
                required=v.get("required", True),
                default=v.get("default"),
            )
            for v in data.get("vars") or []
        ]
        return cls(
            name=data["name"],
            description=(data.get("description") or "").strip(),
            prompt_template=data["prompt_template"],
            negative_prompt=(data.get("negative_prompt") or "").strip(),
            defaults=dict(data["defaults"] or {}),
            vars=vars_,
            requires_loras=list(data.get("requires_loras") or []),
            source=path,
        )

    def resolve(
        self,
        vars: Optional[dict[str, str]] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Produce a GenerateRequest-compatible dict from this preset.

        `vars`      — values for the named template variables
        `overrides` — direct GenerateRequest field overrides (seed, candidates,
                      images, or any knob the caller wants to tweak vs defaults)
        """
        vars = vars or {}
        overrides = overrides or {}

        # Required vars must all be supplied (unless they have a default)
        missing = [
            v.name for v in self.vars
            if v.required and v.default is None and v.name not in vars
        ]
        if missing:
            raise ValueError(
                f"preset '{self.name}' is missing required vars: {missing}"
            )

        # Materialise defaults for vars that weren't supplied
        all_vars = {v.name: v.default for v in self.vars if v.default is not None}
        all_vars.update(vars)

        # Format the prompt template. Use str.format_map with a guard so an
        # unknown {placeholder} produces a clear error instead of a silent
        # KeyError deep in the stack.
        try:
            prompt = self.prompt_template.format_map(_SafeDict(all_vars))
        except KeyError as e:
            raise ValueError(
                f"preset '{self.name}': prompt template references unknown var {e}"
            )

        # Collapse multi-line prompt blocks into a single line (YAML multiline
        # gives us \n between lines; SDXL doesn't care but it's tidier in logs).
        prompt = " ".join(line.strip() for line in prompt.splitlines() if line.strip())
        negative = " ".join(
            line.strip() for line in self.negative_prompt.splitlines() if line.strip()
        )

        body: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative,
        }
        body.update(self.defaults)
        body.update(overrides)
        return body


class _SafeDict(dict):
    """str.format_map helper that raises KeyError with the missing key name."""
    def __missing__(self, key):  # pragma: no cover — tiny helper
        raise KeyError(key)


# ---------------------------------------------------------------------------
# Registry — loaded once at startup, exposed via /presets endpoints
# ---------------------------------------------------------------------------


class PresetRegistry:
    def __init__(self, presets_dir: Path):
        self.presets_dir = presets_dir
        self._presets: dict[str, Preset] = {}
        self.reload()

    def reload(self) -> None:
        self._presets.clear()
        if not self.presets_dir.exists():
            return
        for path in sorted(self.presets_dir.glob("*.yaml")):
            try:
                preset = Preset.from_yaml(path)
            except Exception as e:  # noqa: BLE001 — log + skip, don't crash boot
                import logging
                logging.getLogger("smash-imagegen.presets").warning(
                    "Skipping malformed preset %s: %s", path, e
                )
                continue
            self._presets[preset.name] = preset

    def list(self) -> list[Preset]:
        return list(self._presets.values())

    def get(self, name: str) -> Preset:
        if name not in self._presets:
            raise KeyError(name)
        return self._presets[name]
