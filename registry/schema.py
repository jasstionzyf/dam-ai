"""Model registry: YAML-backed, schema-validated model addressing.

Registry is the single addressing entry: API `model` name -> local path
(container mount point /models). Code only ever uses `from_pretrained(local_path)`.
Invalid registry (bad YAML, missing/unknown fields, duplicates) refuses startup.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_LOADERS = {"sentence_transformers", "open_clip"}
VALID_ENGINES = {"transformers", "open_clip", "vllm"}
VALID_MODALITIES = {"image", "text"}
VALID_POOLING = {"cls", "mean", "pooler", "last_token"}


class RegistryError(Exception):
    """Raised when the model registry is missing, unparsable or schema-invalid."""


@dataclass(frozen=True)
class Preprocess:
    resolution: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pooling: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "mean": list(self.mean),
            "std": list(self.std),
            "pooling": self.pooling,
        }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    loader: str
    engine: str
    dims: int
    modalities: tuple[str, ...]
    normalized: bool
    preprocess: Preprocess
    description: str = ""
    weights: str = field(default="")  # optional: weights filename inside path

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["modalities"] = list(self.modalities)
        d["preprocess"] = self.preprocess.to_dict()
        return {k: v for k, v in d.items() if v != ""}

    @property
    def dam_ai_meta(self) -> dict[str, Any]:
        return {
            "modalities": list(self.modalities),
            "dims": self.dims,
            "normalized": self.normalized,
            "engine": self.engine,
            "loader": self.loader,
        }


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RegistryError(msg)


def _parse_entry(raw: dict[str, Any], index: int) -> ModelSpec:
    _require(isinstance(raw, dict),
             f"models[{index}]: entry must be a mapping, got {type(raw).__name__}")
    allowed = {"name", "path", "loader", "engine", "dims", "modalities", "normalized",
               "preprocess", "description", "weights"}
    unknown = set(raw) - allowed
    _require(not unknown,
             f"models[{index}] ({raw.get('name', '?')}): unknown fields {sorted(unknown)}")

    required = ("name", "path", "loader", "engine",
                "dims", "modalities", "normalized", "preprocess")
    for key in required:
        _require(key in raw, f"models[{index}]: missing required field '{key}'")

    name = raw["name"]
    _require(isinstance(name, str) and name.strip() != "",
             f"models[{index}]: name must be non-empty string")

    preprocess_raw = raw["preprocess"]
    _require(isinstance(preprocess_raw, dict), f"{name}: preprocess must be a mapping")
    _require(set(preprocess_raw) == {"resolution", "mean", "std", "pooling"},
             f"{name}: preprocess must have exactly resolution/mean/std/pooling, "
             f"got {sorted(preprocess_raw)}")
    resolution = preprocess_raw["resolution"]
    _require(isinstance(resolution, int) and not isinstance(resolution, bool) and resolution > 0,
             f"{name}: preprocess.resolution must be positive int")
    mean, std = preprocess_raw["mean"], preprocess_raw["std"]
    for label, seq in (("mean", mean), ("std", std)):
        _require(isinstance(seq, list) and len(seq) == 3
                 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in seq),
                 f"{name}: preprocess.{label} must be list of 3 numbers")
    pooling = preprocess_raw["pooling"]
    _require(pooling in VALID_POOLING,
             f"{name}: preprocess.pooling must be one of {sorted(VALID_POOLING)}")

    _require(raw["loader"] in VALID_LOADERS,
             f"{name}: loader must be one of {sorted(VALID_LOADERS)}")
    _require(raw["engine"] in VALID_ENGINES,
             f"{name}: engine must be one of {sorted(VALID_ENGINES)}")
    dims = raw["dims"]
    _require(isinstance(dims, int) and not isinstance(dims, bool) and dims > 0,
             f"{name}: dims must be positive int")
    modalities = raw["modalities"]
    _require(isinstance(modalities, list) and bool(modalities)
             and all(m in VALID_MODALITIES for m in modalities),
             f"{name}: modalities must be non-empty list subset of {sorted(VALID_MODALITIES)}")
    _require(len(modalities) == len(set(modalities)), f"{name}: modalities has duplicates")
    _require(isinstance(raw["normalized"], bool), f"{name}: normalized must be bool")

    mean = tuple(float(x) for x in mean)  # type: ignore[assignment]
    std = tuple(float(x) for x in std)  # type: ignore[assignment]
    return ModelSpec(
        name=name,
        path=raw["path"],
        loader=raw["loader"],
        engine=raw["engine"],
        dims=dims,
        modalities=tuple(modalities),
        normalized=raw["normalized"],
        preprocess=Preprocess(resolution=resolution, mean=mean, std=std, pooling=pooling),
        description=raw.get("description", ""),
        weights=raw.get("weights", ""),
    )


def load_registry(path: str | Path) -> dict[str, ModelSpec]:
    """Load and validate the model registry. Raises RegistryError on any problem."""
    p = Path(path)
    if not p.is_file():
        raise RegistryError(f"registry file not found: {p}")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise RegistryError(f"registry YAML parse error: {e}") from e

    _require(isinstance(doc, dict), "registry top-level must be a mapping")
    _require(set(doc) == {"models"},
             f"registry top-level must be exactly {{models}}, got {sorted(doc)}")
    _require(isinstance(doc["models"], list) and bool(doc["models"]),
             "registry.models must be a non-empty list")

    specs: dict[str, ModelSpec] = {}
    for i, raw in enumerate(doc["models"]):
        spec = _parse_entry(raw, i)
        _require(spec.name not in specs, f"duplicate model name: {spec.name}")
        specs[spec.name] = spec
    return specs


def load_registry_or_die(path: str | Path) -> dict[str, ModelSpec]:
    """load_registry, with a fatal-exit wrapper message for server startup."""
    try:
        return load_registry(path)
    except RegistryError as e:
        raise SystemExit(f"dam-ai registry refused to start: {e}") from e
