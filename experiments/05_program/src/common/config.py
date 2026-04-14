from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'pyyaml'. Install it to read *.yaml configs."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_all_configs(base_dir: Path) -> Dict[str, Any]:
    protocol = load_yaml_config(base_dir / "protocol.yaml")
    sym = load_yaml_config(base_dir / "models_symbolic.yaml")
    neu = load_yaml_config(base_dir / "models_neural.yaml")
    llm = load_yaml_config(base_dir / "models_llm.yaml")
    return {
        "protocol": protocol,
        "symbolic": sym,
        "neural": neu,
        "llm": llm,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
