from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "train.yaml"


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config() -> tuple[dict, Path]:
    # Carrega variaveis de ambiente e permite sobrescrever o caminho da config.
    load_dotenv()
    config_path = Path(os.getenv("TRAIN_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de config nao encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config invalida: esperado objeto YAML no topo.")
    return cfg, config_path

