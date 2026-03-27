"""
AURA config loader: loads YAML configs with inheritance (base + experiment overlay).
"""

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path, base_config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, optionally overlaying on a base config.

    Args:
        config_path: Path to the experiment config.
        base_config_path: Path to base config. If None, auto-resolves to configs/base.yaml.

    Returns:
        Merged config dict (base values overridden by experiment values).
    """
    config_path = Path(config_path)
    codes_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    # Load base config
    if base_config_path is None:
        base_config_path = codes_dir / "configs" / "base.yaml"

    base = {}
    if Path(base_config_path).exists():
        with open(base_config_path) as f:
            base = yaml.safe_load(f) or {}

    # Load experiment config
    with open(config_path) as f:
        experiment = yaml.safe_load(f) or {}

    # Deep merge: experiment overrides base
    merged = _deep_merge(base, experiment)

    # Expand ~ in path values
    for key in merged:
        if isinstance(merged[key], str) and "~" in merged[key]:
            merged[key] = str(Path(merged[key]).expanduser())

    return merged


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay wins on conflicts."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common CLI arguments shared across train.py and evaluate.py.

    Args:
        parser: ArgumentParser to extend.

    Returns:
        The same parser, extended with common flags.
    """
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode: minimal steps, no GPU-heavy work")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config")
    return parser


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Ensure all path keys in config point to existing directories (create if needed).

    Args:
        config: Config dict.

    Returns:
        Same config dict (paths created as side effect).
    """
    for key in ("data_dir", "results_dir"):
        if key in config:
            Path(config[key]).mkdir(parents=True, exist_ok=True)
    # Sub-paths under data_dir
    if "data_dir" in config:
        data_dir = Path(config["data_dir"])
        (data_dir / "models").mkdir(parents=True, exist_ok=True)
        (data_dir / "phase2a").mkdir(parents=True, exist_ok=True)
    return config
