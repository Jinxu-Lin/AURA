# Component: config
# Source: Codes/CLAUDE.md §5 (Config & Scripts)
"""
AURA config loader: YAML configs with _base_ inheritance, deep merge, CLI override.

Supports:
- _base_ key in YAML for explicit parent config
- Auto-fallback to configs/base.yaml if no _base_ specified
- CLI override via dot-notation: --override "damping_ekfac=0.05" "seeds=[42,123]"
- ~ expansion in path values
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path, base_config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config with inheritance support.

    Resolution order:
    1. If config has _base_ key, load that as base
    2. Else if base_config_path provided, use that
    3. Else auto-resolve to configs/base.yaml (skip if loading base.yaml itself)
    4. Deep-merge experiment config over base

    Args:
        config_path: Path to the experiment config.
        base_config_path: Explicit base config path. If None, auto-resolves.

    Returns:
        Merged config dict.
    """
    config_path = Path(config_path).resolve()
    codes_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    # Load experiment config
    with open(config_path) as f:
        experiment = yaml.safe_load(f) or {}

    # Resolve base config
    base = {}
    if "_base_" in experiment:
        # Explicit base via _base_ key
        base_path = Path(experiment.pop("_base_"))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        if base_path.exists():
            base = load_config(base_path, base_config_path="__none__")
    elif base_config_path == "__none__":
        # Internal sentinel: skip base loading (prevents infinite recursion)
        base = {}
    elif base_config_path is not None:
        if Path(base_config_path).exists():
            with open(base_config_path) as f:
                base = yaml.safe_load(f) or {}
    else:
        # Auto-resolve to configs/base.yaml (skip if we ARE base.yaml)
        auto_base = codes_dir / "configs" / "base.yaml"
        if auto_base.exists() and auto_base.resolve() != config_path:
            with open(auto_base) as f:
                base = yaml.safe_load(f) or {}

    # Deep merge: experiment overrides base
    merged = _deep_merge(base, experiment)

    # Expand ~ in string values
    _expand_tildes(merged)

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


def _expand_tildes(config: dict) -> None:
    """Expand ~ in string values in-place."""
    for key, value in config.items():
        if isinstance(value, str) and "~" in value:
            config[key] = str(Path(value).expanduser())
        elif isinstance(value, dict):
            _expand_tildes(value)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides to config via dot-notation.

    Supports:
    - Simple values: "damping_ekfac=0.05"
    - Nested keys: "gates.bss_partial_rho.pass=0.6"
    - Lists: "seeds=[42,123,456]"
    - Booleans: "dry_run=true"

    Args:
        config: Config dict to modify.
        overrides: List of "key=value" strings.

    Returns:
        Modified config dict.
    """
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        keys = key.strip().split(".")
        value = _parse_value(value.strip())

        # Navigate to the right nested dict
        target = config
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    return config


def _parse_value(value: str) -> Any:
    """Parse a string value into appropriate Python type."""
    # Try JSON parsing for lists, dicts, bools, numbers
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # String
    return value


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
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Config overrides via dot-notation (e.g., damping_ekfac=0.05)")
    return parser


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Ensure all path keys in config point to existing directories.

    Args:
        config: Config dict.

    Returns:
        Same config dict (paths created as side effect).
    """
    for key in ("data_dir", "results_dir"):
        if key in config:
            Path(config[key]).mkdir(parents=True, exist_ok=True)
    if "data_dir" in config:
        data_dir = Path(config["data_dir"])
        (data_dir / "models").mkdir(parents=True, exist_ok=True)
        (data_dir / "phase2a").mkdir(parents=True, exist_ok=True)
    return config
