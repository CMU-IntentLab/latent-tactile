"""
Load config from YAML. CLI args override config values.

Usage in script:
  from config import load_config, merge_args

  cfg = load_config("train_wm", config_path="configs/default.yaml")
  args = merge_args(cfg, parse_args())  # parse_args returns minimal CLI overrides
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    def _yaml_load(f):
        return yaml.safe_load(f)
    HAS_YAML = True
except ImportError:
    try:
        from ruamel.yaml import YAML
        _ryaml = YAML(typ="safe", pure=True)
        def _yaml_load(f):
            return _ryaml.load(f)
        HAS_YAML = True
    except ImportError:
        def _yaml_load(f):
            return {}
        HAS_YAML = False

CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def load_config(section: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load config for section (train_wm, eval_wm, train_decoder, eval_decoder).
    Returns flat dict: dataset keys and section keys at top level.
    """
    path = config_path or str(DEFAULT_CONFIG)
    if not os.path.isfile(path):
        return {}

    if not HAS_YAML:
        return {}

    with open(path) as f:
        data = _yaml_load(f) or {}

    dataset = data.get("dataset", {})
    section_data = data.get(section, {})
    # Merge: dataset first, then section overrides
    merged = {**dataset, **section_data}
    return merged


def get_wandb_config(section: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build config dict for wandb.init(config=...).
    Includes effective args (vars) and the loaded config file defaults.
    """
    effective = dict(vars(args))
    config_path = getattr(args, "config", None)
    if config_path and os.path.isfile(config_path):
        file_cfg = load_config(section, config_path)
        effective["config_file_path"] = config_path
        effective["config_file_defaults"] = file_cfg
    return effective


def merge_args(config: Dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    """
    Merge config dict into CLI namespace. CLI values override config.
    Returns a new namespace with all keys.
    """
    out = argparse.Namespace()
    for k, v in config.items():
        setattr(out, k, v)
    for k in dir(cli):
        if k.startswith("_"):
            continue
        v = getattr(cli, k)
        # Override with CLI value if it was explicitly set (not default None for paths)
        setattr(out, k, v)
    return out
