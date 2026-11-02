from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    project_root = config_path.parent.parent
    config["data_dir"] = (project_root / config["data_dir"]).resolve()
    config["output_dir"] = (project_root / config["output_dir"]).resolve()
    return config, project_root
