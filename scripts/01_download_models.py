"""Pre-download the configured CLIP checkpoint into the local cache.

Usage:
    PYTHONPATH=. python scripts/01_download_models.py [--config config/model_config.yaml]
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.models.clip import CLIPLoader


def main() -> None:
    """Download and cache the checkpoint referenced by the model config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/model_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    loader = CLIPLoader(config.model, config.memory)

    print(f"Downloading '{config.model.name}' into '{config.model.cache_dir}' ...")
    loader.load()
    print("Done.")


if __name__ == "__main__":
    main()
