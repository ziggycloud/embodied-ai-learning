#!/usr/bin/env python3
"""Run an upstream script with Warp memory pools disabled.

Usage (from microduck-playground):
    uv run python ../scripts/run_warp_compat.py scripts/finetune_basketball.py ...

Keeps CUDA enabled. mjlab 1.3.0 disables graph capture without memory pools.
"""
import runpy
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_warp_compat.py SCRIPT [ARGS...]")
    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise SystemExit(f"Script not found: {script}")
    import warp as wp
    wp.config.enable_mempools_at_init = False
    wp.init()
    for device in wp.get_cuda_devices():
        wp.set_mempool_enabled(device, False)
        print(f"{device.alias}: mempool enabled = {wp.is_mempool_enabled(device)}", flush=True)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
