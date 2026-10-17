"""Resumable seed-1 screening queue, ordered from high to low mismatch."""

from pathlib import Path
import os
import re
import subprocess
import time


ROOT = Path("/root/code/NEWSTRIDE")
PYTHON = "/root/miniconda3/envs/stride/bin/python"
RATIOS = ("08", "0.5", "0.2", "0")


def active_training():
    result = subprocess.run(
        ["pgrep", "-f", "[m]ain_train.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def valid_complete(log_path):
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="ignore")
    return "Total training time:" in text and "FileNotFoundError" not in text


def main():
    while active_training():
        time.sleep(30)

    for ratio in RATIOS:
        config_dir = ROOT / "config" / f"overnight_scene15_mr{ratio}"
        log_dir = ROOT / "tuning_logs" / f"overnight_scene15_mr{ratio}"
        log_dir.mkdir(parents=True, exist_ok=True)
        for config_path in sorted(config_dir.glob("*.yaml")):
            if config_path.name == "manifest.yaml":
                continue
            log_path = log_dir / f"{config_path.stem}.log"
            if valid_complete(log_path):
                continue
            log_path.unlink(missing_ok=True)
            config_rel = str(config_path.relative_to(ROOT / "config"))
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    [PYTHON, "-u", str(ROOT / "main_train.py"), "--config", config_rel],
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )


if __name__ == "__main__":
    main()
