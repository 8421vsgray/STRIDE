"""Select top seed-1 candidates per mismatch ratio and validate seeds 2-5."""

from pathlib import Path
import re
import subprocess
import yaml


ROOT = Path("/root/code/NEWSTRIDE")
PYTHON = Path("/root/miniconda3/envs/stride/bin/python")
RATIOS = ("08", "0.5", "0.2", "0")
ACC_RE = re.compile(r"Average K-means Result\s*: ACC =\s*([0-9.]+)")


def ratio_dir(ratio):
    return ROOT / "config" / f"overnight_scene15_mr{ratio}"


def select_candidates(ratio, top_n=2):
    log_dir = ROOT / "tuning_logs" / f"overnight_scene15_mr{ratio}"
    rows = []
    for cfg in sorted(ratio_dir(ratio).glob("*.yaml")):
        if cfg.name == "manifest.yaml":
            continue
        log = log_dir / f"{cfg.stem}.log"
        if not log.exists():
            continue
        values = [float(x) for x in ACC_RE.findall(log.read_text(errors="ignore"))]
        if values:
            rows.append((values[-1], max(values), cfg))
    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return rows[:top_n], rows


def main():
    selected = []
    all_rows = {}
    for ratio in RATIOS:
        rows, all_seen = select_candidates(ratio)
        all_rows[ratio] = all_seen
        selected.extend((ratio, row) for row in rows)

    manifest = []
    for ratio, (final, best, cfg_path) in selected:
        base = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        for seed in (2, 3, 4, 5):
            cfg = dict(base)
            cfg["seed"] = seed
            name = f"{cfg_path.stem}_seed{seed}"
            cfg["output_dir"] = f"output/multiseed_scene15_mr{ratio}/{name}"
            out_cfg = ROOT / "config" / "multiseed_scene15" / f"{name}.yaml"
            out_cfg.parent.mkdir(parents=True, exist_ok=True)
            out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            manifest.append({
                "mismatch_ratio": ratio,
                "parent_candidate": cfg_path.name,
                "seed1_final": final,
                "seed1_best": best,
                "seed": seed,
                "config": str(out_cfg.relative_to(ROOT / "config")),
            })

    (ROOT / "tuning_logs" / "multiseed_scene15_selection.yaml").write_text(
        yaml.safe_dump({"selected": manifest, "screened_counts": {k: len(v) for k, v in all_rows.items()}}, sort_keys=False),
        encoding="utf-8",
    )

    for item in manifest:
        cfg_rel = item["config"]
        log_dir = ROOT / "tuning_logs" / "multiseed_scene15" / f"mr{item['mismatch_ratio']}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{Path(cfg_rel).stem}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [str(PYTHON), "-u", str(ROOT / "main_train.py"), "--config", cfg_rel],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )


if __name__ == "__main__":
    main()
