"""Generate seed-1 screening configs for all Scene15 mismatch ratios."""

from pathlib import Path
import copy
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config" / "scene15_tune_mr08_s2_eta002_k5_fixedlr.yaml"

GEOMETRY = [
    (0.005, 5), (0.010, 5), (0.015, 5), (0.020, 5), (0.030, 5),
    (0.010, 8), (0.015, 8), (0.020, 8),
    (0.010, 12), (0.015, 12), (0.020, 12),
    (0.020, 3), (0.030, 3),
]

STABILITY = [
    (0.020, 5, 0.990, 0.2),
    (0.020, 5, 0.995, 0.2),
    (0.020, 5, 0.980, 0.3),
    (0.020, 5, 0.990, 0.3),
]


def name(mr, eta, steps, momentum=0.98, drop=0.2):
    return (
        f"scene15_mr{mr:g}_s1_eta{eta:g}_k{steps}"
        f"_m{momentum:g}_d{drop:g}"
    ).replace(".", "p")


def main():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    manifest = []
    for mr in (0.0, 0.2, 0.5):
        out = ROOT / "config" / f"overnight_scene15_mr{mr:g}"
        out.mkdir(parents=True, exist_ok=True)
        candidates = [(eta, steps, 0.98, 0.2) for eta, steps in GEOMETRY]
        candidates.extend(STABILITY)
        for eta, steps, momentum, drop in candidates:
            exp_name = name(mr, eta, steps, momentum, drop)
            cfg = copy.deepcopy(base)
            cfg["seed"] = 1
            cfg["m_ratio"] = mr
            cfg["momentum"] = momentum
            cfg["drop_rate"] = drop
            cfg["print_freq"] = 100
            cfg["step_size"] = eta
            cfg["steps"] = steps
            cfg["output_dir"] = f"output/overnight_scene15_mr{mr:g}/{exp_name}"
            (out / f"{exp_name}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )
            manifest.append({
                "name": exp_name,
                "mismatch_ratio": mr,
                "parent": str(BASE.relative_to(ROOT)),
                "hypothesis": "Eta and diffusion depth are independent controls; smoother or stronger diffusion may improve final stability.",
                "step_size": eta,
                "steps": steps,
                "momentum": momentum,
                "drop_rate": drop,
                "seed": 1,
            })
        (out / "manifest.yaml").write_text(
            yaml.safe_dump([x for x in manifest if x["mismatch_ratio"] == mr], sort_keys=False),
            encoding="utf-8",
        )
    print(f"generated {len(manifest)} configs")


if __name__ == "__main__":
    main()
