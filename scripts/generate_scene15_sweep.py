"""Generate a reproducible Scene15 MR=0.8 parameter sweep from one parent config."""

from pathlib import Path
import copy
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config" / "scene15_tune_mr08_s2_eta002_k5_fixedlr.yaml"
OUT = ROOT / "config" / "overnight_scene15_mr08"


def make_name(eta, steps, momentum=0.98, drop=0.2):
    return (
        f"scene15_mr08_s1_eta{eta:g}_k{steps}"
        f"_m{momentum:g}_d{drop:g}"
    ).replace(".", "p")


def main():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    candidates = []
    # Geometry sweep: deliberately do not hold eta*steps fixed.
    for eta, steps in [
        (0.005, 5), (0.010, 5), (0.015, 5), (0.020, 5),
        (0.030, 5), (0.050, 5),
        (0.010, 8), (0.015, 8), (0.020, 8), (0.030, 8),
        (0.010, 12), (0.015, 12), (0.020, 12),
        (0.020, 3), (0.030, 3), (0.050, 3),
    ]:
        candidates.append((eta, steps, 0.98, 0.2))

    # Stability sweep around the current eta=0.02, K=5 setting.
    for momentum, drop in [
        (0.99, 0.2), (0.995, 0.2),
        (0.98, 0.3), (0.99, 0.3), (0.995, 0.3),
    ]:
        candidates.append((0.020, 5, momentum, drop))

    manifest = []
    for eta, steps, momentum, drop in candidates:
        name = make_name(eta, steps, momentum, drop)
        cfg = copy.deepcopy(base)
        cfg["seed"] = 1
        cfg["momentum"] = momentum
        cfg["drop_rate"] = drop
        cfg["print_freq"] = 100
        cfg["step_size"] = eta
        cfg["steps"] = steps
        cfg["output_dir"] = f"output/overnight_scene15_mr08/{name}"
        path = OUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        manifest.append({
            "name": name,
            "parent": str(BASE.relative_to(ROOT)),
            "hypothesis": "Finite-step structural diffusion may improve final stability; eta and steps are independent controls.",
            "step_size": eta,
            "steps": steps,
            "momentum": momentum,
            "drop_rate": drop,
            "seed": 1,
        })

    (OUT / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"generated {len(manifest)} configs in {OUT}")


if __name__ == "__main__":
    main()
