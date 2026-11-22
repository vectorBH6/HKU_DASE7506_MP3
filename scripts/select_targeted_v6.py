"""Apply the V6 predeclared selection rule using only train holdout and validation."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from hashlib import sha256


def load(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    families = {
        "baseline": ("e50_v5_baseline", "onpolicy_v001/epoch_02"),
        "generator": ("e60_v6_generator", "v6_generator/epoch_01"),
        "chooser": ("e61_v6_chooser", "v6_chooser/epoch_01"),
    }
    rows = {}
    for name, (experiment, adapter) in families.items():
        diagnostic = load(f"reports/v6_{name}_diagnostic.json")
        run = Path("runs") / experiment / "validation/r001"
        score = load(run / "score.json")
        meta = load(run / "run_metadata.json")
        holdout = load(f"runs/v6_holdout/{name}/score.json")
        held_meta = load(f"runs/v6_holdout/{name}/prediction_metadata.json")
        config = "submission/configs/contract_v4_selected.json"
        weights = Path("artifacts/adapters") / adapter / "adapter_model.safetensors"
        weight_hash = digest(weights)
        assert meta["status"] == "completed" and meta["device"] == "cpu" and len(meta["cpu_affinity"]) == 4
        assert Path(meta["adapter_path"]).resolve() == (ROOT / weights.parent).resolve()
        assert diagnostic["n"] == score["n"] == 120 and diagnostic["final_correct"] == score["correct"]
        assert score["predictions_sha256"] == digest(run / "predictions.json")
        assert all(len(r["cases"]) == 3 for r in score["per_question"])
        assert held_meta["adapter_sha256"] == weight_hash and held_meta["config_sha256"] == digest(config)
        assert held_meta["device"] == "cuda" and holdout["n"] == held_meta["rows"] == 98
        assert meta["config"] == load(config)
        assert diagnostic["costs"]["max_generations"] <= 4 and diagnostic["costs"]["max_charged_tools"] <= 8
        rows[name] = {
            "adapter": str(weights.parent), "adapter_sha256": weight_hash, "validation_run": str(run),
            "validation_correct": score["correct"], "validation_covered": diagnostic["tasks_with_correct_checked_candidate"],
            "validation_generations": diagnostic["costs"]["model_generations"],
            "holdout_correct": holdout["correct"], "holdout_covered": holdout["covered"],
        }
    base = rows["baseline"]
    eligible = [name for name, row in rows.items() if row["validation_correct"] > base["validation_correct"]
                and row["holdout_correct"] >= base["holdout_correct"]]
    eligible.sort(key=lambda name: (-rows[name]["validation_correct"], -rows[name]["holdout_correct"],
                                    rows[name]["validation_generations"], name))
    lock = {"locked_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": digest("reports/v6_plan_cn.md"),
            "inference_config": config, "config_sha256": digest(config),
            "rows": rows, "eligible": eligible, "selected_for_test": eligible[0] if eligible else None}
    # Once a decision exists it is immutable. Re-running verifies the same decision.
    path = ROOT / "reports/v6_selection.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if {**lock, "locked_utc": existing["locked_utc"]} != existing:
            raise ValueError("Existing selection does not match the current validation artifacts")
    else:
        with path.open("x") as out:
            json.dump(lock, out, ensure_ascii=False, indent=2)
            out.write("\n")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
