"""Build the V12 course upload files, or test the exported runtime in a fresh starter."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e121_v12_fresh"
LOCK = ROOT / "reports/v12/v12_replay_lock.json"
DEST = ROOT / "dist/v12"
OUTPUTS = ROOT / "outputs"
REPORT = Path("reports/v12/FINAL_REPORT.md")
STAGES = (
    ("sft_v002", "sft_v001"),
    ("onpolicy_v001", "onpolicy_v001"),
    ("v7_paired", "v7/paired"),
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# These modules and package initializer form the import closure of the scored factory.
RUNTIME_MODULES = {
    "__init__",
    "contract_agent",
    "feedback_agent",
    "model",
    "pool_edits_v8",
    "pool_rank_v8",
    "recovery_agent",
    "review_agent",
    "review_pool_v8",
    "review_prompts",
    "review_terminal_v9",
    "semantic_agent",
    "sql_utils",
}


def read(path):
    return json.loads(path.read_text())


def dump(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(root, name, data):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def archive(folder, outfile, limit):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if sum(p.stat().st_size for p in files) > limit:
        raise RuntimeError(f"Extracted-size limit exceeded for {outfile.name}")
    with ZipFile(outfile, "w") as zipf:
        for p in files:
            record = ZipInfo(p.relative_to(folder).as_posix(), (2026, 1, 1, 0, 0, 0))
            record.external_attr = 0o100644 << 16
            record.create_system = 3
            zipf.writestr(
                record, p.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9
            )
    if outfile.stat().st_size > limit:
        raise RuntimeError(f"Archive-size limit exceeded for {outfile.name}")
    with ZipFile(outfile) as zipf:
        if zipf.testzip():
            raise RuntimeError(f"Invalid ZIP: {outfile}")


def stage_runtime(source, adapter):
    lock = read(LOCK)
    snapshot = RUN / "test/r001/source_snapshot/submission"
    for name, expected in lock["files"].items():
        p = Path(name)
        if not name.startswith("submission/"):
            continue
        dest = (
            Path("submission.py")
            if name == lock["entrypoint"]
            else p.relative_to("submission")
        )
        content = (ROOT / p).read_bytes()
        if sha(content) != expected or content != (snapshot / dest).read_bytes():
            raise RuntimeError(f"V12 snapshot differs for {p}")
        if p.parent.name == "sql_repair_agent" and p.stem not in RUNTIME_MODULES:
            continue
        put(source, dest, content)
    put(
        source,
        "configs/selected.json",
        (source / "configs/v12_v9_fallback.json").read_bytes(),
    )
    put(source, "requirements.txt", (ROOT / "submission/requirements.txt").read_bytes())
    put(source, "reproduce.py", (ROOT / "scripts/reproduce_v12_export.py").read_bytes())
    for name in ADAPTER_FILES:
        p = Path(lock["adapter"]) / name
        content = (ROOT / p).read_bytes()
        if sha(content) != lock["files"][str(p)]:
            raise RuntimeError(f"Adapter identity mismatch for {name}")
        if name == "adapter_config.json":
            cfg = json.loads(content)
            cfg["base_model_name_or_path"] = "Qwen/Qwen3-0.6B"
            content = dump(cfg)
        put(adapter, name, content)
    for split, correct, n in (("test", 191, 240), ("validation", 99, 120)):
        run = RUN / split / "r001"
        pred, grade = (
            (run / "predictions.json").read_bytes(),
            (run / "score.json").read_bytes(),
        )
        summary = json.loads(grade)
        if (
            summary["split"],
            summary["correct"],
            summary["n"],
            summary["predictions_sha256"],
        ) != (split, correct, n, sha(pred)):
            raise RuntimeError(f"Unexpected score or predictions hash in {split}")
        put(source, "outputs/" + split + ".json", pred)
        put(source, "outputs/" + split + "-score.json", grade)


def ensure_public_outputs(source):
    for name in ("test.json", "test-score.json"):
        data = (source / "outputs" / name).read_bytes()
        dest = OUTPUTS / name
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f"Conflicting existing output: {dest}")
        put(OUTPUTS, name, data)


def stage_evidence(source):
    for split in ("validation", "test"):
        run = RUN / split / "r001"
        records = []
        with (run / "generations.jsonl").open() as log:
            for line in log:
                record = json.loads(line)
                record["identity"]["base_path"] = "$MP3_BASE_MODEL"
                record["identity"]["adapter"]["path"] = "$MP3_ADAPTER_DIR"
                records.append(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
        log_bytes = ("\n".join(records) + "\n").encode()
        put(
            source,
            f"audit/{split}/generations.jsonl.gz",
            gzip.compress(log_bytes, mtime=0),
        )
        summary = read(ROOT / f"reports/v12/v12_official_{split}.json")["candidate"]
        put(
            source,
            f"audit/{split}/resources_and_totals.json",
            dump(
                {
                    k: summary[k]
                    for k in (
                        "n",
                        "correct",
                        "total_charged_tools",
                        "generations",
                        "resources",
                    )
                }
            ),
        )
    for stage, prepared in STAGES:
        training = ROOT / "runs/training" / stage
        prefix = "training_records/" + stage + "/"
        for src, dest in (
            (training / "training_metadata.json", "training_metadata.json"),
            (training / "loss.jsonl", "loss.jsonl"),
            (ROOT / f"training/configs/lora_{stage}.json", "config.json"),
            (
                ROOT / "artifacts/prepared_data" / prepared / "provenance.json",
                "provenance.json",
            ),
        ):
            put(source, prefix + dest, src.read_bytes())
    put(
        source,
        "training_records/v7_paired/verification.json",
        (ROOT / "runs/training/v7_paired/artifact_verification.json").read_bytes(),
    )
    put(
        source,
        "training_records/v8_pool_audit.json",
        (ROOT / "runs/v8_pool_train/audit.json").read_bytes(),
    )
    put(
        source,
        "training_records/v8_pool_verification.json",
        (ROOT / "runs/v8_pool_train/verification.json").read_bytes(),
    )
    put(source, REPORT, (ROOT / REPORT).read_bytes())
    put(
        source,
        REPORT.with_suffix(".pdf"),
        (ROOT / REPORT.with_suffix(".pdf")).read_bytes(),
    )
    for image in sorted((ROOT / REPORT.parent / "figures").glob("*.png")):
        put(source, REPORT.parent / "figures" / image.name, image.read_bytes())
    if not list((source / REPORT.parent / "figures").glob("*.png")):
        raise RuntimeError("Missing report figures")
    put(source, "README.md", (ROOT / "scripts/v12_export_readme.md.in").read_bytes())
    for file in source.rglob("*"):
        if not file.is_file() or file.suffix in {".png", ".pdf"}:
            continue
        data = file.read_bytes()
        if file.suffix == ".gz":
            data = gzip.decompress(data)
        if any(
            token in data
            for token in (
                b"/home/rvector/",
                b"/tmp/",
                b"BEGIN PRIVATE KEY",
                b"ghp_",
                b"github_pat_",
            )
        ):
            raise ValueError(
                f"Unexpected local path or private credential material in {file.name}"
            )


def clean_starter(folder):
    official = read(ROOT / "PACKAGE_MANIFEST.json")["files"]
    for name, entry in official.items():
        path = ROOT / name
        if name == "README.md" and sha(path.read_bytes()) != entry["sha256"]:
            path = ROOT / "README_en.md"
        data = path.read_bytes()
        if sha(data) != entry["sha256"]:
            raise RuntimeError(f"Official starter hash differs: {name}")
        put(folder, name, data)
    put(folder, "PACKAGE_MANIFEST.json", (ROOT / "PACKAGE_MANIFEST.json").read_bytes())


def audit_runtime(source, adapter, scratch):
    starter = scratch / "starter"
    clean_starter(starter)
    # Use only the packaging outputs and a clean course starter on Python's import path.
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, (source, starter))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="42",
        MP3_BASE_MODEL=str(ROOT / "artifacts/models/Qwen3-0.6B"),
        MP3_ADAPTER_DIR=str(adapter),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        MP3_DEVICE="cpu",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        MP3_GENERATION_LOG=str(scratch / "smoke-generations.jsonl"),
    )
    out = scratch / "rescored"
    command = [
        sys.executable,
        "-B",
        str(source / "reproduce.py"),
        "--starter",
        str(starter),
        "--split",
        "test",
        "--output-dir",
        str(out),
        "--score-only",
    ]
    subprocess.run(command, env=env, cwd=scratch, check=True)
    expected_score = read(source / "outputs/test-score.json")
    actual = read(out / "test-score.json")
    actual.pop("scoring_seconds")
    expected_score.pop("scoring_seconds")
    if actual != expected_score:
        raise ValueError("Independent official replay score differs")
    # One known validation task with a live generation, chosen independently of its grade.
    first_record = json.loads(
        (RUN / "validation/r001/generations.jsonl").open().readline()
    )
    task = first_record["task_id"]
    probe = """import json,os,sys
from pathlib import Path
from mp3_sql.common import load_bank
from mp3_sql.environment import Tools,rollout
from mp3_sql.predict import load_factory
from sql_repair_agent import model
source,target,task=sys.argv[1:]
assert Path(model.__file__).resolve().is_relative_to(source)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
rows,manifest,folder=load_bank('validation')
row=next(x for x in rows if x['id']==task)
factory=load_factory(Path(source)/'submission.py')
trace=rollout(row,Tools(folder,manifest),factory(Path(source)/'configs/selected.json'))
Path(target).write_text(json.dumps(trace))
"""
    tracefile = scratch / "smoke-trace.json"
    subprocess.run(
        [sys.executable, "-B", "-c", probe, str(source), str(tracefile), task],
        env=env,
        cwd=scratch,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    expected_trace = next(
        p
        for p in read(source / "outputs/validation.json")["predictions"]
        if p["id"] == task
    )
    if expected_trace != read(tracefile):
        raise ValueError("Live exported validation rollout differs")
    actual_gen = list(map(json.loads, (scratch / "smoke-generations.jsonl").open()))
    expected_gen = [
        r
        for r in map(json.loads, (RUN / "validation/r001/generations.jsonl").open())
        if r["task_id"] == task
    ]
    keys = (
        "messages",
        "rendered_prompt",
        "generated_text",
        "input_tokens",
        "new_tokens",
        "generation",
        "attempt",
    )
    if not actual_gen or len(actual_gen) != len(expected_gen):
        raise ValueError("Unexpected number of smoke generations")
    for a, b in zip(actual_gen, expected_gen, strict=True):
        if (
            any(a[k] != b[k] for k in keys)
            or a["identity"]["adapter"]["sha256"] != b["identity"]["adapter"]["sha256"]
        ):
            raise ValueError("Exported prompts, generation or adapter identity differs")
    print(
        f"Independent replay matched on all 240 scores and three-instance outcomes. "
        f"Isolated live validation smoke matched the full tool trace and {len(actual_gen)} generations."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Check the independent public outputs and a live task",
    )
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="v12_export_") as td:
        scratch = Path(td)
        source, adapter = scratch / "source", scratch / "adapter"
        stage_runtime(source, adapter)
        ensure_public_outputs(source)
        if args.verify_runtime:
            archive(source, scratch / "source-smoke.zip", 16 * 2**20)
            archive(adapter, scratch / "adapter.zip", 512 * 2**20)
            for z, target in (
                (scratch / "source-smoke.zip", scratch / "exported"),
                (scratch / "adapter.zip", scratch / "lora"),
            ):
                with ZipFile(z) as f:
                    f.extractall(target)
            audit_runtime(scratch / "exported", scratch / "lora", scratch)
            return
        stage_evidence(source)
        DEST.mkdir(parents=True, exist_ok=True)
        for p, limit in ((source, 16 * 2**20), (adapter, 512 * 2**20)):
            target = scratch / (p.name + ".zip")
            archive(p, target, limit)
            final = DEST / target.name
            # Replace on the destination filesystem so unfinished ZIPs never become visible.
            with tempfile.NamedTemporaryFile(dir=DEST, delete=False) as temp:
                temp_path = Path(temp.name)
                with target.open("rb") as f:
                    shutil.copyfileobj(f, temp)
            temp_path.chmod(0o644)
            temp_path.replace(final)
            print(
                f"{final.relative_to(ROOT)}   {final.stat().st_size:,} bytes   SHA-256 {sha(final.read_bytes())}"
            )


if __name__ == "__main__":
    main()
