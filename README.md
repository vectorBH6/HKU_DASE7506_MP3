# DASE7506 Mini-project 3 · SQL Repair Agent

| Student | Student ID | Evaluated method |
| --- | --- | --- |
| Li Shanghang | 3036705979 | V12 SQL repair agent |

V12 repaired or preserved the correct SQL on **191 of 240 public test tasks (79.58%)**. Each answer had to return the expected result on all three grading databases. This is a local self-evaluation with the course scorer.

## Clone and set up the environment

The reference run used Linux, Python 3.12.3 and four logical CPU cores. On a fresh Ubuntu 24.04 machine, install the system prerequisites once.

```bash
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-venv time
```

Clone the project, then run the remaining commands in the same Bash shell from the repository root.

```bash
git clone https://github.com/vectorBH6/HKU_DASE7506_MP3.git
cd HKU_DASE7506_MP3

unset PYTHONPATH
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/setup_env.sh --cpu
python -m mp3_sql.check_data
```

The setup script installs CPU PyTorch 2.7.1 and the pinned project dependencies, then runs `pip check`. The data check should report 480 training, 120 validation and 240 test tasks and verify the database hashes.

## Reproduce V12

Use the two release ZIPs with the official predictor and data in this checkout. If the ZIPs are distributed separately, save them as `dist/v12/source.zip` and `dist/v12/adapter.zip` first. Each run below writes into a fresh directory under `runs/reproduction/`. [Source ZIP](dist/v12/source.zip) · [LoRA ZIP](dist/v12/adapter.zip)

```bash
mkdir -p runs/reproduction
run_dir=$(mktemp -d "$PWD/runs/reproduction/v12.XXXXXXXX")
python -m zipfile -e dist/v12/source.zip "$run_dir/source"
python -m zipfile -e dist/v12/adapter.zip "$run_dir/adapter"
starter_dir="$PWD"
```

**Quick score check.** This scores the bundled predictions on all three database instances per task. It takes a few seconds and can run before the model download. The expected result is **191/240 (79.58%)**.

```bash
python "$run_dir/source/reproduce.py" --starter "$starter_dir" \
  --split test --score-only --output-dir "$run_dir/rescore"
```

**Generate fresh SQL.** Download the pinned `Qwen/Qwen3-0.6B` snapshot once while online. The runner then uses the local base and LoRA for offline CPU inference. For peer review against a separate copy of the official course starter, set `starter_dir` to its extracted root before running these commands.

```bash
python scripts/download_model.py
export MP3_BASE_MODEL="$PWD/artifacts/models/Qwen3-0.6B"
export MP3_ADAPTER_DIR="$run_dir/adapter"

python "$run_dir/source/reproduce.py" --starter "$starter_dir" \
  --split validation --output-dir "$run_dir/validation"
python "$run_dir/source/reproduce.py" --starter "$starter_dir" \
  --split test --output-dir "$run_dir/test"
```

The downloader pins revision `c1899de289a04d12100db370d81485cdf75e47ca`. The validation score is written to `$run_dir/validation/validation-score.json` and the test score to `$run_dir/test/test-score.json`. The reference prediction times were about 24 and 91 minutes, respectively.

### Original experiment workspace

With the original experiment runs, frozen checkpoints and training artifacts present, the historical runner can also generate both splits directly from this checkout.

```bash
.venv/bin/python -B scripts/run_v12_fresh.py --split validation --cpu-offset 0
.venv/bin/python -B scripts/run_v12_fresh.py --split test --cpu-offset 0
```

## How V12 works

The agent checks the original draft against the public example. It uses the question, schema, observed errors and shared repair hints to propose SQLite queries. A local Qwen3-0.6B model with a LoRA adapter works alongside a train-fitted candidate selector. When the query it is about to submit has already failed a public check, V12 spends any remaining calls on targeted text-condition repairs. It adopts a repair only after a successful public check.

## Results

| Official split | Correct | Accuracy | Charged tool calls | Model generations |
| --- | ---: | ---: | ---: | ---: |
| Validation | 99 / 120 | 82.50% | 561 | 275 |
| Public test | **191 / 240** | **79.58%** | 1,119 | 596 |

On the public test, V12 corrected two V9 answers and preserved all 189 that V9 got right. Prediction took 90.6 minutes on four CPU threads. Sampled process-tree memory peaked at 3.97 GiB, and GNU time recorded 4.88 GiB maximum RSS. Scoring took 6.5 seconds. The report includes the architecture, controlled comparisons and failure analysis. [Illustrated report (PDF)](reports/v12/FINAL_REPORT.pdf) · [Report (Markdown)](reports/v12/FINAL_REPORT.md) · [Test score](outputs/test-score.json) · [Test predictions](outputs/test.json) · [Experiment archive](reports/README.md)

## Course submission

Host both ZIPs at immutable HTTPS download URLs. On the course page, enter the student ID, select `outputs/test-score.json`, and paste the source and adapter URLs. Confirm the individual-work and privacy declaration, choose **Prepare score submission**, then **Submit on GitHub** and create the issue. The deadline is **30 November 2026 at 23:59 (UTC+8)**. [Course submission page](https://xudongwu-0.github.io/courses/dase7506/mp3/) · [Original English specification](README_en.md) · [Chinese specification](README_cn.md)
