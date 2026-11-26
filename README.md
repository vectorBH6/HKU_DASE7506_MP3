# DASE7506 Mini-project 3 · SQL Repair Agent

| Student | Student ID | Evaluated method |
| --- | --- | --- |
| Li Shanghang | 3036705979 | V17 SQL repair agent |

V17 scored **194/240 (80.83%) on the public test** and **102/120 (85.00%) on validation** in fresh four-core CPU runs. Each answer had to return the expected result on all three grading databases. This is a local self-evaluation with the course scorer.

[Read the four-page final report (PDF)](reports/v17/report.pdf)

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

## Reproduce V17

Use the two release ZIPs with the official predictor and data in this checkout. If the ZIPs are distributed separately, save them as `dist/v17/source.zip` and `dist/v12/adapter.zip` first. Each run below writes into a fresh directory under `runs/reproduction/`. [Source ZIP](dist/v17/source.zip) · [LoRA ZIP](dist/v12/adapter.zip)

```bash
mkdir -p runs/reproduction
run_dir=$(mktemp -d "$PWD/runs/reproduction/v17.XXXXXXXX")
python -m zipfile -e dist/v17/source.zip "$run_dir/source"
python -m zipfile -e dist/v12/adapter.zip "$run_dir/adapter"
starter_dir="$PWD"
```

**Quick score check.** This scores the bundled predictions on all three database instances per task. It takes a few seconds and can run before the model download. The expected result is **194/240 (80.83%)**.

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

The downloader pins revision `c1899de289a04d12100db370d81485cdf75e47ca`. The validation score is written to `$run_dir/validation/validation-score.json` and the test score to `$run_dir/test/test-score.json`. The reference prediction times were 26.3 and 56.4 minutes, respectively.

## How V17 works

The agent checks the draft against the public example, requests schema or training examples through the course tools, and proposes repairs with a local Qwen3-0.6B model and the V7 LoRA adapter. V17 adds a final review of what the SQL counts and which tables it needs. For example, a query for battles that lost a brig also joined a table of deaths. That join discarded ships without a recorded death. When the existing query failed its public check, V17 could remove the unused join while keeping the original battle, ship and ship-type conditions.

## Results

| Official split | Correct | Accuracy | Charged tool calls | Model generations |
| --- | ---: | ---: | ---: | ---: |
| Validation | 102 / 120 | 85.00% | 565 | 275 |
| Public test | **194 / 240** | **80.83%** | 1,121 | 596 |

V17 kept all 193 of V15's correct public-test answers and added one. V18 and V20 also scored 194 in paired comparisons while using additional model generations. V17's fresh prediction run took 56.4 minutes on four logical CPUs, with peak sampled process-tree RSS of 4.72 GiB. Official scoring took 6.7 seconds. [English report (PDF)](reports/v17/report.pdf) · [Official test score](dist/v17/test-score.json) · [Measured results](reports/v17/results.json) · [V17 controller](submission/v17_agent.py) · [Version comparison archive](reports/README.md)

## Course submission

Host both ZIPs at immutable HTTPS download URLs. On the course page, enter the student ID, select the V17 score at `dist/v17/test-score.json` (also `outputs/test-score.json` inside the V17 source ZIP), and paste the source and adapter URLs. Confirm the individual-work and privacy declaration, choose **Prepare score submission**, then **Submit on GitHub** and create the issue. The deadline is **30 November 2026 at 23:59 (UTC+8)**. [Course submission page](https://xudongwu-0.github.io/courses/dase7506/mp3/) · [Original English specification](README_en.md) · [Chinese specification](README_cn.md)
