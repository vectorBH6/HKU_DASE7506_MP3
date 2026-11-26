# Experiment reports

Li Shanghang's V17 agent scored **194/240 (80.83%) on public test** and **102/120 (85.00%) on validation** in fresh four-core CPU runs. The report covers the agent loop, controlled ablations, resource use and reproduction. [Final V17 report (PDF)](v17/report.pdf) · [V17 measurements](v17/results.json) · [Previous V12 report](v12/FINAL_REPORT.md)

| Version | Experiment | Report and primary evidence |
| --- | --- | --- |
| [V1](v1/) | Baselines, supervised fine-tuning and feedback | [Report](v1/report.md) · [Results](v1/results.md) |
| [V2](v2/) | On-policy training and candidate review | [Report](v2/v2_report.md) · [Results](v2/v2_results.md) |
| [V3](v3/) | Recovery after a rejected action | [Report](v3/v3_report.md) · [Results](v3/v3_results.md) |
| [V4](v4/) | Training on execution feedback and correcting structural constraints | [Report](v4/v4_report.md) · [Results](v4/v4_results.md) |
| [V4b](v4b/) | Comparison of fallback and constraint policies | [Report](v4b/v4b_report.md) · [Results](v4b/v4b_results.json) |
| [V5](v5/) | Candidate coverage, search and GRPO | [Report](v5/v5_report.md) · [Results](v5/v5_results.json) |
| [V6](v6/) | Generating and selecting SQL repairs | [Report](v6/v6_report.md) · [Selection](v6/v6_selection.json) |
| [V7](v7/) | Paired training data for SQL correction | [Report](v7/v7_report.md) · [Results](v7/v7_official_results.json) |
| [V8](v8/) | Structured repair and comparison of checked candidates | [Report](v8/v8_report.md) · [Results](v8/v8_pool_results.json) |
| [V9](v9/) | Question cues and output projection | [Development probes](v9/v9_probe_report.md) · [Official runs](v9/v9_official_report.md) |
| [V10](v10/) | V4 proposals with V8 review | [Comparison report](comparisons/v10_v12_report.md) · [Validation](v10/v10_official_validation.json) · [Test](v10/v10_official_test.json) |
| [V11](v11/) | String-predicate repair after a failed V10 check | [Comparison report](comparisons/v10_v12_report.md) · [Validation](v11/v11_official_validation.json) · [Test](v11/v11_official_test.json) |
| [V12](v12/) | String-predicate repair after a failed V9 check | [Final English report](v12/FINAL_REPORT.md) · [Coverage of validation candidates](v12/v12_validation_coverage.json) |
| [V13 development probe](v13/) | Foreign-key path repair | [Training and validation evidence](v13/README.md) |
| [V17](v17/report.pdf) | Join cardinality and surplus tables | [Final report (PDF)](v17/report.pdf) · [Fresh CPU results](v17/results.json) |

[Cross-version comparisons](comparisons/) include the V10–V12 experiment report, the V10/V11 coverage audit, the teacher-distillation study and the shared V1/V2 reproduction logs.

## Running historical scripts

Some archived experiment scripts and checksum manifests use the original report paths. To run those scripts, prepare a separate working copy with the historical layout. From the project root, choose a new sibling directory and run the following commands. The current V12 reproduction commands work directly in the main checkout.

```bash
bash scripts/prepare_report_workspace.sh ../mp3-history
cd ../mp3-history
```
