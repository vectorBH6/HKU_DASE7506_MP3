# V13 foreign-key path repair

This development probe tests whether adding a foreign-key path and rebinding a column together can recover SQL that the existing candidates miss. The official training split supplied the schemas and SQL used to develop the edits. [Probe implementation](../../scripts/probe_v13_fk_paths.py)

| Comparison | Result | Evidence |
| --- | --- | --- |
| Fixed tool budget on 98 held-out training questions | 75 to 76 correct | [Training and holdout results](../../runs/v13_fk_paths/train_holdout.json) |
| Apply as the final V12 repair | 99/120 before and after | [Validation results](../../runs/v13_fk_paths/validation.json) |
| Try the first candidate earlier | 99/120 before and after | [Timing comparison](../../runs/v13_fk_paths/early_validation.json) |

Evaluation stopped after validation. V12 has the highest recorded public-test score. [V12 report](../v12/FINAL_REPORT.md)
