# RepoPilot Eval Summary
**Date**: 1780976150.1410263
**Samples evaluated**: 5
**Errors**: 3 (Textualize/textual#4705:4720, ansible/ansible#86228:86302, ansible/ansible#86395:86403)

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| file_recall@1 (mean) | 0.000 |
| file_recall@1 (median) | 0.000 |
| file_recall@3 (mean) | 0.000 |
| file_recall@3 (median) | 0.000 |
| file_recall@5 (mean) | 0.000 |
| file_recall@5 (median) | 0.000 |
| patch_apply_rate | 0.000 |
| test_pass_rate | N/A (no test-relevant samples) |
| avg cost per sample | $0.003166 |
| total cost | $0.006332 |
| total input tokens | 6,786 |
| total output tokens | 12,500 |

## Per-Sample Results

| # | Sample ID | file_recall@1 | file_recall@3 | file_recall@5 | patch_apply | test_pass | cost |
|---|-----------|--------------|--------------|--------------|-------------|-----------|------|
| 1 | `tox-dev/tox#3075:3748` | 0.00 | 0.00 | 0.00 | ✗ | N/A | $0.004579 |
| 2 | `Textualize/textual#4705:4720` | — | — | — | — | — | error: clone_failed |
| 3 | `python/mypy#20532:20643` | 0.00 | 0.00 | 0.00 | ✗ | N/A | $0.001753 |
| 4 | `ansible/ansible#86228:86302` | — | — | — | — | — | error: clone_failed |
| 5 | `ansible/ansible#86395:86403` | — | — | — | — | — | error: clone_failed |

## Notes

- **file_recall@k**: fraction of actual changed files found in agent's top-k predictions
- **patch_apply_rate**: fraction of agent-generated patches that cleanly apply with `git apply`
- **test_pass_rate**: fraction of applied patches where `pytest` passes (only for `has_tests_changed=true` samples)
- **cost**: estimated DeepSeek API cost ($0.27/M input, $0.36/M output)
- **model**: deepseek-v4-flash (fallback during peak hours)
