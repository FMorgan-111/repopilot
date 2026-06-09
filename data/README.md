# RepoPilot Dataset

> GitHub Issue → Fix paired dataset for training and evaluating code repair agents.
> Lives inside the [RepoPilot](https://github.com/FMorgan-111/repopilot) repo.

## Dataset Card

### Overview

~5,000 real-world bug fixes mined from 50 open-source Python repositories.
Each sample pairs a GitHub Issue (bug report) with the merged Pull Request's
unified diff. Used to train and evaluate RepoPilot's LangGraph agent.

### Format

JSONL — one JSON object per line. [Schema](schema.json)

### Quick Load

```python
import json

# Full dataset
with open("data/dataset-merged.jsonl") as f:
    for line in f:
        record = json.loads(line)
        print(record["id"], record["issue"]["title"])

# 50-record sample
with open("data/samples/issues_fixes.jsonl") as f:
    for line in f:
        record = json.loads(line)
        print(record["id"], record["issue"]["title"])
```

### Collection Methodology

- **Source**: GitHub REST API, authenticated
- **Repositories**: 50 popular Python projects (frameworks, libraries, tools, ML)
- **Filtering**: bug label, merged PR, 1-5 files changed, 5-300 lines, no bots, no docs/deps
- **Linking**: fixes/closes/resolves keywords + GitHub timeline cross-references

### Fields

| Field | Description |
|-------|-------------|
| `id` | `owner/repo#issue:pr` |
| `repo` | owner, name, stars, language |
| `issue` | number, url, title, body, labels, timestamps |
| `pr` | number, url, title, body, merged_at, linked_by |
| `patch` | full_diff, per-file patches with additions/deletions |
| `signals` | has_tests_changed, fix_size_bucket (small/medium/large) |
| `collected_at` | ISO8601 timestamp |

### Splits

Coming in v0.2 — repo-level train/valid/test split.

### Known Biases

- Python only (v0)
- English-language issues only
- GitHub-only (no GitLab/Bitbucket)
- Bias toward well-maintained, popular repositories

### License

The collection script is MIT. Individual diffs retain their original repository's license. See LICENSE for details.

### Citation

```
@dataset{repopilot-dataset,
  title = {RepoPilot Dataset: GitHub Issue → Fix Pairs},
  author = {RepoPilot Contributors},
  year = {2026},
  url = {https://github.com/FMorgan-111/repopilot/tree/main/data}
```

### Versions

| Version | Date | Records | Notes |
|---------|------|---------|-------|
| v0.1 | 2026-06 | ~5000 | Initial Python collection |
