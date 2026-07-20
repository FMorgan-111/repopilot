from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_documents_installation_memory_and_selected_model():
    readme = _read("README.md")

    assert 'pip install "repopilot[memory]"' in readme
    assert 'pip install -e ".[memory,dev]"' in readme
    assert "REPOPILOT_ENABLE_EPISODES=1" in readme
    assert "gemini-3.5-flash:stable" in readme
    assert "NumPy" in readme


def test_env_example_uses_openai_compatible_gemini_defaults():
    example = _read(".env.example")

    assert "LLM_API_KEY=***" in example
    assert "OPENAI_BASE_URL=https://linoapi.com.cn/v1" in example
    assert "LLM_MODEL=gemini-3.5-flash:stable" in example


def test_contributor_guide_uses_package_metadata_instead_of_manual_dependency_fixes():
    guide = _read("CLAUDE.md")

    assert 'pip install -e ".[memory,dev]"' in guide
    assert "requirements.txt 未列全" not in guide
    assert "end_to_end" in guide
    assert "oracle_files" in guide


def test_progress_report_is_marked_historical_without_stale_current_claims():
    progress = _read("docs/PROGRESS_REPORT.md")

    assert "历史快照" in progress[:300]
    assert "252 个单测全绿" not in progress
    assert "生成产物默认不提交" in progress


def test_technical_adjustments_labels_seeded_result_as_oracle_files():
    adjustments = _read("docs/2026-07-05-technical-adjustments.md")

    assert "oracle_files" in adjustments
    assert "不能代表端到端定位成功率" in adjustments


def test_eval_report_does_not_claim_an_obsolete_hard_coded_model():
    report_source = _read("eval/report.py")

    assert "deepseek-v4-flash" not in report_source
