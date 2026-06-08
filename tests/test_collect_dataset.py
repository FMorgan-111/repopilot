from scripts.collect_dataset import (
    build_dataset_record,
    extract_closing_issue_numbers,
    fix_size_bucket,
    has_excluded_labels,
    has_tests_changed,
    is_bot_actor,
    is_generated_or_vendor_path,
    is_meaningful_issue,
    should_keep_pr_files,
)


def test_extract_closing_issue_numbers_from_pr_body():
    body = "Fixes #12, closes owner/repo#34 and resolves https://github.com/a/b/issues/56"

    assert extract_closing_issue_numbers(body) == {12, 34, 56}


def test_filter_helpers_reject_unwanted_inputs():
    assert has_excluded_labels(["bug", "dependencies"]) is True
    assert has_excluded_labels(["bug"]) is False
    assert is_bot_actor("dependabot[bot]") is True
    assert is_bot_actor("alice") is False
    assert is_generated_or_vendor_path("vendor/pkg/module.py") is True
    assert is_generated_or_vendor_path("src/pkg/module.py") is False
    assert is_meaningful_issue({"title": "Crash", "body": "too short"}) is False
    assert is_meaningful_issue({"title": "Crash", "body": "A real failure report with traceback details."}) is True


def test_should_keep_pr_files_enforces_file_and_line_bounds():
    files = [
        {"filename": "src/app.py", "status": "modified", "additions": 5, "deletions": 1, "patch": "@@"},
        {"filename": "tests/test_app.py", "status": "modified", "additions": 6, "deletions": 1, "patch": "@@"},
    ]

    assert should_keep_pr_files(files) is True
    assert has_tests_changed(files) is True
    assert fix_size_bucket(files) == "small"

    lock_only = [{"filename": "poetry.lock", "status": "modified", "additions": 20, "deletions": 5}]
    assert should_keep_pr_files(lock_only) is False

    too_large = [{"filename": "src/app.py", "status": "modified", "additions": 301, "deletions": 0}]
    assert should_keep_pr_files(too_large) is False


def test_build_dataset_record_shape():
    repo = {"full_name": "owner/project", "stargazers_count": 42, "language": "Python"}
    issue = {
        "number": 7,
        "html_url": "https://github.com/owner/project/issues/7",
        "title": "Crash on login",
        "body": "A real failure report with traceback details.",
        "labels": [{"name": "bug"}],
        "created_at": "2026-01-01T00:00:00Z",
        "closed_at": "2026-01-02T00:00:00Z",
    }
    pr = {
        "number": 8,
        "html_url": "https://github.com/owner/project/pull/8",
        "title": "Fix login crash",
        "body": "Fixes #7",
        "merged_at": "2026-01-03T00:00:00Z",
    }
    files = [
        {"filename": "src/app.py", "status": "modified", "additions": 5, "deletions": 1, "patch": "@@"}
    ]

    record = build_dataset_record(repo, issue, pr, "diff --git a/src/app.py b/src/app.py", files, "fixes_keyword")

    assert record["id"] == "owner/project#7:8"
    assert record["repo"]["stars"] == 42
    assert record["issue"]["labels"] == ["bug"]
    assert record["pr"]["linked_by"] == "fixes_keyword"
    assert record["patch"]["files"][0]["path"] == "src/app.py"
    assert record["signals"]["fix_size_bucket"] == "small"
