from __future__ import annotations

import subprocess

from src.local_search import search_local_checkout


def test_search_local_checkout_reads_only_tracked_source_files(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "def login():\n    raise ValueError('login')\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "login ValueError",
        encoding="utf-8",
    )
    (tmp_path / "secret.py").write_text(
        "login ValueError",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "src/auth.py", "README.md"],
        check=True,
    )

    results = search_local_checkout(str(tmp_path), ["login", "ValueError"])

    assert [item["path"] for item in results] == ["src/auth.py"]
    assert "raise ValueError" in results[0]["content"]


def test_search_local_checkout_skips_oversized_and_binary_files(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "large.py").write_text("login\n" * 100, encoding="utf-8")
    (tmp_path / "binary.py").write_bytes(b"login\xff\x00")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "large.py", "binary.py"],
        check=True,
    )

    results = search_local_checkout(
        str(tmp_path),
        ["login"],
        max_file_bytes=50,
    )

    assert results == []
