from __future__ import annotations

import subprocess

from src.local_search import (
    find_local_references,
    list_local_related_tests,
    read_local_range,
    read_local_symbol,
    search_local_checkout,
    search_local_symbol,
    search_local_text,
)


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


def test_bounded_local_source_helpers_use_tracked_files(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "widget.py").write_text(
        "class Widget:\n    def render(self):\n        return 1\n\ndef caller():\n    return Widget().render()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_widget.py").write_text(
        "from src.widget import Widget\ndef test_render():\n    assert Widget().render() == 1\n",
        encoding="utf-8",
    )
    (tmp_path / "untracked.py").write_text("Widget", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src", "tests"], check=True)

    assert "src/widget.py" in search_local_symbol(str(tmp_path), "Widget")
    assert "return 1" in search_local_text(str(tmp_path), "return 1")
    assert "def render" in read_local_symbol(str(tmp_path), "src/widget.py", "Widget.render")
    assert read_local_range(str(tmp_path), "src/widget.py", 1, 2).startswith("1: class Widget")
    references = find_local_references(str(tmp_path), "Widget")
    assert "tests/test_widget.py" in references
    assert "untracked.py" not in references
    assert list_local_related_tests(str(tmp_path), "src/widget.py") == ["tests/test_widget.py"]


def test_read_local_symbol_supports_non_python_source(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "widget.js").write_text(
        "export function renderWidget() {\n  return 'ok';\n}\n\nconst ignored = 1;\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "widget.js"], check=True)

    result = read_local_symbol(str(tmp_path), "widget.js", "renderWidget")

    assert "function renderWidget" in result
    assert "return 'ok'" in result
    assert "ignored" not in result
