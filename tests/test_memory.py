from __future__ import annotations

import pytest

from libertati.storage.memory import slugify


def test_slugify():
    assert slugify("@Alice") == "alice"
    assert slugify("Cool Group!") == "cool-group"
    assert slugify("Тест") == "unknown"  # non-ascii stripped -> fallback


def test_read_write_overwrite(memory):
    assert memory.read("self.md") == ""
    memory.overwrite("self.md", "I am Ana")
    assert memory.read("self.md").strip() == "I am Ana"
    memory.overwrite("self.md", "new content")
    assert memory.read("self.md").strip() == "new content"


def test_append(memory):
    memory.append("todo.md", "- first")
    memory.append("todo.md", "- second")
    content = memory.read("todo.md")
    assert "first" in content and "second" in content
    assert content.index("first") < content.index("second")


def test_user_and_group_paths(memory):
    memory.append(memory.user_path("@Bob"), "- likes rust")
    assert "rust" in memory.read_user("@Bob")
    memory.append(memory.group_path("Fun Chat"), "- meme central")
    assert "meme" in memory.read_group("Fun Chat")
    assert "bob" in memory.list_users()


def test_path_traversal_rejected(memory):
    with pytest.raises(ValueError):
        memory.read("../secret.md")
    with pytest.raises(ValueError):
        memory.overwrite("/etc/passwd.md", "x")


def test_non_md_rejected(memory):
    with pytest.raises(ValueError):
        memory.read("self.txt")


def test_snapshot(memory):
    memory.overwrite("world.md", "news")
    snap = memory.snapshot()
    assert set(snap) == {"self.md", "world.md", "social.md", "todo.md"}
    assert snap["world.md"].strip() == "news"
