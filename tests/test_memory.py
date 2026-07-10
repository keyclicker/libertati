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


def test_memory_write_emits_event(tmp_path):
    import json

    from libertati.observability import EventLogger
    from libertati.storage.memory import MemoryStore

    ev = EventLogger(path=tmp_path / "e.jsonl", enabled=True)
    mem = MemoryStore(tmp_path / "mem", events=ev)
    mem.append("todo.md", "- one")
    mem.append("todo.md", "- two")
    ev.close()
    rows = [json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    writes = [r for r in rows if r["type"] == "memory_write"]
    assert len(writes) == 2
    assert writes[0]["path"] == "todo.md"
    assert writes[0]["mode"] == "append"
    assert writes[1]["delta"] > 0


def test_max_file_chars_trims_oldest_keeping_header(tmp_path):
    from libertati.storage.memory import MemoryStore

    mem = MemoryStore(tmp_path / "mem", max_file_chars=120)
    mem.overwrite("world.md", "# Світ")
    for i in range(50):
        mem.append("world.md", f"- подія номер {i}")
    content = mem.read("world.md")
    assert len(content) <= 120
    assert content.startswith("# Світ")  # header preserved
    assert "подія номер 49" in content   # most recent kept
    assert "подія номер 0" not in content  # oldest trimmed away


def test_no_cap_keeps_everything(tmp_path):
    from libertati.storage.memory import MemoryStore

    mem = MemoryStore(tmp_path / "mem")  # no cap
    for i in range(200):
        mem.append("world.md", f"- line {i}")
    assert "line 0" in mem.read("world.md")
    assert "line 199" in mem.read("world.md")


def test_snapshot(memory):
    memory.overwrite("world.md", "news")
    snap = memory.snapshot()
    assert set(snap) == {"self.md", "world.md", "social.md", "todo.md", "reading.md"}
    assert snap["world.md"].strip() == "news"
