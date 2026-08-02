"""Tests for the chat approval registry."""

from pathlib import Path

from libertati.chats import ChatRegistry


def make_registry(tmp_path: Path, enabled: bool = True) -> ChatRegistry:
    """Build a registry over a file in a temporary directory."""
    return ChatRegistry(tmp_path / "chats.toml", enabled)


def test_disabled_mode_allows_everything(tmp_path: Path) -> None:
    """With approval off, every chat passes and no file appears."""
    registry = make_registry(tmp_path, enabled=False)
    assert registry.register(100, "Alice (private)") is True
    assert registry.check(100) is True
    assert not registry.path.exists()


def test_new_chat_lands_unapproved(tmp_path: Path) -> None:
    """A first-seen chat is denied and appended as false with its name."""
    registry = make_registry(tmp_path)
    assert registry.register(-500, "friends (group)") is False
    content = registry.path.read_text(encoding="utf-8")
    assert content.startswith("# Chat approvals")
    assert "-500 = false  # friends (group)\n" in content


def test_register_does_not_duplicate(tmp_path: Path) -> None:
    """Re-registering a known chat leaves the file unchanged."""
    registry = make_registry(tmp_path)
    registry.register(100, "Alice (private)")
    before = registry.path.read_text(encoding="utf-8")
    assert registry.register(100, "Alice (private)") is False
    assert registry.path.read_text(encoding="utf-8") == before


def test_user_approval_applies_live(tmp_path: Path) -> None:
    """Flipping the value to true approves the chat on the next check."""
    registry = make_registry(tmp_path)
    registry.register(100, "Alice (private)")
    text = registry.path.read_text(encoding="utf-8")
    registry.path.write_text(text.replace("false", "true"), encoding="utf-8")
    assert registry.check(100) is True
    assert registry.register(100, "Alice (private)") is True


def test_check_is_read_only(tmp_path: Path) -> None:
    """check() never creates or appends to the file."""
    registry = make_registry(tmp_path)
    assert registry.check(100) is False
    assert not registry.path.exists()


def test_broken_file_denies_and_preserves(tmp_path: Path) -> None:
    """A file that fails to parse denies all chats and is not written to."""
    registry = make_registry(tmp_path)
    registry.path.write_text("100 = maybe???\n", encoding="utf-8")
    assert registry.check(100) is False
    assert registry.register(200, "Bob (private)") is False
    assert registry.path.read_text(encoding="utf-8") == "100 = maybe???\n"


def test_label_whitespace_is_collapsed(tmp_path: Path) -> None:
    """Newlines in a chat name can't break the TOML file."""
    registry = make_registry(tmp_path)
    registry.register(100, "evil\nname = true (private)")
    assert registry.check(100) is False
    assert "evil name = true (private)" in registry.path.read_text(encoding="utf-8")


def test_label_control_characters_are_dropped(tmp_path: Path) -> None:
    """Control characters in a chat name can't corrupt the TOML file.

    TOML forbids them even inside comments; a crafted name that made the
    file unparseable would deny every chat until repaired by hand.
    """
    registry = make_registry(tmp_path)
    registry.register(100, "evil\x00\x08name (private)")
    assert "evilname (private)" in registry.path.read_text(encoding="utf-8")
    # The file must still parse: registering another chat appends to it.
    assert registry.register(200, "Bob (private)") is False
    assert registry.check(100) is False
