"""Chat approval registry backed by a user-editable TOML file.

Two modes, chosen by the ``chat_approval`` setting:

- Off (default): every chat is allowed; the file is never touched.
- On: only chats marked ``true`` in the file reach the agent. Any chat
  seen for the first time is appended as ``false`` (with a name
  comment), so approving is just flipping the value to ``true``. The
  file is re-read on every check — edits apply live, no restart.
"""

import logging
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)

#: Written once when the file is first created.
HEADER = """\
# Chat approvals: flip a chat to true to let the agent see it.
# New chats are appended as false automatically. Edits apply live.
"""


class ChatRegistry:
    """Decides which chats the agent is allowed to see.

    The file is flat TOML — one ``chat_id = bool`` per line (chat ids,
    including negative group ids, are valid bare keys). A file that
    fails to parse denies everything and is never appended to, so a
    user typo can't be silently clobbered.
    """

    def __init__(self, path: Path, enabled: bool) -> None:
        """Remember the file location and whether approval mode is on."""
        self.path = path
        self.enabled = enabled

    def _load(self) -> dict[str, object] | None:
        """Parse the approvals file; {} when absent, None when broken."""
        if not self.path.exists():
            return {}
        try:
            return tomllib.loads(self.path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            log.error("cannot parse %s: %s — denying all chats", self.path, exc)
            return None

    def check(self, chat_id: int) -> bool:
        """Is the chat approved? Read-only; unknown chats are not."""
        if not self.enabled:
            return True
        approvals = self._load()
        if approvals is None:
            return False
        return approvals.get(str(chat_id)) is True

    def register(self, chat_id: int, label: str) -> bool:
        """Check approval, appending unknown chats as unapproved."""
        if not self.enabled:
            return True
        approvals = self._load()
        if approvals is None:
            return False
        if str(chat_id) in approvals:
            return approvals[str(chat_id)] is True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = "" if self.path.exists() else HEADER
        # The label is attacker-controlled (chat title / sender name).
        # Collapsing whitespace kills line injection; dropping the
        # remaining unprintable characters keeps a crafted name from
        # rendering the whole TOML file unparseable — which would deny
        # every chat until the user repairs it by hand.
        comment = "".join(
            char for char in " ".join(label.split()) if char.isprintable()
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"{header}{chat_id} = false  # {comment}\n")
        log.info("new chat %s (%s) awaiting approval in %s", chat_id, label, self.path)
        return False
