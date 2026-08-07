"""SQLAlchemy Core metadata for the bot's SQLite database.

The single source of truth for the schema: the async persistence layer
(`db.py`), the spy's sync engines and Alembic autogenerate all import
these tables. Timestamps are TEXT in SQLite's ``datetime('now')`` format
(``YYYY-MM-DD HH:MM:SS``, UTC) except ``messages.date``/``edit_date``,
which hold full ISO-8601 for parity with the Telegram API. Booleans are
INTEGER 0/1. Must not import aiogram or openai — the spy loads it at
startup.
"""

from sqlalchemy import Column, ForeignKey, Index, Integer, MetaData, Table, Text, text

metadata = MetaData()

# SQLite fills TEXT timestamp columns with this on insert; the exact
# "YYYY-MM-DD HH:MM:SS" shape is load-bearing (lexicographic compares in
# dream queries, the spy's stamp parser).
_NOW = text("(datetime('now'))")

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("is_bot", Integer, nullable=False, server_default=text("0")),
    Column("username", Text),
    Column("first_name", Text),
    Column("last_name", Text),
    Column("language_code", Text),
    Column("raw", Text, nullable=False),
    Column("first_seen_at", Text, nullable=False, server_default=_NOW),
    Column("last_seen_at", Text, nullable=False, server_default=_NOW),
)

chats = Table(
    "chats",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("type", Text, nullable=False),
    Column("title", Text),
    Column("username", Text),
    Column("raw", Text, nullable=False),
    Column("first_seen_at", Text, nullable=False, server_default=_NOW),
    Column("last_seen_at", Text, nullable=False, server_default=_NOW),
)

messages = Table(
    "messages",
    metadata,
    Column("chat_id", Integer, ForeignKey("chats.id"), primary_key=True),
    Column("message_id", Integer, primary_key=True, autoincrement=False),
    Column("from_user_id", Integer, ForeignKey("users.id")),
    # Full ISO datetime; named "date" for parity with the Telegram API field.
    Column("date", Text, nullable=False),
    Column("edit_date", Text),
    Column("content_type", Text, nullable=False),
    Column("text", Text),
    Column("caption", Text),
    Column("reply_to_message_id", Integer),
    # Forum topic id; NULL outside forum topics (incl. the General topic).
    Column("message_thread_id", Integer),
    Column("media_group_id", Text),
    # Filled when a message's media is described, not when it is saved:
    # it is the join key to media_notes, and a message whose file nobody
    # has looked at has no note to join to.
    Column("media_uid", Text),
    Column("outgoing", Integer, nullable=False, server_default=text("0")),
    Column("raw", Text, nullable=False),
    Column("saved_at", Text, nullable=False, server_default=_NOW),
    Index("idx_messages_chat_date", "chat_id", "date"),
    Index("idx_messages_from_user", "from_user_id"),
    # Covers the per-topic lookups (topic_observed, topic_name,
    # list_topics, topic-scoped history), which otherwise walk a chat by
    # date.
    Index("idx_messages_chat_thread", "chat_id", "message_thread_id", "date"),
)

# What a media file turned out to depict, in words. Keyed by Telegram's
# file_unique_id rather than by message: the same sticker or forwarded
# photo appears in many chats and is worth describing exactly once.
media_notes = Table(
    "media_notes",
    metadata,
    Column("file_unique_id", Text, primary_key=True),
    Column("kind", Text, nullable=False),
    Column("note", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("created_at", Text, nullable=False, server_default=_NOW),
)

# Last message exposed through get_recent_messages, per chat/topic.  Zero
# represents a whole-chat cursor; Telegram topic ids are positive.
message_read_cursors = Table(
    "message_read_cursors",
    metadata,
    Column("chat_id", Integer, ForeignKey("chats.id"), primary_key=True),
    Column(
        "message_thread_id",
        Integer,
        primary_key=True,
        autoincrement=False,
        server_default=text("0"),
    ),
    Column("message_id", Integer, nullable=False),
)

context = Table(
    "context",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("item", Text, nullable=False),
    Column("created_at", Text, nullable=False, server_default=_NOW),
    sqlite_autoincrement=True,
)

agent_turns = Table(
    "agent_turns",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("start_context_id", Integer, nullable=False),
    Column("end_context_id", Integer),
    Column("status", Text, nullable=False, server_default=text("'running'")),
    Column("started_at", Text, nullable=False, server_default=_NOW),
    Column("finished_at", Text),
    sqlite_autoincrement=True,
)

api_usage = Table(
    "api_usage",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("response_id", Text),
    Column("turn_id", Integer),
    Column("dream_id", Integer),
    Column("input_context_id", Integer, nullable=False, server_default=text("0")),
    Column("model", Text, nullable=False),
    Column("input_tokens", Integer, nullable=False),
    Column("cached_tokens", Integer, nullable=False),
    Column("cache_write_tokens", Integer, nullable=False),
    Column("output_tokens", Integer, nullable=False),
    Column("reasoning_tokens", Integer, nullable=False),
    Column("total_tokens", Integer, nullable=False),
    Column("created_at", Text, nullable=False, server_default=_NOW),
    sqlite_autoincrement=True,
)

# One row per dream: its budget has to survive restarts, and a dream
# that dies before writing its journal entry still has to count against
# that budget.
dreams = Table(
    "dreams",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("trigger", Text, nullable=False),
    Column("status", Text, nullable=False, server_default=text("'running'")),
    Column("steps", Integer, nullable=False, server_default=text("0")),
    Column("summary", Text),
    Column("started_at", Text, nullable=False, server_default=_NOW),
    Column("finished_at", Text),
    Index("idx_dreams_started", "started_at"),
    sqlite_autoincrement=True,
)

# A dream's context, kept apart from the waking one: it is written for
# inspection only and never read back, so nothing here can leak into
# what the waking agent is sent.
dream_context = Table(
    "dream_context",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("dream_id", Integer, ForeignKey("dreams.id"), nullable=False),
    Column("item", Text, nullable=False),
    Column("created_at", Text, nullable=False, server_default=_NOW),
    Index("idx_dream_context_dream", "dream_id", "id"),
    sqlite_autoincrement=True,
)

wakeups = Table(
    "wakeups",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("due_at", Text, nullable=False),
    Column("note", Text, nullable=False),
    Column("done", Integer, nullable=False, server_default=text("0")),
    Column("created_at", Text, nullable=False, server_default=_NOW),
    Index("idx_wakeups_due", "done", "due_at"),
    sqlite_autoincrement=True,
)

# Instructions typed at the operator console (the spy TUI) and waiting
# to be handed to the agent as events. The console is a separate
# process that shares nothing with the bot but this file, so the table
# is the channel; ``urgent`` picks between waiting for the running turn
# to end and landing between its rounds.
steering = Table(
    "steering",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("text", Text, nullable=False),
    Column("urgent", Integer, nullable=False, server_default=text("0")),
    Column("done", Integer, nullable=False, server_default=text("0")),
    Column("created_at", Text, nullable=False, server_default=_NOW),
    Index("idx_steering_pending", "done", "id"),
    sqlite_autoincrement=True,
)
