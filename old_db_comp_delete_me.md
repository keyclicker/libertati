# Old-database compatibility removed

The move to SQLAlchemy + Alembic dropped every mechanism that patched an
old database up to the current schema on the fly. A database created
before this change has no `alembic_version` table, so the startup
`upgrade head` will try to create tables that already exist and fail.

**Escape hatch for a database whose schema is already current:** stamp
it once and it joins the migration chain —

```sh
uv run alembic stamp head
```

(with `alembic.ini`'s `sqlalchemy.url` pointing at the file). For
anything older, delete the file and let the bot recreate it.

Stamping is only safe for a database *born* current. One whose schema
looks current because `_ensure_column` patched it on the way is exactly
the case removals 2 and 3 below describe: its old rows still carry a
NULL `message_thread_id` and reply links into topic-creation service
messages, and after the stamp nothing ever repairs them.

## What was removed

1. **`Database._ensure_column`** and its five column adds:
   `api_usage.turn_id`, `api_usage.input_context_id`,
   `api_usage.dream_id`, `messages.message_thread_id`,
   `messages.media_uid`. All five columns are part of the initial
   Alembic revision now; a pre-Alembic database missing them is never
   patched again.
2. **Connect-time backfill of `messages.message_thread_id`** from
   `raw -> '$.is_topic_message'` for rows saved before the column
   existed. Old rows keep a NULL topic id.
3. **Connect-time NULL-ing of forum pseudo-replies**
   (`reply_to_message_id` pointing at a topic-creation service message)
   on rows saved before `effective_reply_to` filtered them at write
   time. Old rows keep the bogus link and would pollute reply threads.
4. **`idx_messages_chat_thread` created outside `SCHEMA`** (it named a
   column the old migration added, so `SCHEMA` could not carry it). It
   lives in the initial revision now.
5. **Spy fallback for pre-dream usage tables**: `fetch_usage` retried
   without the `dream_id` column when the query failed with
   `OperationalError` on a database written before dreams existed. The
   spy now assumes the current schema.
6. **Spy `OperationalError` guards** in `newest_id`, `latest_dream` and
   `latest_recorded_dream`, which turned a missing `dreams` /
   `dream_context` table into an empty view instead of a crash. Same
   assumption now: the schema is current or the viewer fails loudly.
7. **Deleted tests** that pinned the above:
   `test_usage_schema_migrates_existing_table`,
   `test_messages_schema_migrates_and_backfills`,
   `test_usage_survives_a_database_without_the_dream_column`,
   `test_dream_lookups_tolerate_a_database_without_dreams`.
