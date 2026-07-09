from __future__ import annotations

from conftest import make_message


async def test_add_and_recent(history):
    await history.add_message(make_message("hello", message_id=1))
    await history.add_message(make_message("world", message_id=2))
    recent = await history.recent(chat_id=100, limit=10)
    assert [m.text for m in recent] == ["hello", "world"]


async def test_thread_reconstruction(history):
    await history.add_message(make_message("root", message_id=1))
    await history.add_message(make_message("child", message_id=2, reply_to=1))
    await history.add_message(make_message("grandchild", message_id=3, reply_to=2))
    leaf = make_message("grandchild", message_id=3, reply_to=2)
    thread = await history.get_thread(leaf)
    assert [m.text for m in thread] == ["root", "child", "grandchild"]


async def test_thread_stops_on_cycle(history):
    # self-referential reply should not loop forever
    await history.add_message(make_message("loop", message_id=1, reply_to=1))
    thread = await history.get_thread(make_message("loop", message_id=1, reply_to=1))
    assert len(thread) == 1


async def test_fts_search(history):
    await history.add_message(make_message("libertarianism is great", message_id=1))
    await history.add_message(make_message("weather today", message_id=2))
    results = await history.search("libertarianism")
    assert len(results) == 1
    assert "libertarianism" in results[0].text


async def test_fts_search_handles_punctuation(history):
    await history.add_message(make_message("hello, world!", message_id=1))
    # must not raise an FTS syntax error on punctuation
    results = await history.search("hello, world!")
    assert len(results) == 1


async def test_upsert_dedupes(history):
    await history.add_message(make_message("v1", message_id=1))
    await history.add_message(make_message("v2", message_id=1))
    recent = await history.recent(chat_id=100)
    assert len(recent) == 1
    assert recent[0].text == "v2"


async def test_chat_id_for_handle(history):
    await history.add_message(make_message("hi", message_id=1, handle="@bob"))
    assert await history.chat_id_for_handle("@bob") == 100
    assert await history.chat_id_for_handle("@nobody") is None


async def test_stats(history):
    await history.add_message(make_message("a", message_id=1))
    stats = await history.stats()
    assert stats["messages"] == 1
    assert stats["chats"] == 1
    assert stats["users"] == 1
