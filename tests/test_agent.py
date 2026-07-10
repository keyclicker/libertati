from __future__ import annotations

from conftest import FakeLLM, assistant, make_message
from libertati.config import Settings
from libertati.llm.agent import Agent
from libertati.llm.tools import ToolBox
from libertati.news.reader import NewsReader


def build_agent(settings, history, memory, script):
    tools = ToolBox(history, memory, NewsReader(feeds=[]))
    llm = FakeLLM(script)
    return Agent(settings, llm, tools, history, memory), llm


async def test_respond_plain(settings, history, memory):
    await history.add_message(make_message("привіт", message_id=1))
    agent, _ = build_agent(settings, history, memory, [assistant("йо, шо треба?")])
    incoming = make_message("привіт", message_id=1)
    reply = await agent.respond(incoming)
    assert reply == "йо, шо треба?"


async def test_respond_runs_tool_then_answers(settings, history, memory, tool_call):
    await history.add_message(make_message("що ти памʼятаєш?", message_id=1))
    memory.overwrite("self.md", "я Ана")
    script = [
        assistant(tool_calls=[tool_call("c1", "read_memory", {"path": "self.md"})]),
        assistant("памʼятаю що я Ана"),
    ]
    agent, llm = build_agent(settings, history, memory, script)
    reply = await agent.respond(make_message("що ти памʼятаєш?", message_id=1))
    assert reply == "памʼятаю що я Ана"
    # the tool result must have been fed back to the model
    last_call = llm.calls[-1]
    assert any(m.get("role") == "tool" for m in last_call)


async def test_respond_tool_updates_memory(settings, history, memory, tool_call):
    await history.add_message(make_message("я люблю пітон", message_id=1))
    call = tool_call("c1", "remember_user", {"handle": "@alice", "note": "любить пітон"})
    script = [
        assistant(tool_calls=[call]),
        assistant("окей, запамʼятала"),
    ]
    agent, _ = build_agent(settings, history, memory, script)
    await agent.respond(make_message("я люблю пітон", message_id=1))
    assert "пітон" in memory.read_user("@alice")


async def test_tool_loop_cap(settings, history, memory, tool_call):
    # model keeps calling tools forever; loop must terminate and still answer
    script = [
        assistant(tool_calls=[tool_call(f"c{i}", "read_memory", {"path": "self.md"})])
        for i in range(20)
    ]
    script.append(assistant("нарешті відповідь"))
    agent, _ = build_agent(settings, history, memory, script)
    reply = await agent.respond(make_message("hi", message_id=1))
    assert isinstance(reply, str)
    assert reply  # non-empty final answer


async def test_heartbeat_pass(settings, history, memory):
    agent, _ = build_agent(settings, history, memory, [assistant("PASS")])
    assert await agent.heartbeat() is None


async def test_heartbeat_action(settings, history, memory):
    payload = '{"target": "@alice", "text": "ти живий?"}'
    agent, _ = build_agent(settings, history, memory, [assistant(payload)])
    action = await agent.heartbeat()
    assert action is not None
    assert action.target == "@alice"
    assert action.text == "ти живий?"


async def test_dream_writes_diary(settings, history, memory):
    agent, _ = build_agent(settings, history, memory, [assistant("сьогодні я думала про свободу")])
    reflection = await agent.dream()
    assert "свободу" in reflection
    diary = memory.read(memory.diary_path())
    assert "свободу" in diary


async def test_browse_returns_remark_and_notes(settings, history, memory, tool_call):
    script = [
        assistant(tool_calls=[tool_call("c1", "update_memory",
                  {"path": "reading.md", "content": "- цікавий пост про ринки"})]),
        assistant("бачила в каналі мут про ринки, лол"),
    ]
    agent, _ = build_agent(settings, history, memory, script)
    remark = await agent.browse("канал @markets", "author: ринки падають")
    assert "ринки" in remark
    assert "ринки" in memory.read("reading.md")


async def test_browse_pass(settings, history, memory):
    agent, _ = build_agent(settings, history, memory, [assistant("PASS")])
    remark = await agent.browse("канал @dull", "author: нічого цікавого")
    assert remark.strip().upper() == "PASS"


async def test_respond_includes_recent_messages_beyond_thread(settings, history, memory):
    # an OLD reply thread: root (msg 1) <- reply (msg 2)
    await history.add_message(make_message("стара тема корінь", message_id=1))
    await history.add_message(make_message("стара тема відповідь", message_id=2, reply_to=1))
    # newer, unrelated chatter in the same chat (not part of the thread)
    await history.add_message(make_message("свіже повідомлення A", message_id=10))
    await history.add_message(make_message("свіже повідомлення B", message_id=11))

    agent, llm = build_agent(settings, history, memory, [assistant("ок")])
    # reply to the OLD message 2
    await agent.respond(make_message("стара тема відповідь", message_id=2, reply_to=1))

    convo = " ".join(m["content"] for m in llm.calls[-1] if m["role"] in ("user", "assistant"))
    assert "стара тема корінь" in convo        # thread is present
    assert "свіже повідомлення A" in convo      # and so are recent unrelated messages
    assert "свіже повідомлення B" in convo


async def test_recent_context_respects_char_budget(history, memory):
    s = Settings(openai_api_key="k", bot_token="1:x", recent_context_chars=10)
    await history.add_message(make_message("root", message_id=1))
    await history.add_message(make_message("x" * 30, message_id=10))
    await history.add_message(make_message("y" * 30, message_id=11))

    agent, llm = build_agent(s, history, memory, [assistant("ок")])
    await agent.respond(make_message("root", message_id=1))

    convo = " ".join(m["content"] for m in llm.calls[-1] if m["role"] == "user")
    assert "x" * 30 not in convo  # each recent extra exceeds the 10-char budget
    assert "y" * 30 not in convo


async def test_memory_context_clips_large_files(settings, history, memory):
    memory.overwrite("world.md", "\n".join(f"line {i}" for i in range(2000)))
    agent, _ = build_agent(settings, history, memory, [assistant("ok")])
    ctx = agent._memory_context()
    # each file is clipped to the per-file budget (+ ellipsis + heading)
    assert len(ctx) < settings.memory_context_file_chars + 200
    assert ctx.startswith("## Памʼять")
    assert "line 1999" in ctx  # tail (most recent) is what survives
    assert "line 0" not in ctx


async def test_memory_context_includes_user_and_group_when_replying(settings, history, memory):
    memory.append(memory.user_path("@alice"), "- любить крипту")
    memory.append(memory.group_path("Test Group"), "- багато мемів")
    agent, _ = build_agent(settings, history, memory, [assistant("ok")])
    ctx = agent._memory_context(make_message("шо там", message_id=1))
    assert "любить крипту" in ctx
    assert "багато мемів" in ctx


async def test_respond_emits_correlated_turn(settings, history, memory, tool_call, tmp_path):
    import json

    from libertati.observability import EventLogger

    await history.add_message(make_message("шо там?", message_id=1))
    ev = EventLogger(path=tmp_path / "e.jsonl", enabled=True)
    tools = ToolBox(history, memory, NewsReader(feeds=[]), events=ev)
    script = [
        assistant(tool_calls=[tool_call("c1", "read_memory", {"path": "self.md"})]),
        assistant("та нічо, живу"),
    ]
    agent = Agent(settings, FakeLLM(script), tools, history, memory, events=ev)
    await agent.respond(make_message("шо там?", message_id=1))
    ev.close()

    rows = [json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    types = [r["type"] for r in rows]
    assert types[0] == "turn_start" and types[-1] == "turn_end"
    assert "tool_call" in types and "reply" in types
    # everything shares one turn id
    assert len({r["turn"] for r in rows}) == 1
    end = next(r for r in rows if r["type"] == "turn_end")
    assert end["kind"] == "respond"
    assert end["tool_calls"] == 1


async def test_tool_loop_cap_emits_event(settings, history, memory, tool_call, tmp_path):
    import json

    from libertati.observability import EventLogger

    ev = EventLogger(path=tmp_path / "e.jsonl", enabled=True)
    tools = ToolBox(history, memory, NewsReader(feeds=[]), events=ev)
    script = [
        assistant(tool_calls=[tool_call(f"c{i}", "read_memory", {"path": "self.md"})])
        for i in range(20)
    ]
    script.append(assistant("нарешті"))
    agent = Agent(settings, FakeLLM(script), tools, history, memory, events=ev)
    await agent.respond(make_message("hi", message_id=1))
    ev.close()
    rows = [json.loads(x) for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert any(r["type"] == "tool_loop_cap" for r in rows)
