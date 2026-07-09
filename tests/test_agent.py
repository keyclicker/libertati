from __future__ import annotations

from conftest import FakeLLM, assistant, make_message
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
