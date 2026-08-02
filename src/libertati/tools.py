"""Agent tool definitions (schemas) and their execution.

Adding a tool: write its JSON schema into ``TOOLS``, implement an async
handler on :class:`Toolbox`, and register it in ``self._handlers``.

Schemas use strict mode, so argument types are guaranteed by the API and
handlers don't need defensive casts; optional parameters are nullable.

Built-in tools (web search) execute server-side: they produce no
``function_call`` items, so the agent loop needs no handler for them —
they just have to be listed, which :func:`build_tools` does.
"""

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message, ReactionTypeEmoji, ReplyParameters
from aiogram.utils.chat_action import ChatActionSender
from openai import AsyncOpenAI
from openai.types.responses import ToolParam

from libertati import clock
from libertati.db import Database
from libertati.memory import Mind

log = logging.getLogger(__name__)

#: Bounds for the simulated typing time of outgoing messages.
TYPING_MIN_SECONDS = 1.0
TYPING_MAX_SECONDS = 8.0


def typing_delay(text: str, chars_per_second: float) -> float:
    """How long to pretend to type a message, with human jitter.

    A non-positive speed disables the emulation (returns 0).
    """
    if chars_per_second <= 0:
        return 0.0
    seconds = TYPING_MIN_SECONDS + len(text) / chars_per_second
    return min(seconds, TYPING_MAX_SECONDS) * random.uniform(0.8, 1.2)


#: Instructions for the one-shot recall extraction call.
RECALL_PROMPT = (
    "You are the long-term memory of a person texting on Telegram. Below "
    "are their notes — a curated '# Memory' section and fresh dated "
    "'# Inbox' entries — then a query. Answer the query from the notes "
    "only: quote or paraphrase the relevant entries (with dates when they "
    "matter) and say plainly when the notes contain nothing relevant."
)

#: Instructions for the one-shot memory overview call.
SUMMARY_PROMPT = (
    "You are the long-term memory of a person texting on Telegram. Below "
    "are their notes — a curated '# Memory' section and fresh dated "
    "'# Inbox' entries. Give a short general overview of what is "
    "remembered: the people and key facts, recurring themes, open plans "
    "and promises, and the time span covered. A map, not the details — "
    "specifics can be fetched later with targeted recall."
)

# ==========================================================
#                        Messaging
# ==========================================================
MESSAGING_TOOLS: list[ToolParam] = [
    {
        "type": "function",
        "name": "send_message",
        "description": (
            "Send a Telegram message to a chat. This is the ONLY way to "
            "actually say something; plain text output sends nothing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Target chat id (shown in incoming events).",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Message text. Telegram markdown only: *bold*, "
                        "_italic_, `code`, ```blocks```. Headers, tables "
                        "and list markup don't render — plain text and "
                        "bare URLs instead."
                    ),
                },
                "reply_to_message_id": {
                    "type": ["integer", "null"],
                    "description": (
                        "Message id to quote-reply to, or null. Quote only "
                        "when it clarifies what you're answering: the message "
                        "is buried under newer ones, or a group conversation "
                        "has several threads going. When you're answering the "
                        "latest message — always in private chats, usually in "
                        "groups with one active thread — pass null: quoting "
                        "every message reads robotic."
                    ),
                },
            },
            "required": ["chat_id", "text", "reply_to_message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "send_sticker",
        "description": (
            "Send a sticker to a chat — sometimes a sticker says it "
            "better than words. Only stickers you've seen can be sent: "
            "pick a file_id from list_stickers first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Target chat id.",
                },
                "file_id": {
                    "type": "string",
                    "description": "Sticker file_id (from list_stickers).",
                },
            },
            "required": ["chat_id", "file_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_stickers",
        "description": (
            "List stickers you can send: every sticker seen in any chat, "
            "with its emoji, set name and file_id for send_sticker."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "forward_message",
        "description": (
            "Forward a message from one chat to another — share a meme, "
            "a link, a photo someone sent you. The recipient sees the "
            "original sender."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to_chat_id": {
                    "type": "integer",
                    "description": "Chat to forward the message to.",
                },
                "from_chat_id": {
                    "type": "integer",
                    "description": "Chat the message is currently in.",
                },
                "message_id": {
                    "type": "integer",
                    "description": "Id of the message to forward.",
                },
            },
            "required": ["to_chat_id", "from_chat_id", "message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "react",
        "description": (
            "Put an emoji reaction on a message — the lightest way to "
            "acknowledge something without texting back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat the message is in.",
                },
                "message_id": {
                    "type": "integer",
                    "description": "Message to react to.",
                },
                "emoji": {
                    "type": ["string", "null"],
                    "description": (
                        "One emoji from Telegram's reaction set, e.g. "
                        "👍 ❤ 🔥 🎉 😁 😢 🤔 👏 💯 🙏; null removes "
                        "your reaction."
                    ),
                },
            },
            "required": ["chat_id", "message_id", "emoji"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "edit_message",
        "description": (
            "Rewrite the text of a message you sent — fix a typo, correct "
            "a fact. Works only on your own recent messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat the message is in.",
                },
                "message_id": {
                    "type": "integer",
                    "description": "Id of your message to edit.",
                },
                "text": {
                    "type": "string",
                    "description": "New message text (same markdown rules).",
                },
            },
            "required": ["chat_id", "message_id", "text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "delete_message",
        "description": (
            "Delete a message you sent — retract something that shouldn't "
            "stay. Works only on your own recent messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat the message is in.",
                },
                "message_id": {
                    "type": "integer",
                    "description": "Id of your message to delete.",
                },
            },
            "required": ["chat_id", "message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# ==========================================================
#                        Scheduling
# ==========================================================
SCHEDULING_TOOLS: list[ToolParam] = [
    {
        "type": "function",
        "name": "schedule_wakeup",
        "description": (
            "Set an alarm for your future self: at the given time you will "
            "receive a wakeup event carrying your note. Use it whenever you "
            "intend to do something later — don't rely on remembering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "when": {
                    "type": "string",
                    "description": (
                        "When to wake up, 'YYYY-MM-DD HH:MM' in your own timezone."
                    ),
                },
                "note": {
                    "type": "string",
                    "description": "Note to your future self: what and why.",
                },
            },
            "required": ["when", "note"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_wakeups",
        "description": (
            "List your pending wakeups: id, due time and note. Use to "
            "check what you've already planned before scheduling more."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "cancel_wakeup",
        "description": (
            "Cancel a pending wakeup by id when the plan behind it is no "
            "longer relevant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "wakeup_id": {
                    "type": "integer",
                    "description": "Id of the wakeup to cancel (see list_wakeups).",
                },
            },
            "required": ["wakeup_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# ==========================================================
#                     Chats & History
# ==========================================================
HISTORY_TOOLS: list[ToolParam] = [
    {
        "type": "function",
        "name": "list_chats",
        "description": (
            "List every chat you know: id, type, name, message count and "
            "time of the last message. Use to see who you can talk to, "
            "e.g. before reaching out to someone."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_chat_info",
        "description": (
            "Fetch a chat's live profile from Telegram: name, username, "
            "bio or description, and for groups the member count and "
            "admins."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat id to look up.",
                },
            },
            "required": ["chat_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_chat_members",
        "description": (
            "List who you've seen talking in a chat (from stored "
            "history), with message counts and last activity. Telegram "
            "hides a group's full roster from bots, so silent members "
            "don't appear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat id to list members of.",
                },
            },
            "required": ["chat_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_recent_messages",
        "description": (
            "Fetch messages stored for a chat, newest last. Use to recall "
            "context beyond what you remember; page further into the past "
            "with before_message_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat id to read history from.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "description": "How many messages to fetch (max 50); null = 20.",
                },
                "before_message_id": {
                    "type": ["integer", "null"],
                    "description": (
                        "Only messages older than this id; null starts at "
                        "the newest. To page back, pass the smallest "
                        "message_id of the previous batch."
                    ),
                },
            },
            "required": ["chat_id", "limit", "before_message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_messages",
        "description": (
            "Search one chat's whole history for messages containing a "
            "text fragment (case-insensitive), newest first. Use to find "
            "what was said long ago without paging through everything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "integer",
                    "description": "Chat id to search in.",
                },
                "query": {
                    "type": "string",
                    "description": "Text fragment to look for.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "description": "How many matches to return (max 50); null = 20.",
                },
            },
            "required": ["chat_id", "query", "limit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# ==========================================================
#                          Memory
# ==========================================================
MEMORY_TOOLS: list[ToolParam] = [
    {
        "type": "function",
        "name": "remember",
        "description": (
            "Save one durable fact to your long-term memory (survives "
            "forever, unlike this context). Use for things worth keeping: "
            "people, preferences, promises, your own plans. One short "
            "fact per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact to remember, one short sentence.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "recall",
        "description": (
            "Ask your long-term memory a question. Use before answering "
            "anything that depends on the past beyond what you currently "
            "see: names, preferences, promises, earlier plans."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you are trying to remember, as a question.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "summarize_memory",
        "description": (
            "Get a general overview of everything in your long-term "
            "memory — what do you even remember? Use to orient yourself; "
            "follow up with recall for specifics."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

#: All function tools, in the order the model sees them.
TOOLS: list[ToolParam] = [
    *MESSAGING_TOOLS,
    *SCHEDULING_TOOLS,
    *HISTORY_TOOLS,
    *MEMORY_TOOLS,
]


def build_tools(web_search: bool) -> list[ToolParam]:
    """Return the tool list for the API, optionally with built-in web search.

    Web search runs on OpenAI's side; it is opt-in because most
    OpenAI-compatible endpoints don't support it.
    """
    if web_search:
        return [*TOOLS, {"type": "web_search"}]
    return list(TOOLS)


class Toolbox:
    """Executes the agent's tool calls against application resources.

    One instance serves the single agent loop; tools that target a chat
    take an explicit ``chat_id`` argument from the model.
    """

    def __init__(
        self,
        db: Database,
        bot: Bot,
        tz: ZoneInfo,
        client: AsyncOpenAI,
        recall_model: str,
        mind: Mind,
        typing_chars_per_second: float,
    ) -> None:
        """Keep resource handles and build the name-to-handler dispatch."""
        self.db = db
        self.bot = bot
        self.tz = tz
        self.client = client
        self.recall_model = recall_model
        self.mind = mind
        self.typing_chars_per_second = typing_chars_per_second
        self._handlers = {
            # messaging
            "send_message": self._send_message,
            "send_sticker": self._send_sticker,
            "list_stickers": self._list_stickers,
            "forward_message": self._forward_message,
            "react": self._react,
            "edit_message": self._edit_message,
            "delete_message": self._delete_message,
            # scheduling
            "schedule_wakeup": self._schedule_wakeup,
            "list_wakeups": self._list_wakeups,
            "cancel_wakeup": self._cancel_wakeup,
            # chats & history
            "list_chats": self._list_chats,
            "get_chat_info": self._get_chat_info,
            "list_chat_members": self._list_chat_members,
            "get_recent_messages": self._get_recent_messages,
            "search_messages": self._search_messages,
            # memory
            "remember": self._remember,
            "recall": self._recall,
            "summarize_memory": self._summarize_memory,
        }

    async def run(self, name: str, arguments: str | None) -> str:
        """Execute one tool call and return its result as a string.

        Never raises: unknown tools, malformed arguments and handler
        failures all come back as ``error: …`` strings so the model can
        recover on the next loop step.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return f"error: unknown tool {name!r}"
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "error: invalid tool arguments"
        log.info("tool call: %s(%s)", name, args)
        try:
            return await handler(args)
        except Exception as exc:
            log.exception("tool %s failed", name)
            return f"error: {exc}"

    # ==========================================================
    #                        Messaging
    # ==========================================================

    async def _send_message(self, args: dict[str, Any]) -> str:
        """Send a message (optionally as a reply) and persist it as outgoing.

        Shows the Telegram "typing…" indicator for a length-proportional
        moment first, so replies land at a human pace. Text is sent as
        Telegram markdown; when Telegram rejects the markup (model
        output with stray ``*``/``_`` is easy to unbalance), the message
        is resent as plain text rather than lost.
        """
        reply_to = args.get("reply_to_message_id")
        delay = typing_delay(args["text"], self.typing_chars_per_second)
        if delay > 0:
            async with ChatActionSender.typing(chat_id=args["chat_id"], bot=self.bot):
                await asyncio.sleep(delay)
        reply_parameters = (
            ReplyParameters(message_id=reply_to) if reply_to is not None else None
        )
        sent = await self._markdown_send(
            lambda parse_mode: self.bot.send_message(
                args["chat_id"],
                args["text"],
                parse_mode=parse_mode,
                reply_parameters=reply_parameters,
            )
        )
        await self.db.save_message(sent, outgoing=True)
        return (
            f"sent message {sent.message_id} to chat {sent.chat.id}"
            f" at {clock.format_now(self.tz)}"
        )

    async def _send_sticker(self, args: dict[str, Any]) -> str:
        """Send a known sticker by file_id and persist it as outgoing."""
        sent = await self.bot.send_sticker(args["chat_id"], args["file_id"])
        await self.db.save_message(sent, outgoing=True)
        return f"sent sticker as message {sent.message_id} to chat {sent.chat.id}"

    async def _list_stickers(self, args: dict[str, Any]) -> str:
        """Return every sticker seen so far (sendable file_ids) as JSON."""
        rows = await self.db.known_stickers(50)
        if not rows:
            return "no stickers seen yet — stickers people send you land here"
        return json.dumps(rows, ensure_ascii=False)

    async def _forward_message(self, args: dict[str, Any]) -> str:
        """Forward a message between chats and persist the copy as outgoing."""
        sent = await self.bot.forward_message(
            args["to_chat_id"], args["from_chat_id"], args["message_id"]
        )
        await self.db.save_message(sent, outgoing=True)
        return (
            f"forwarded message {args['message_id']} from chat "
            f"{args['from_chat_id']} to chat {args['to_chat_id']} "
            f"as message {sent.message_id}"
        )

    @staticmethod
    async def _markdown_send(send: Callable[[str | None], Awaitable[Any]]) -> Any:
        """Call ``send`` with markdown; resend plain when Telegram rejects it.

        Model output with stray ``*``/``_`` is easy to unbalance; the
        message then degrades to plain text rather than getting lost.
        """
        try:
            return await send(ParseMode.MARKDOWN)
        except TelegramBadRequest:
            return await send(None)

    async def _react(self, args: dict[str, Any]) -> str:
        """Set or remove an emoji reaction on a message."""
        emoji = args["emoji"]
        reaction = [ReactionTypeEmoji(emoji=emoji)] if emoji else []
        await self.bot.set_message_reaction(
            args["chat_id"], args["message_id"], reaction=reaction
        )
        target = f"message {args['message_id']} in chat {args['chat_id']}"
        if emoji:
            return f"reacted {emoji} to {target}"
        return f"reaction removed from {target}"

    async def _edit_message(self, args: dict[str, Any]) -> str:
        """Rewrite one of the bot's own messages and persist the new text."""
        edited = await self._markdown_send(
            lambda parse_mode: self.bot.edit_message_text(
                text=args["text"],
                chat_id=args["chat_id"],
                message_id=args["message_id"],
                parse_mode=parse_mode,
            )
        )
        if isinstance(edited, Message):
            await self.db.save_message(edited, outgoing=True)
        return f"edited message {args['message_id']} in chat {args['chat_id']}"

    async def _delete_message(self, args: dict[str, Any]) -> str:
        """Delete one of the bot's own messages on Telegram and in the DB."""
        await self.bot.delete_message(args["chat_id"], args["message_id"])
        await self.db.delete_message(args["chat_id"], args["message_id"])
        return f"deleted message {args['message_id']} in chat {args['chat_id']}"

    # ==========================================================
    #                        Scheduling
    # ==========================================================

    async def _schedule_wakeup(self, args: dict[str, Any]) -> str:
        """Store a future wakeup for the agent itself."""
        try:
            due = clock.parse_local(args["when"], self.tz)
        except ValueError:
            return "error: 'when' must be 'YYYY-MM-DD HH:MM'"
        if due <= datetime.now(UTC):
            return (
                f"error: {args['when']} is in the past,"
                f" now is {clock.format_now(self.tz)}"
            )
        wakeup_id = await self.db.add_wakeup(clock.utc_stamp(due), args["note"])
        return f"wakeup #{wakeup_id} scheduled for {clock.format_local(due, self.tz)}"

    async def _list_wakeups(self, args: dict[str, Any]) -> str:
        """Return pending wakeups (id, local due time, note) as JSON."""
        rows = await self.db.pending_wakeups()
        if not rows:
            return "no pending wakeups"
        wakeups = [
            {
                "id": row["id"],
                "due": clock.format_local(
                    clock.parse_utc_stamp(row["due_at"]), self.tz
                ),
                "note": row["note"],
            }
            for row in rows
        ]
        return json.dumps(wakeups, ensure_ascii=False)

    async def _cancel_wakeup(self, args: dict[str, Any]) -> str:
        """Cancel one pending wakeup by id."""
        if await self.db.cancel_wakeup(args["wakeup_id"]):
            return f"wakeup #{args['wakeup_id']} cancelled"
        return f"error: no pending wakeup #{args['wakeup_id']}"

    # ==========================================================
    #                     Chats & History
    # ==========================================================

    async def _list_chats(self, args: dict[str, Any]) -> str:
        """Return all known chats with names and activity stats as JSON."""
        chats = await self.db.list_chats()
        if not chats:
            return "no chats yet"
        return json.dumps(chats, ensure_ascii=False)

    async def _get_chat_info(self, args: dict[str, Any]) -> str:
        """Return a chat's live Telegram profile as JSON.

        For group chats the member count and admin names are fetched too;
        for private chats Telegram only exposes the profile fields.
        """
        chat = await self.bot.get_chat(args["chat_id"])
        info: dict[str, Any] = {
            "chat_id": chat.id,
            "type": chat.type,
            "title": chat.title,
            "first_name": chat.first_name,
            "last_name": chat.last_name,
            "username": chat.username,
            "bio": chat.bio,
            "description": chat.description,
        }
        if chat.type != "private":
            info["member_count"] = await self.bot.get_chat_member_count(chat.id)
            admins = await self.bot.get_chat_administrators(chat.id)
            info["admins"] = [
                f"{member.user.full_name}"
                + (f" @{member.user.username}" if member.user.username else "")
                for member in admins
            ]
        return json.dumps(
            {key: value for key, value in info.items() if value is not None},
            ensure_ascii=False,
        )

    async def _list_chat_members(self, args: dict[str, Any]) -> str:
        """Return users seen talking in a chat as JSON."""
        rows = await self.db.chat_members(args["chat_id"])
        if not rows:
            return "nobody seen talking in this chat yet"
        return json.dumps(rows, ensure_ascii=False)

    async def _get_recent_messages(self, args: dict[str, Any]) -> str:
        """Return a page of a chat's messages as JSON, oldest first."""
        limit = max(1, min(args.get("limit") or 20, 50))
        rows = await self.db.recent_messages(
            args["chat_id"], limit, args.get("before_message_id")
        )
        return json.dumps(rows, ensure_ascii=False)

    async def _search_messages(self, args: dict[str, Any]) -> str:
        """Return a chat's messages matching a substring as JSON."""
        limit = max(1, min(args.get("limit") or 20, 50))
        rows = await self.db.search_messages(args["chat_id"], args["query"], limit)
        if not rows:
            return "no matches"
        return json.dumps(rows, ensure_ascii=False)

    # ==========================================================
    #                          Memory
    # ==========================================================

    async def _remember(self, args: dict[str, Any]) -> str:
        """Append one dated fact to the memory inbox."""
        text = args["text"].strip()
        self.mind.append_inbox(text, clock.format_now(self.tz))
        return f"remembered: {text}"

    async def _read_memory(self, instructions: str, input_text: str) -> str:
        """Run one no-loop extraction call over the memory notes."""
        response = await self.client.responses.create(
            model=self.recall_model,
            instructions=instructions,
            input=input_text,
            store=False,
        )
        return response.output_text or "recall came back empty"

    async def _recall(self, args: dict[str, Any]) -> str:
        """Answer a query from memory + inbox via a one-shot extraction call."""
        notes = self.mind.notes()
        if not notes:
            return "memory is empty"
        return await self._read_memory(
            RECALL_PROMPT, f"{notes}\n\nQuery: {args['query']}"
        )

    async def _summarize_memory(self, args: dict[str, Any]) -> str:
        """Return a general overview of everything in memory + inbox."""
        notes = self.mind.notes()
        if not notes:
            return "memory is empty"
        return await self._read_memory(SUMMARY_PROMPT, notes)
