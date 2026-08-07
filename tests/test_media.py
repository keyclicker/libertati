"""Tests for resolving, compressing and describing message media."""

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from aiogram import Bot
from conftest import run_sql
from openai import AsyncOpenAI, BadRequestError, InternalServerError

from libertati.db import Database
from libertati.media import (
    MediaLens,
    audio_command,
    frames_command,
    image_command,
    media_ref,
)

STICKER = {
    "file_id": "sticker-file",
    "file_unique_id": "sticker-uid",
    "width": 512,
    "height": 512,
    "is_animated": False,
    "is_video": False,
    "type": "regular",
    "emoji": "😼",
    "set_name": "cats",
}


def photo_payload() -> dict[str, Any]:
    """Build a message payload carrying Telegram's photo size ladder."""
    return {
        "photo": [
            {"file_id": "tiny", "file_unique_id": "tiny-uid", "width": 90},
            {"file_id": "small", "file_unique_id": "small-uid", "width": 320},
            {"file_id": "big", "file_unique_id": "big-uid", "width": 1280},
        ]
    }


class FakeResponses:
    """Records the one-shot describe calls the lens makes."""

    def __init__(self, text: str, delay: float = 0.0) -> None:
        """Answer every call with the same text, after ``delay``."""
        self.text = text
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        #: Set to make every call refuse (a Responses refusal part) or
        #: raise, the way a real provider would.
        self.refusal: str | None = None
        self.error: Exception | None = None

    async def create(self, **kwargs: Any) -> Any:
        """Return a fake Responses-API result."""
        self.calls.append(kwargs)
        self.started.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.refusal is not None:
            part = SimpleNamespace(type="refusal", refusal=self.refusal)
            message = SimpleNamespace(type="message", content=[part])
            return SimpleNamespace(output_text="", output=[message], usage=None)
        return SimpleNamespace(output_text=self.text, usage=None)


class FakeTranscriptions:
    """Records the speech-to-text calls the lens makes."""

    def __init__(self, text: str) -> None:
        """Answer every call with the same transcript."""
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        """Answer the way a ``json`` response format is handed back."""
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class FakeClient:
    """Minimal stand-in for the OpenAI client the lens is given."""

    def __init__(
        self,
        text: str = "a cat knocking a mug over",
        audio: str = "",
        delay: float = 0.0,
    ) -> None:
        """Wire the two endpoints the lens can reach."""
        self.responses = FakeResponses(text, delay)
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions(audio))


class FakeBot:
    """Bot that "downloads" by handing back an in-memory buffer."""

    def __init__(self, data: bytes = b"original-bytes") -> None:
        """Start with nothing downloaded."""
        self.data = data
        self.downloads: list[str] = []

    async def download(self, file_id: str) -> io.BytesIO | None:
        """Record the download and return the placeholder bytes."""
        self.downloads.append(file_id)
        return io.BytesIO(self.data)


@pytest.fixture(autouse=True)
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], bytes]]:
    """Stand in for ffmpeg: every pipe run "encodes" to placeholder bytes.

    Records each (command, stdin) pair, so a test can check what would
    have been piped where. ffprobe runs go through the same stub; its
    non-numeric answer reads as an unknown duration, which is fine —
    duration only shapes the ffmpeg filter, and no real ffmpeg runs.
    """
    calls: list[tuple[list[str], bytes]] = []

    async def fake_run(command: list[str], data: bytes) -> bytes | None:
        calls.append((command, data))
        return b"encoded"

    monkeypatch.setattr("libertati.media._run", fake_run)
    return calls


def make_lens(
    db: Database,
    client: FakeClient | None = None,
    bot: FakeBot | None = None,
    **extra: Any,
) -> MediaLens:
    """Build a lens over the real database and fake I/O."""
    return MediaLens(
        db=db,
        bot=cast(Bot, bot or FakeBot()),
        client=cast(AsyncOpenAI, client or FakeClient()),
        model="eyes",
        describe_prompt="describe it",
        answer_prompt="answer the question",
        **extra,
    )


# ==========================================================
#                    Reading a payload
# ==========================================================


def test_photo_downloads_a_small_variant_but_is_named_by_the_largest() -> None:
    """The cheap variant is fetched; identity must not depend on that."""
    ref = media_ref(photo_payload())
    assert ref is not None
    assert ref.file_id == "small"
    assert ref.file_unique_id == "big-uid"
    assert ref.source == "image"


def test_photo_without_a_wide_enough_variant_takes_the_largest() -> None:
    """A tiny photo has nothing better to offer than its full size."""
    ref = media_ref(
        {"photo": [{"file_id": "tiny", "file_unique_id": "u", "width": 90}]}
    )
    assert ref is not None
    assert ref.file_id == "tiny"


def test_sticker_hint_carries_its_emoji_and_set() -> None:
    """What a sticker means is mostly in the metadata around it."""
    ref = media_ref({"sticker": STICKER})
    assert ref is not None
    assert "😼" in ref.hint
    assert "cats" in ref.hint
    assert ref.source == "image"


def test_animated_sticker_is_described_through_its_thumbnail() -> None:
    """Lottie stickers decode nowhere here, but Telegram ships a still."""
    ref = media_ref(
        {
            "sticker": {
                **STICKER,
                "is_animated": True,
                "thumbnail": {
                    "file_id": "thumb-file",
                    "file_unique_id": "thumb-uid",
                    "width": 128,
                    "height": 128,
                },
            }
        }
    )
    assert ref is not None
    assert ref.file_id == "thumb-file"
    # The cache key stays the sticker's, or the thumbnail would become a
    # second entry describing the same sticker.
    assert ref.file_unique_id == "sticker-uid"
    assert ref.source == "image"


def test_video_sticker_goes_through_the_frame_path() -> None:
    """A webm sticker is a tiny video and ffmpeg can read it."""
    ref = media_ref({"sticker": {**STICKER, "is_video": True}})
    assert ref is not None
    assert ref.source == "video"
    assert ref.file_id == "sticker-file"


def test_gif_and_video_note_are_frame_sources() -> None:
    """Everything that moves resolves to the same source kind."""
    gif = media_ref(
        {
            "animation": {
                "file_id": "gif",
                "file_unique_id": "gif-uid",
                "width": 320,
                "height": 240,
                "duration": 4,
            }
        }
    )
    assert gif is not None
    assert (gif.kind, gif.source) == ("animation", "video")
    assert "4s" in gif.hint
    note = media_ref(
        {
            "video_note": {
                "file_id": "vn",
                "file_unique_id": "vn-uid",
                "length": 240,
                "duration": 7,
            }
        }
    )
    assert note is not None
    assert (note.kind, note.source) == ("video_note", "video")


def test_voice_is_an_audio_source() -> None:
    """Voice messages take the transcription branch."""
    ref = media_ref(
        {"voice": {"file_id": "v", "file_unique_id": "v-uid", "duration": 12}}
    )
    assert ref is not None
    assert (ref.kind, ref.source) == ("voice", "audio")


def test_messages_without_describable_media_resolve_to_nothing() -> None:
    """Text and documents are left alone."""
    assert media_ref({"text": "hi"}) is None
    assert media_ref({"document": {"file_id": "d", "file_unique_id": "d-uid"}}) is None


# ==========================================================
#                      Building frames
# ==========================================================


def test_every_command_reads_stdin_and_writes_stdout() -> None:
    """Files are what this module avoids; ffmpeg touches pipes only."""
    for command in (image_command(), frames_command(3, 4.0), audio_command()):
        assert command[command.index("-i") + 1] == "pipe:0"
        assert command[-1] == "pipe:1"


def test_frames_command_tiles_the_clip_across_its_duration() -> None:
    """Three frames of a four-second gif, sampled evenly, side by side."""
    command = frames_command(3, 4.0)
    filters = command[command.index("-vf") + 1]
    assert "fps=3/4.000" in filters
    assert "tile=3x1" in filters


def test_frames_command_falls_back_to_one_still() -> None:
    """A clip of unknown length would tile into a half-empty image."""
    assert frames_command(3, 0.0) == image_command()


def test_audio_command_produces_small_mono_opus() -> None:
    """Voice is kept re-transcribable, not hi-fi."""
    command = audio_command()
    assert "libopus" in command
    assert command[command.index("-ac") + 1] == "1"


# ==========================================================
#                       Describing
# ==========================================================


async def test_a_file_is_described_once_and_read_back_forever(
    db: Database,
) -> None:
    """The second look at the same sticker costs nothing."""
    client = FakeClient()
    bot = FakeBot()
    lens = make_lens(db, client, bot)

    first = await lens.look(10, 1, {"sticker": STICKER})
    second = await lens.look(10, 2, {"sticker": STICKER})

    assert first == "a cat knocking a mug over"
    assert second == first
    assert len(client.responses.calls) == 1
    assert bot.downloads == ["sticker-file"]
    assert await db.media_note("sticker-uid") == first


async def test_the_original_reaches_ffmpeg_as_bytes_never_a_file(
    db: Database, ffmpeg: list[tuple[list[str], bytes]]
) -> None:
    """What Telegram hands over is piped straight through, byte for byte."""
    bot = FakeBot(b"webm-bytes")
    lens = make_lens(db, bot=bot)

    await lens.look(10, 1, {"sticker": {**STICKER, "is_video": True}})

    assert [data for _, data in ffmpeg] == [b"webm-bytes", b"webm-bytes"]
    probe, encode = (command for command, _ in ffmpeg)
    assert probe[0] == "ffprobe"
    assert encode[0] == "ffmpeg"


async def test_the_model_is_sent_what_ffmpeg_piped_out(db: Database) -> None:
    """The upload is ffmpeg's stdout, not the original."""
    client = FakeClient()
    lens = make_lens(db, client)

    await lens.look(10, 1, {"sticker": STICKER})

    image = client.responses.calls[0]["input"][0]["content"][1]["image_url"]
    assert image == "data:image/webp;base64,ZW5jb2RlZA=="  # b"encoded"


async def test_a_file_ffmpeg_cannot_decode_stays_undescribed(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decode failure is a missing note, never an exception or a call."""

    async def broken_run(command: list[str], data: bytes) -> bytes | None:
        return None

    monkeypatch.setattr("libertati.media._run", broken_run)
    client = FakeClient()
    lens = make_lens(db, client)

    assert await lens.look(10, 1, {"sticker": STICKER}) is None
    assert client.responses.calls == []
    assert await db.media_note("sticker-uid") is None


async def test_a_file_past_telegrams_limit_is_never_fetched(db: Database) -> None:
    """Telegram would refuse the download, so it is not attempted."""
    bot = FakeBot()
    lens = make_lens(db, bot=bot)
    huge = {"sticker": {**STICKER, "file_size": 21 * 1024 * 1024}}

    assert await lens.look(10, 1, huge) is None
    assert bot.downloads == []


async def test_a_described_message_is_linked_to_its_file(db: Database) -> None:
    """The link is the join a transcript renders the note through."""
    lens = make_lens(db)
    await run_sql(db, "INSERT INTO chats (id, type, raw) VALUES (10, 'private', '{}')")
    await run_sql(
        db,
        """
        INSERT INTO messages (chat_id, message_id, date, content_type, raw)
        VALUES (10, 1, '2026-08-04T05:46:31+00:00', 'sticker', '{}')
        """,
    )

    await lens.look(10, 1, {"sticker": STICKER})

    row = await db.message_row(10, 1)
    assert row is not None
    assert row["media_note"] == "a cat knocking a mug over"


async def test_a_note_is_folded_and_capped(db: Database) -> None:
    """Notes live inside one transcript line and are paid for per turn."""
    client = FakeClient("a very\nlong\nanswer " + "x" * 200)
    lens = make_lens(db, client, note_chars=30)

    note = await lens.look(10, 1, {"sticker": STICKER})

    assert note is not None
    assert "\n" not in note
    assert len(note) <= 31  # the cap plus the ellipsis marking the cut
    assert note.startswith("a very long answer")


async def test_an_empty_answer_is_not_stored(db: Database) -> None:
    """A model that said nothing must not poison the cache with it."""
    lens = make_lens(db, FakeClient(""))

    assert await lens.look(10, 1, {"sticker": STICKER}) is None
    assert await db.media_note("sticker-uid") is None


async def test_voice_needs_a_configured_transcription_model(db: Database) -> None:
    """Without one, voice messages stay undescribed instead of failing."""
    client = FakeClient(audio="see you at six")
    lens = make_lens(db, client)
    voice = {"voice": {"file_id": "v", "file_unique_id": "v-uid", "duration": 3}}

    assert await lens.look(10, 1, voice) is None
    assert client.audio.transcriptions.calls == []


async def test_voice_is_transcribed_when_a_model_is_configured(db: Database) -> None:
    """The transcript is the note, stored like any other."""
    client = FakeClient(audio="see you at six")
    lens = make_lens(db, client, transcribe_model="ears")

    note = await lens.look(
        10, 1, {"voice": {"file_id": "v", "file_unique_id": "v-uid", "duration": 3}}
    )

    assert note == "see you at six"
    assert client.responses.calls == []
    call = client.audio.transcriptions.calls[0]
    # Not "text": OpenRouter rejects that format outright, and json is
    # the one every transcription endpoint offers. The upload is named
    # ".ogg" because endpoints read the format off the filename.
    assert call["response_format"] == "json"
    assert call["file"][0].endswith(".ogg")


async def test_a_slow_description_does_not_hold_up_the_event(db: Database) -> None:
    """The event goes out bare; the note lands for every later transcript."""
    client = FakeClient("a cat", delay=0.05)
    lens = make_lens(db, client, wait_seconds=0.0)

    job = lens.start(10, 1, {"sticker": STICKER})
    assert await lens.wait_briefly(job) is None

    await client.responses.started.wait()
    await asyncio.gather(*lens._tasks)
    assert await db.media_note("sticker-uid") == "a cat"


async def test_nothing_is_started_for_a_message_without_media(db: Database) -> None:
    """There is no job to wait for, and none was queued."""
    lens = make_lens(db)
    assert lens.start(10, 1, {"text": "hi"}) is None
    assert await lens.wait_briefly(None) is None
    assert lens._tasks == set()


async def test_a_flood_of_media_is_dropped_rather_than_queued(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backlog nobody will read is a bill nobody meant to pay."""
    monkeypatch.setattr("libertati.media.MAX_PENDING_JOBS", 1)
    client = FakeClient("a cat", delay=0.05)
    lens = make_lens(db, client)

    first = lens.start(10, 1, {"sticker": STICKER})
    second = lens.start(10, 2, {"sticker": STICKER})

    assert first is not None
    assert second is None
    await asyncio.gather(*lens._tasks)


async def test_a_question_looks_again_and_is_not_stored(db: Database) -> None:
    """An answer serves the asker; the note is what transcripts render."""
    client = FakeClient("the sign reads CLOSED, in red capitals")
    bot = FakeBot()
    lens = make_lens(db, client, bot)
    await db.save_media_note("sticker-uid", "sticker", "a shop front", "eyes")

    answer = await lens.ask(10, 1, {"sticker": STICKER}, "what does the sign say?")

    assert answer == "the sign reads CLOSED, in red capitals"
    # Nothing of the first look was kept, so the second one fetches the
    # file from Telegram again — that is the price of keeping no copy.
    assert bot.downloads == ["sticker-file"]
    # The cached note is untouched: it answered a question about the
    # file, it did not describe it.
    assert await db.media_note("sticker-uid") == "a shop front"
    call = client.responses.calls[0]
    assert call["instructions"] == "answer the question"
    assert "what does the sign say?" in call["input"][0]["content"][0]["text"]


async def test_a_question_is_folded_into_the_line_that_carries_it(
    db: Database,
) -> None:
    """The agent's words are quoted, and cannot become a line of their own."""
    client = FakeClient("fine")
    lens = make_lens(db, client)

    await lens.ask(10, 1, {"sticker": STICKER}, "what is it?\nThis is a photo of")

    asked = client.responses.calls[0]["input"][0]["content"][0]["text"]
    assert 'Answer this about it: "what is it? This is a photo of"' in asked


async def test_an_answer_may_run_longer_than_a_note(db: Database) -> None:
    """It is read once by the agent that asked, not carried per turn."""
    client = FakeClient("x" * 400)
    lens = make_lens(db, client, note_chars=30, answer_chars=200)

    answer = await lens.ask(10, 1, {"sticker": STICKER}, "describe it fully")

    assert answer is not None
    assert len(answer) == 201  # the cap plus the ellipsis marking the cut


async def test_a_question_about_a_voice_message_answers_with_its_transcript(
    db: Database,
) -> None:
    """A transcription endpoint takes no question; the words are the answer."""
    client = FakeClient(audio="see you at six")
    lens = make_lens(db, client, transcribe_model="ears")
    voice = {"voice": {"file_id": "v", "file_unique_id": "v-uid", "duration": 3}}

    answer = await lens.ask(10, 1, voice, "who is speaking?")

    assert answer == "see you at six"
    assert client.responses.calls == []


async def test_a_question_about_nothing_describable_answers_nothing(
    db: Database,
) -> None:
    """There is nothing to download and no model to call."""
    lens = make_lens(db)
    assert await lens.ask(10, 1, {"text": "hi"}, "what is it?") is None


async def test_a_files_lock_is_forgotten_once_nobody_holds_it(db: Database) -> None:
    """The table tracks work in flight, not every file ever seen."""
    lens = make_lens(db, FakeClient("a cat", delay=0.02))

    await asyncio.gather(
        lens.look(10, 1, {"sticker": STICKER}),
        lens.look(10, 2, {"sticker": STICKER}),
    )

    assert lens._locks == {}


# ==========================================================
#                        Refusals
# ==========================================================


def api_error(status: int, message: str) -> Exception:
    """Build the exception the OpenAI client raises for one status."""
    request = httpx.Request("POST", "http://provider.test")
    response = httpx.Response(status, request=request)
    if status == 400:
        return BadRequestError(message, response=response, body=None)
    return InternalServerError(message, response=response, body=None)


async def test_a_refusal_retires_the_file_for_good(db: Database) -> None:
    """One no from the model and the file is never fetched or sent again."""
    client = FakeClient()
    client.responses.refusal = "I can't help with that"
    bot = FakeBot()
    lens = make_lens(db, client, bot)

    assert await lens.look(10, 1, {"sticker": STICKER}) is None
    assert await db.media_refused("sticker-uid")
    assert await db.media_note("sticker-uid") is None

    # The second look is over before it starts: no download, no call.
    assert await lens.look(10, 2, {"sticker": STICKER}) is None
    assert bot.downloads == ["sticker-file"]
    assert len(client.responses.calls) == 1


async def test_a_content_policy_400_is_a_refusal_too(db: Database) -> None:
    """Some providers say no as an error rather than a refusal part."""
    client = FakeClient()
    client.responses.error = api_error(400, "your input was flagged")
    lens = make_lens(db, client)

    assert await lens.look(10, 1, {"sticker": STICKER}) is None
    assert await db.media_refused("sticker-uid")


async def test_a_transient_error_is_not_a_refusal(db: Database) -> None:
    """A 500 tonight must not blacklist an innocent file forever."""
    client = FakeClient()
    client.responses.error = api_error(500, "upstream fell over")
    lens = make_lens(db, client)

    assert await lens.look(10, 1, {"sticker": STICKER}) is None
    assert not await db.media_refused("sticker-uid")


async def test_a_malformed_request_is_our_bug_not_a_refusal(db: Database) -> None:
    """A 400 without a policy smell stays retryable — we asked wrong."""
    client = FakeClient()
    client.responses.error = api_error(400, "image exceeds maximum dimensions")
    lens = make_lens(db, client)

    assert await lens.look(10, 1, {"sticker": STICKER}) is None
    assert not await db.media_refused("sticker-uid")


async def test_a_question_about_a_retired_file_is_never_asked(db: Database) -> None:
    """The flag guards the asking path the same as the describing one."""
    await db.save_media_refusal("sticker-uid", "sticker", "eyes")
    client = FakeClient()
    bot = FakeBot()
    lens = make_lens(db, client, bot)

    assert await lens.ask(10, 1, {"sticker": STICKER}, "what is it?") is None
    assert bot.downloads == []
    assert client.responses.calls == []


async def test_a_refused_question_retires_the_file_too(db: Database) -> None:
    """Both model calls carry the same media; a no on either retires it."""
    client = FakeClient()
    client.responses.refusal = "not this one"
    lens = make_lens(db, client)

    assert await lens.ask(10, 1, {"sticker": STICKER}, "what is it?") is None
    assert await db.media_refused("sticker-uid")


async def test_media_without_eyes_is_never_looked_at(db: Database) -> None:
    """No configured model means the whole feature is simply off."""
    from libertati.config import Settings
    from libertati.prompts import load_prompts

    settings = Settings(bot_token="t", api_key="k", model="m", media_model=None)
    prompts = load_prompts(Path("prompts.toml"))
    assert (
        MediaLens.from_settings(settings, prompts, db=db, bot=cast(Bot, FakeBot()))
        is None
    )
