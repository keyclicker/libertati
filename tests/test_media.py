"""Tests for resolving, compressing and describing message media."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aiogram import Bot
from openai import AsyncOpenAI

from libertati.db import Database
from libertati.media import (
    MediaLens,
    artifact_name,
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

    async def create(self, **kwargs: Any) -> Any:
        """Return a fake Responses-API result."""
        self.calls.append(kwargs)
        self.started.set()
        if self.delay:
            await asyncio.sleep(self.delay)
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
    """Bot that "downloads" by writing a placeholder file."""

    def __init__(self) -> None:
        """Start with nothing downloaded."""
        self.downloads: list[str] = []

    async def download(self, file_id: str, destination: Path) -> None:
        """Record the download and leave a byte behind."""
        self.downloads.append(file_id)
        Path(destination).write_bytes(b"x")


def make_lens(
    db: Database,
    tmp_path: Path,
    client: FakeClient | None = None,
    bot: FakeBot | None = None,
    **extra: Any,
) -> MediaLens:
    """Build a lens over the real database and fake I/O."""
    lens = MediaLens(
        db=db,
        bot=cast(Bot, bot or FakeBot()),
        client=cast(AsyncOpenAI, client or FakeClient()),
        model="eyes",
        media_dir=tmp_path / "media",
        describe_prompt="describe it",
        **extra,
    )
    lens.ensure()
    return lens


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


def test_artifact_name_keeps_telegram_ids_and_tames_anything_else() -> None:
    """A file id must never be able to name a path of its choosing."""
    assert artifact_name("AgADAg-w_1", ".webp") == "AgADAg-w_1.webp"
    escaped = artifact_name("../../etc/passwd", ".webp")
    assert "/" not in escaped
    assert escaped.endswith(".webp")


def test_frames_command_tiles_the_clip_across_its_duration() -> None:
    """Three frames of a four-second gif, sampled evenly, side by side."""
    command = frames_command(Path("in.mp4"), Path("out.webp"), 3, 4.0)
    filters = command[command.index("-vf") + 1]
    assert "fps=3/4.000" in filters
    assert "tile=3x1" in filters


def test_frames_command_falls_back_to_one_still() -> None:
    """A clip of unknown length would tile into a half-empty image."""
    assert frames_command(Path("in.mp4"), Path("out.webp"), 3, 0.0) == image_command(
        Path("in.mp4"), Path("out.webp")
    )


def test_audio_command_produces_small_mono_opus() -> None:
    """Voice is kept re-transcribable, not hi-fi."""
    command = audio_command(Path("in.oga"), Path("out.ogg"))
    assert "libopus" in command
    assert command[command.index("-ac") + 1] == "1"


# ==========================================================
#                       Describing
# ==========================================================


async def test_a_file_is_described_once_and_read_back_forever(
    db: Database, tmp_path: Path
) -> None:
    """The second look at the same sticker costs nothing."""
    client = FakeClient()
    bot = FakeBot()
    lens = make_lens(db, tmp_path, client, bot)
    # Standing in for an artifact an earlier look already compressed.
    (lens.media_dir / artifact_name("sticker-uid", ".webp")).write_bytes(b"webp")

    first = await lens.look(10, 1, {"sticker": STICKER})
    second = await lens.look(10, 2, {"sticker": STICKER})

    assert first == "a cat knocking a mug over"
    assert second == first
    assert len(client.responses.calls) == 1
    assert bot.downloads == []
    assert await db.media_note("sticker-uid") == first


async def test_a_described_message_is_linked_to_its_file(
    db: Database, tmp_path: Path
) -> None:
    """The link is the join a transcript renders the note through."""
    lens = make_lens(db, tmp_path)
    (lens.media_dir / artifact_name("sticker-uid", ".webp")).write_bytes(b"webp")
    await db.conn.execute(
        "INSERT INTO chats (id, type, raw) VALUES (10, 'private', '{}')"
    )
    await db.conn.execute(
        """
        INSERT INTO messages (chat_id, message_id, date, content_type, raw)
        VALUES (10, 1, '2026-08-04T05:46:31+00:00', 'sticker', '{}')
        """
    )

    await lens.look(10, 1, {"sticker": STICKER})

    row = await db.message_row(10, 1)
    assert row is not None
    assert row["media_note"] == "a cat knocking a mug over"


async def test_a_note_is_folded_and_capped(db: Database, tmp_path: Path) -> None:
    """Notes live inside one transcript line and are paid for per turn."""
    client = FakeClient("a very\nlong\nanswer " + "x" * 200)
    lens = make_lens(db, tmp_path, client, note_chars=30)
    (lens.media_dir / artifact_name("sticker-uid", ".webp")).write_bytes(b"webp")

    note = await lens.look(10, 1, {"sticker": STICKER})

    assert note is not None
    assert "\n" not in note
    assert len(note) <= 31  # the cap plus the ellipsis marking the cut
    assert note.startswith("a very long answer")


async def test_an_empty_answer_is_not_stored(db: Database, tmp_path: Path) -> None:
    """A model that said nothing must not poison the cache with it."""
    lens = make_lens(db, tmp_path, FakeClient(""))
    (lens.media_dir / artifact_name("sticker-uid", ".webp")).write_bytes(b"webp")

    assert await lens.look(10, 1, {"sticker": STICKER}) is None
    assert await db.media_note("sticker-uid") is None


async def test_voice_needs_a_configured_transcription_model(
    db: Database, tmp_path: Path
) -> None:
    """Without one, voice messages stay undescribed instead of failing."""
    client = FakeClient(audio="see you at six")
    lens = make_lens(db, tmp_path, client)
    voice = {"voice": {"file_id": "v", "file_unique_id": "v-uid", "duration": 3}}

    assert await lens.look(10, 1, voice) is None
    assert client.audio.transcriptions.calls == []


async def test_voice_is_transcribed_when_a_model_is_configured(
    db: Database, tmp_path: Path
) -> None:
    """The transcript is the note, stored like any other."""
    client = FakeClient(audio="see you at six")
    lens = make_lens(db, tmp_path, client, transcribe_model="ears")
    (lens.media_dir / artifact_name("v-uid", ".ogg")).write_bytes(b"opus")

    note = await lens.look(
        10, 1, {"voice": {"file_id": "v", "file_unique_id": "v-uid", "duration": 3}}
    )

    assert note == "see you at six"
    assert client.responses.calls == []
    # Not "text": OpenRouter rejects that format outright, and json is
    # the one every transcription endpoint offers.
    assert client.audio.transcriptions.calls[0]["response_format"] == "json"


async def test_a_slow_description_does_not_hold_up_the_event(
    db: Database, tmp_path: Path
) -> None:
    """The event goes out bare; the note lands for every later transcript."""
    client = FakeClient("a cat", delay=0.05)
    lens = make_lens(db, tmp_path, client, wait_seconds=0.0)
    (lens.media_dir / artifact_name("sticker-uid", ".webp")).write_bytes(b"webp")

    job = lens.start(10, 1, {"sticker": STICKER})
    assert await lens.wait_briefly(job) is None

    await client.responses.started.wait()
    await asyncio.gather(*lens._tasks)
    assert await db.media_note("sticker-uid") == "a cat"


async def test_nothing_is_started_for_a_message_without_media(
    db: Database, tmp_path: Path
) -> None:
    """There is no job to wait for, and none was queued."""
    lens = make_lens(db, tmp_path)
    assert lens.start(10, 1, {"text": "hi"}) is None
    assert await lens.wait_briefly(None) is None
    assert lens._tasks == set()


async def test_a_flood_of_media_is_dropped_rather_than_queued(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backlog nobody will read is a bill nobody meant to pay."""
    monkeypatch.setattr("libertati.media.MAX_PENDING_JOBS", 1)
    client = FakeClient("a cat", delay=0.05)
    lens = make_lens(db, tmp_path, client)
    (lens.media_dir / artifact_name("sticker-uid", ".webp")).write_bytes(b"webp")

    first = lens.start(10, 1, {"sticker": STICKER})
    second = lens.start(10, 2, {"sticker": STICKER})

    assert first is not None
    assert second is None
    await asyncio.gather(*lens._tasks)


async def test_a_files_lock_is_forgotten_once_nobody_holds_it(
    db: Database, tmp_path: Path
) -> None:
    """The table tracks work in flight, not every file ever seen."""
    lens = make_lens(db, tmp_path, FakeClient("a cat", delay=0.02))
    (lens.media_dir / artifact_name("sticker-uid", ".webp")).write_bytes(b"webp")

    await asyncio.gather(
        lens.look(10, 1, {"sticker": STICKER}),
        lens.look(10, 2, {"sticker": STICKER}),
    )

    assert lens._locks == {}


async def test_media_without_eyes_is_never_looked_at(
    db: Database, tmp_path: Path
) -> None:
    """No configured model means the whole feature is simply off."""
    from libertati.config import Settings
    from libertati.prompts import load_prompts

    settings = Settings(bot_token="t", api_key="k", model="m", media_model=None)
    prompts = load_prompts(Path("prompts.toml"))
    assert (
        MediaLens.from_settings(settings, prompts, db=db, bot=cast(Bot, FakeBot()))
        is None
    )


async def test_an_artifact_is_published_only_once_it_is_whole(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything in media_dir is described unseen, so it must be complete.

    ffmpeg therefore never writes there: it fills a temporary the caller
    renames into place, and a run killed halfway leaves the work undone
    rather than a truncated picture every later look would trust.
    """
    lens = make_lens(db, tmp_path)
    artifact = lens.media_dir / artifact_name("sticker-uid", ".webp")
    targets: list[Path] = []

    async def fake_run(command: list[str]) -> bytes | None:
        """Encode into whatever the command was told to write."""
        target = Path(command[-1])
        targets.append(target)
        target.write_bytes(b"webp")
        return b""

    monkeypatch.setattr("libertati.media._run", fake_run)
    assert await lens.look(10, 1, {"sticker": STICKER}) == "a cat knocking a mug over"

    assert targets == [lens._tmp_dir / artifact.name]
    assert artifact.exists()
    assert list(lens._tmp_dir.iterdir()) == []
