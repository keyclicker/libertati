"""Cheap media understanding: one short note per file, cached forever.

Nothing binary ever reaches the agent's context. A picture, sticker, gif,
video or voice message is turned into a sentence or two by a separate,
deliberately cheap model call, and that text is what the transcript shows
where it would otherwise print a bare ``<photo>``. Notes are keyed by
Telegram's ``file_unique_id``, so the sticker a group spams all day is
described once and free forever after.

Frames are the poor man's video support: a few stills tiled into one
small image, which any vision model reads — no native video input, no
whole file uploaded anywhere. Voice notes take the other branch and go to
a speech-to-text endpoint.

Only compressed derivatives are kept (under ``media_dir``): one small
webp per picture or frame strip, one low-bitrate opus per voice message.
Originals live in ``tmp/`` for exactly as long as ffmpeg needs them, and
so does the artifact until it is whole — what lands in ``media_dir`` is
described without a second look, so it may never be half-written.
"""

import asyncio
import base64
import hashlib
import logging
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogram import Bot
from openai import AsyncOpenAI

from libertati.config import Settings
from libertati.db import Database
from libertati.prompts import Prompts

log = logging.getLogger(__name__)

#: Width of every stored frame. Small on purpose: the describer is asked
#: what is going on, not to read fine print, and a low-detail image costs
#: a fraction of a full-size one on every provider that prices by tile.
FRAME_WIDTH = 320

#: webp quality of the stored artifact (0-100, ffmpeg scale).
FRAME_QUALITY = 60

#: Mono opus bitrate kept for voice/audio. Speech survives it; the file
#: is a tenth of the original and stays re-transcribable.
AUDIO_BITRATE = "16k"

#: Telegram refuses ``getFile`` past 20 MB, so anything bigger cannot be
#: looked at however hard we try.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

#: Hard limit on one ffmpeg/ffprobe run. A malformed video must not pin a
#: worker forever.
FFMPEG_TIMEOUT = 60.0

#: Media jobs allowed to run at once. ffmpeg is the expensive part and
#: this bot is expected to live on small hardware.
CONCURRENT_JOBS = 2

#: Descriptions allowed to be waiting for a slot at once. Two jobs run
#: while the rest queue, so a group dumping an album a second builds a
#: backlog nobody will still care about by the time it drains — and one
#: model call each. Past this, media is left undescribed until the queue
#: empties; ``look_at_media`` still covers whatever was skipped.
MAX_PENDING_JOBS = 64

#: Message fields carrying media, in the order a message is inspected.
#: Documents are deliberately absent: they are arbitrary files, often
#: large, and rarely worth a model call.
_VIDEO_KINDS = ("animation", "video", "video_note")
_AUDIO_KINDS = ("voice", "audio")

#: Telegram content types :func:`media_ref` can resolve. A message of
#: any other kind is not worth serializing a payload to find that out —
#: the handler sees one per message, media or not.
MEDIA_CONTENT_TYPES = frozenset({"photo", "sticker", *_VIDEO_KINDS, *_AUDIO_KINDS})


@dataclass(frozen=True)
class MediaRef:
    """One media file worth describing, resolved from a message payload.

    ``file_id`` is what gets downloaded and ``file_unique_id`` is the
    cache identity — the two differ for a photo (the variant we fetch is
    not the one that names the photo) and for animated stickers (the
    downloaded file is Telegram's own preview).
    """

    kind: str
    file_id: str
    file_unique_id: str
    #: How the artifact is built: ``image``, ``video`` or ``audio``.
    source: str
    #: What the file is, in words, for the describer's prompt.
    hint: str
    size: int = 0


# ==========================================================
#                    Reading a payload
# ==========================================================


def _photo_ref(variants: list[dict[str, Any]]) -> MediaRef | None:
    """Pick the photo variant to download and the one that names it.

    Telegram serves a photo pre-resized, which is the whole image
    pipeline for free: the smallest variant at least :data:`FRAME_WIDTH`
    wide is already the artifact. Identity is the largest variant's
    ``file_unique_id`` — variants have one each, and the largest is the
    only one every message of that photo is guaranteed to carry.
    """
    sizes = [item for item in variants if item.get("file_id")]
    if not sizes:
        return None
    sizes.sort(key=lambda item: item.get("width") or 0)
    largest = sizes[-1]
    wanted = next(
        (item for item in sizes if (item.get("width") or 0) >= FRAME_WIDTH), largest
    )
    return MediaRef(
        kind="photo",
        file_id=wanted["file_id"],
        file_unique_id=largest["file_unique_id"],
        source="image",
        hint="a photo",
        size=wanted.get("file_size") or 0,
    )


def _sticker_ref(sticker: dict[str, Any]) -> MediaRef | None:
    """Resolve a sticker, routing each of its three formats sensibly.

    Video stickers (webm) go through ffmpeg like any other clip. Animated
    ones are Lottie vectors nothing here can render, so their Telegram
    thumbnail stands in — one still of an animation is what the format
    affords. Identity stays the sticker's own id either way, so the
    thumbnail never becomes a second cache entry.
    """
    hints = ["a sticker"]
    if emoji := sticker.get("emoji"):
        hints.append(f"for the emoji {emoji}")
    if set_name := sticker.get("set_name"):
        hints.append(f"from the set “{set_name}”")
    file_id = sticker.get("file_id")
    source = "image"
    size = sticker.get("file_size") or 0
    if sticker.get("is_video"):
        source = "video"
        hints.append("(animated)")
    elif sticker.get("is_animated"):
        thumbnail = sticker.get("thumbnail") or {}
        file_id = thumbnail.get("file_id")
        size = thumbnail.get("file_size") or 0
        hints.append("(animated; this is one still of it)")
    if not file_id:
        return None
    return MediaRef(
        kind="sticker",
        file_id=file_id,
        file_unique_id=sticker["file_unique_id"],
        source=source,
        hint=" ".join(hints),
        size=size,
    )


def _clip_ref(kind: str, clip: dict[str, Any]) -> MediaRef | None:
    """Resolve a gif, video or round video note as a frame source."""
    if not clip.get("file_id"):
        return None
    names = {
        "animation": "a silent gif",
        "video": "a video",
        "video_note": "a round video note",
    }
    hint = names[kind]
    if duration := clip.get("duration"):
        hint += f", {duration}s long"
    return MediaRef(
        kind=kind,
        file_id=clip["file_id"],
        file_unique_id=clip["file_unique_id"],
        source="video",
        hint=hint,
        size=clip.get("file_size") or 0,
    )


def _sound_ref(kind: str, sound: dict[str, Any]) -> MediaRef | None:
    """Resolve a voice message or music file as a transcription source."""
    if not sound.get("file_id"):
        return None
    hint = "a voice message" if kind == "voice" else "an audio file"
    if duration := sound.get("duration"):
        hint += f", {duration}s long"
    for key in ("performer", "title", "file_name"):
        if value := sound.get(key):
            hint += f" ({value})"
            break
    return MediaRef(
        kind=kind,
        file_id=sound["file_id"],
        file_unique_id=sound["file_unique_id"],
        source="audio",
        hint=hint,
        size=sound.get("file_size") or 0,
    )


def media_ref(payload: dict[str, Any]) -> MediaRef | None:
    """Return the media a stored message carries, or ``None``.

    Works on the raw Telegram payload — the JSON in ``messages.raw`` and
    the dump of a live :class:`~aiogram.types.Message` alike — so a
    message coming off the wire and one read back years later resolve
    through the same code.
    """
    if photo := payload.get("photo"):
        return _photo_ref(photo)
    if sticker := payload.get("sticker"):
        return _sticker_ref(sticker)
    for kind in _VIDEO_KINDS:
        if clip := payload.get(kind):
            return _clip_ref(kind, clip)
    for kind in _AUDIO_KINDS:
        if sound := payload.get(kind):
            return _sound_ref(kind, sound)
    return None


# ==========================================================
#                      Building frames
# ==========================================================


def artifact_name(file_unique_id: str, suffix: str) -> str:
    """Build the on-disk name of one file's artifact.

    Telegram's ids are URL-safe base64 and land unchanged; anything else
    is reduced to a hash, so an unexpected id can never name a path
    outside the media directory.
    """
    safe = re.sub(r"[^A-Za-z0-9_-]", "", file_unique_id)
    if safe != file_unique_id or not safe:
        digest = hashlib.sha256(file_unique_id.encode("utf-8")).hexdigest()[:16]
        safe = f"{safe[:16]}-{digest}" if safe else digest
    return f"{safe}{suffix}"


def image_command(source: Path, target: Path) -> list[str]:
    """Build the ffmpeg call turning one picture into the stored webp."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={FRAME_WIDTH}:-2:force_original_aspect_ratio=decrease",
        "-frames:v",
        "1",
        "-quality",
        str(FRAME_QUALITY),
        str(target),
    ]


def frames_command(
    source: Path, target: Path, frames: int, duration: float
) -> list[str]:
    """Build the ffmpeg call tiling a clip's frames into one webp.

    Frames are sampled at an even rate across the clip and laid out left
    to right, so one image carries the whole thing. A clip whose duration
    is unknown (or too short to sample twice) contributes its first frame
    alone rather than risking a half-empty tile.
    """
    if frames < 2 or duration <= 0:
        return image_command(source, target)
    scale = f"scale={FRAME_WIDTH}:-2:force_original_aspect_ratio=decrease"
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"fps={frames}/{duration:.3f},{scale},tile={frames}x1",
        "-frames:v",
        "1",
        "-quality",
        str(FRAME_QUALITY),
        str(target),
    ]


def audio_command(source: Path, target: Path) -> list[str]:
    """Build the ffmpeg call re-encoding sound to small mono opus."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        AUDIO_BITRATE,
        str(target),
    ]


async def _run(command: list[str]) -> bytes | None:
    """Run one ffmpeg-family command; return its stdout, or ``None``.

    Failures are logged and swallowed: a file that will not decode is a
    file without a description, never an exception reaching a handler.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        log.exception("could not start %s", command[0])
        return None
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=FFMPEG_TIMEOUT
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        log.warning("%s timed out", command[0])
        return None
    if process.returncode != 0:
        log.warning(
            "%s failed (%s): %s",
            command[0],
            process.returncode,
            stderr.decode("utf-8", "replace").strip().splitlines()[-1:],
        )
        return None
    return stdout


async def probe_duration(source: Path) -> float:
    """Return a media file's duration in seconds, or 0 when unknown."""
    output = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ]
    )
    try:
        return max(0.0, float((output or b"").decode().strip()))
    except ValueError:
        return 0.0


# ==========================================================
#                       Describing
# ==========================================================


class MediaLens:
    """Turns media files into cached text notes, and never raises.

    One instance is shared by the bot handlers (which describe what
    arrives) and the ``look_at_media`` tool (which describes what the
    agent asks about). Both land in the same per-file lock, so a picture
    forwarded into three chats at once is still described once.
    """

    def __init__(
        self,
        *,
        db: Database,
        bot: Bot,
        client: AsyncOpenAI,
        model: str,
        media_dir: Path,
        describe_prompt: str,
        transcribe_model: str | None = None,
        max_frames: int = 3,
        note_chars: int = 220,
        wait_seconds: float = 6.0,
    ) -> None:
        """Keep the handles and the (empty) per-file lock table."""
        self.db = db
        self.bot = bot
        self.client = client
        self.model = model
        self.media_dir = media_dir
        self.describe_prompt = describe_prompt
        self.transcribe_model = transcribe_model
        self.max_frames = max_frames
        self.note_chars = note_chars
        self.wait_seconds = wait_seconds
        #: Per-file lock and how many jobs are holding or awaiting it.
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self._jobs = asyncio.Semaphore(CONCURRENT_JOBS)
        # Background describe tasks are held here for their lifetime:
        # asyncio only keeps weak references, and a collected task is a
        # description that silently never happened.
        self._tasks: set[asyncio.Task[str | None]] = set()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        prompts: Prompts,
        *,
        db: Database,
        bot: Bot,
    ) -> "MediaLens | None":
        """Build the lens, or ``None`` when no media model is configured.

        Without a model there is nothing to fall back to — the feature is
        simply off, transcripts keep saying ``<photo>``, and every caller
        skips the whole path on the ``None``.
        """
        if not settings.media_model:
            return None
        return cls(
            db=db,
            bot=bot,
            client=AsyncOpenAI(
                api_key=settings.media_api_key or settings.api_key,
                base_url=settings.media_base_url or settings.base_url,
            ),
            model=settings.media_model,
            media_dir=settings.media_dir,
            describe_prompt=prompts.media_describe,
            transcribe_model=settings.transcribe_model or None,
            max_frames=settings.media_max_frames,
            note_chars=settings.media_note_chars,
            wait_seconds=settings.media_wait_seconds,
        )

    def ensure(self) -> None:
        """Create the media directories, emptying leftover temporaries.

        Nothing in ``tmp/`` is meant to outlive the job that put it
        there — an original waiting for ffmpeg, an artifact waiting to be
        whole — so whatever is still there is debris from a killed
        process.
        """
        self.media_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _tmp_dir(self) -> Path:
        """Directory holding a job's files until they are wanted."""
        return self.media_dir / "tmp"

    # ---------------------- entry points ----------------------

    async def look(
        self, chat_id: int, message_id: int, payload: dict[str, Any]
    ) -> str | None:
        """Describe one message's media, waiting for the work to finish.

        Returns the note, or ``None`` when the message carries nothing
        describable, the file is out of reach or the model call failed.
        Never raises: callers are event handlers and tool handlers.
        """
        ref = media_ref(payload)
        if ref is None:
            return None
        try:
            return await self._note(chat_id, message_id, ref)
        except Exception:
            log.exception("describing %s in chat %s failed", ref.kind, chat_id)
            return None

    def start(
        self, chat_id: int, message_id: int, payload: dict[str, Any]
    ) -> "asyncio.Task[str | None] | None":
        """Describe one message's media in the background, if it has any.

        Returns the running job so a caller that wants the note can wait
        for it with :meth:`wait_briefly`; the work is started either way,
        the moment the message lands, so a burst of pictures is looked at
        all at once and the note is there when any of them later rides
        along with an event as context.
        """
        if media_ref(payload) is None:
            return None
        if len(self._tasks) >= MAX_PENDING_JOBS:
            log.warning(
                "media backlog full (%d jobs); leaving msg %s in chat %s undescribed",
                len(self._tasks),
                message_id,
                chat_id,
            )
            return None
        task = asyncio.create_task(self.look(chat_id, message_id, payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def wait_briefly(self, job: "asyncio.Task[str | None] | None") -> str | None:
        """Give a started description ``wait_seconds`` to land.

        Used on the path an event takes: a note that arrives in time is
        worth waiting a moment for, but a slow model must not hold up the
        agent's reply. The job carries on in the background either way,
        so the note is there for every later transcript.
        """
        if job is None:
            return None
        done, _ = await asyncio.wait({job}, timeout=self.wait_seconds)
        return job.result() if done else None

    # ------------------------- work --------------------------

    @asynccontextmanager
    async def _file_lock(self, file_unique_id: str) -> AsyncIterator[None]:
        """Hold one file's lock, forgetting it once nobody wants it.

        The lock is what keeps a sticker forwarded into three chats at
        once from being described three times. Counting holders rather
        than leaving the entry behind keeps the table the size of the
        work in flight, not of every file the bot has ever seen.
        """
        lock, holders = self._locks.get(file_unique_id, (asyncio.Lock(), 0))
        self._locks[file_unique_id] = (lock, holders + 1)
        try:
            async with lock:
                yield
        finally:
            lock, holders = self._locks[file_unique_id]
            if holders > 1:
                self._locks[file_unique_id] = (lock, holders - 1)
            else:
                del self._locks[file_unique_id]

    async def _note(self, chat_id: int, message_id: int, ref: MediaRef) -> str | None:
        """Return a file's note, describing it once if nobody has yet."""
        async with self._file_lock(ref.file_unique_id):
            note = await self.db.media_note(ref.file_unique_id)
            if note is None:
                note = await self._describe(ref)
                if note is not None:
                    await self.db.save_media_note(
                        ref.file_unique_id, ref.kind, note, self._model_for(ref)
                    )
        if note is not None:
            # Only now does the message's transcript line have something
            # to join to; writing the link earlier would show a note for
            # a file the description of which never arrived.
            await self.db.set_message_media(chat_id, message_id, ref.file_unique_id)
        return note

    def _model_for(self, ref: MediaRef) -> str:
        """Name the model that produced (or would produce) a note."""
        if ref.source == "audio":
            return self.transcribe_model or ""
        return self.model

    async def _describe(self, ref: MediaRef) -> str | None:
        """Build the artifact if needed, then ask a model what it is."""
        if ref.source == "audio" and not self.transcribe_model:
            return None
        if ref.size > MAX_DOWNLOAD_BYTES:
            log.info(
                "skipping %s: %d bytes is past Telegram's limit", ref.kind, ref.size
            )
            return None
        # Named by container, not codec: a transcription endpoint reads
        # the format off the filename it is handed, and ".ogg" is on
        # every provider's list where ".opus" is on few.
        suffix = ".ogg" if ref.source == "audio" else ".webp"
        artifact = self.media_dir / artifact_name(ref.file_unique_id, suffix)
        async with self._jobs:
            # An artifact that survived from an earlier look is described
            # again without touching Telegram or ffmpeg.
            if not artifact.exists() and not await self._build(ref, artifact):
                return None
            if ref.source == "audio":
                return await self._transcribe(artifact)
            return await self._describe_image(ref, artifact)

    async def _build(self, ref: MediaRef, artifact: Path) -> bool:
        """Download the original and compress it into ``artifact``.

        Both intermediate files live in ``tmp/`` and neither survives the
        call: the original because it is the one file here nobody wants
        on disk, the half-encoded artifact because everything in
        ``media_dir`` is trusted and described without a second look. A
        process killed mid-encode therefore loses the work, rather than
        leaving behind a truncated picture it would describe forever.
        """
        original = self._tmp_dir / artifact_name(ref.file_unique_id, ".bin")
        staged = self._tmp_dir / artifact.name
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            await self.bot.download(ref.file_id, destination=original)
            if ref.source == "video":
                duration = await probe_duration(original)
                command = frames_command(original, staged, self.max_frames, duration)
            elif ref.source == "audio":
                command = audio_command(original, staged)
            else:
                command = image_command(original, staged)
            if await _run(command) is None or not staged.exists():
                return False
            staged.replace(artifact)
        finally:
            original.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)
        return artifact.exists()

    async def _describe_image(self, ref: MediaRef, artifact: Path) -> str | None:
        """Ask the vision model to put one small image into words."""
        data = base64.b64encode(artifact.read_bytes()).decode("ascii")
        hint = ref.hint
        if ref.source == "video":
            hint += (
                f" — the image is up to {self.max_frames} frames of it,"
                " tiled left to right in time order"
            )
        response = await self.client.responses.create(
            model=self.model,
            instructions=self.describe_prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"This is {hint}."},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/webp;base64,{data}",
                            # Cheapest tier every provider offers: enough
                            # to say what is going on at this size.
                            "detail": "low",
                        },
                    ],
                }
            ],
            store=False,
        )
        return self._clean(response.output_text)

    async def _transcribe(self, artifact: Path) -> str | None:
        """Send one voice/audio artifact to the speech-to-text endpoint.

        ``json`` rather than ``text``: it is the one response format
        every transcription endpoint offers (OpenRouter rejects ``text``
        outright), and the answer is read the same either way — some
        compatible endpoints hand back the bare string regardless.
        """
        response = await self.client.audio.transcriptions.create(
            model=self.transcribe_model or "",
            file=(artifact.name, artifact.read_bytes(), "audio/ogg"),
            response_format="json",
        )
        text = response if isinstance(response, str) else getattr(response, "text", "")
        return self._clean(text)

    def _clean(self, text: str | None) -> str | None:
        """Fold and cap a model's answer into a storable note.

        Notes are read back into a line-oriented transcript, so folding is
        not cosmetic: a note is stored as one line because the format has
        no way to say where a second one would end.
        """
        note = " ".join((text or "").split())
        if not note:
            return None
        if len(note) > self.note_chars:
            note = note[: self.note_chars].rstrip() + "…"
        return note
