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

A model may refuse a file — its provider's content policy, not ours.
A refusal is remembered (``media_refusals``) and is final: that file is
never downloaded or sent to a model again, so the provider sees any
given file at most once. Only an actual no counts — a timeout or a
server error is not a refusal and stays retryable.

Nothing binary touches disk either. A file is downloaded into memory,
handed to ffmpeg as an anonymous in-memory file, and sent to the model
as the bytes ffmpeg writes back down its stdout; when the note lands,
the bytes are gone. What survives a look is text in the database — a
machine this bot runs on never holds a stranger's picture at rest. A
second look at the same file (a ``look_at_media`` question) downloads it
again; Telegram keeps the original, so the ``file_id`` is the cache.
"""

import asyncio
import base64
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

from aiogram import Bot
from openai import AsyncOpenAI, BadRequestError

from libertati.config import Settings
from libertati.db import Database
from libertati.prompts import Prompts

log = logging.getLogger(__name__)

#: Width of every frame sent to the model. Small on purpose: the
#: describer is asked what is going on, not to read fine print, and a
#: low-detail image costs a fraction of a full-size one on every
#: provider that prices by tile.
FRAME_WIDTH = 320

#: webp quality of the encoded frame (0-100, ffmpeg scale).
FRAME_QUALITY = 60

#: Mono opus bitrate voice/audio is sent at. Speech survives it; the
#: upload is a tenth of the original and stays transcribable.
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
    #: How the upload is built: ``image``, ``video`` or ``audio``.
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
    wide is already frame-sized. Identity is the largest variant's
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

# Every command reads the original out of an anonymous in-memory file
# and writes the result to stdout: files on disk are what this module is
# built to avoid, but a *seekable* input is not optional. A pipe is not
# seekable, and an mp4 whose moov atom sits at the end — which is most
# of what people upload, Telegram stores it as sent — cannot be demuxed
# without one. So the descriptor goes to the child instead of the bytes.


@contextmanager
def in_memory_file(data: bytes) -> Iterator[int]:
    """Hold ``data`` in an anonymous file and yield its descriptor.

    A memfd lives in RAM, has no name in any directory and is gone when
    the last descriptor closes — so ffmpeg gets something it can seek in
    without a stranger's file ever existing on disk. The child reads it
    by path (:func:`source_path`) rather than by inheriting our offset,
    so several runs over the same original are free and independent.
    """
    fd = os.memfd_create("libertati-media", os.MFD_CLOEXEC)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        yield fd
    finally:
        os.close(fd)


def source_path(source: int) -> str:
    """Name a descriptor as the path the ffmpeg child opens it by.

    ``pass_fds`` leaves the descriptor at the same number in the child,
    where ``/proc/self/fd`` resolves to the child's own table — so this
    path means the memfd there, opened afresh at offset zero.
    """
    return f"/proc/self/fd/{source}"


def image_command(source: int) -> list[str]:
    """Build the ffmpeg call turning one picture into a small webp."""
    return [
        "ffmpeg",
        "-i",
        source_path(source),
        "-vf",
        f"scale={FRAME_WIDTH}:-2:force_original_aspect_ratio=decrease",
        "-frames:v",
        "1",
        "-c:v",
        "libwebp",
        "-quality",
        str(FRAME_QUALITY),
        "-f",
        "image2pipe",
        "pipe:1",
    ]


def frames_command(source: int, frames: int, duration: float) -> list[str]:
    """Build the ffmpeg call tiling a clip's frames into one webp.

    Frames are sampled at an even rate across the clip and laid out left
    to right, so one image carries the whole thing. A clip whose duration
    is unknown (or too short to sample twice) contributes its first frame
    alone rather than risking a half-empty tile.
    """
    if frames < 2 or duration <= 0:
        return image_command(source)
    scale = f"scale={FRAME_WIDTH}:-2:force_original_aspect_ratio=decrease"
    return [
        "ffmpeg",
        "-i",
        source_path(source),
        "-vf",
        f"fps={frames}/{duration:.3f},{scale},tile={frames}x1",
        "-frames:v",
        "1",
        "-c:v",
        "libwebp",
        "-quality",
        str(FRAME_QUALITY),
        "-f",
        "image2pipe",
        "pipe:1",
    ]


def audio_command(source: int) -> list[str]:
    """Build the ffmpeg call re-encoding sound to small mono opus."""
    return [
        "ffmpeg",
        "-i",
        source_path(source),
        "-vn",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        AUDIO_BITRATE,
        "-f",
        "ogg",
        "pipe:1",
    ]


async def _run(command: list[str], source: int) -> bytes | None:
    """Run one ffmpeg-family command over ``source``; stdout, or ``None``.

    The command already names the descriptor as a path; handing the
    descriptor itself to the child (``pass_fds``) is what makes that path
    resolve there. Failures are logged and swallowed: a file that will
    not decode is a file without a description, never an exception
    reaching a handler.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            pass_fds=(source,),
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


async def probe_duration(source: int) -> float:
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
            source_path(source),
        ],
        source,
    )
    try:
        return max(0.0, float((output or b"").decode().strip()))
    except ValueError:
        return 0.0


# ==========================================================
#                       Refusals
# ==========================================================


class _Refusal(Exception):
    """The model declined to look at a file, as a matter of policy."""


#: The machine-readable labels a provider gives a refusal. Matched
#: whole, against the error's code and type only.
_POLICY_CODES = frozenset(
    {
        "content_policy_violation",
        "content_filter",
        "invalid_prompt",
        "moderation_blocked",
        "prompt_blocked",
    }
)

#: What marks the *message* of an uncoded 400 as the provider saying no
#: rather than us asking wrong. Every one is a phrase, and only the
#: message is searched: a bare word tested against the whole error reads
#: "safety" out of a rejected ``safety_identifier`` parameter and retires
#: an innocent file for good, and nothing ever clears a refusal.
_POLICY_PHRASES = (
    "content policy",
    "content management policy",
    "content filter",
    "usage policy",
    "usage policies",
    "safety system",
    "flagged as",
)


def _error_labels(error: BadRequestError) -> set[str]:
    """Collect the codes a 400 carries, from wherever it carries them.

    The client fills ``code`` from an OpenAI-shaped body; a provider
    that answers in its own shape leaves it empty and names the reason
    inside the body instead.
    """
    labels = {str(getattr(error, "code", "") or "")}
    body = getattr(error, "body", None)
    detail = body.get("error") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        labels |= {str(detail.get(key) or "") for key in ("code", "type")}
    return {label.lower() for label in labels if label}


def _policy_error(error: BadRequestError) -> bool:
    """Whether a 400 refuses the content instead of the request.

    A plain bad request is our bug and must stay retryable; only a
    policy no retires the file for good, so this errs towards no.
    """
    if _error_labels(error) & _POLICY_CODES:
        return True
    message = str(getattr(error, "message", "") or "").lower()
    return any(phrase in message for phrase in _POLICY_PHRASES)


def _refusal_reason(response: Any) -> str | None:
    """Return the refusal a response carries in place of an answer.

    The Responses API marks a refusal as its own content part rather
    than a status: a message item whose content holds a ``refusal``
    part instead of (or beside) the usual ``output_text``.
    """
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return getattr(part, "refusal", None) or "refused"
    return None


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
        describe_prompt: str,
        answer_prompt: str = "",
        transcribe_model: str | None = None,
        max_frames: int = 3,
        note_chars: int = 220,
        answer_chars: int = 700,
        wait_seconds: float = 6.0,
    ) -> None:
        """Keep the handles and the (empty) per-file lock table."""
        self.db = db
        self.bot = bot
        self.client = client
        self.model = model
        self.describe_prompt = describe_prompt
        self.answer_prompt = answer_prompt
        self.transcribe_model = transcribe_model
        self.max_frames = max_frames
        self.note_chars = note_chars
        self.answer_chars = answer_chars
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
        skips the whole path on the ``None``. The describer talks to the
        same endpoint with the same key as the agent itself: one route,
        so a model named here is one the configured provider serves.
        """
        if not settings.media_model:
            return None
        return cls(
            db=db,
            bot=bot,
            client=AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url),
            model=settings.media_model,
            describe_prompt=prompts.media_describe,
            answer_prompt=prompts.media_answer,
            transcribe_model=settings.transcribe_model or None,
            max_frames=settings.media_max_frames,
            note_chars=settings.media_note_chars,
            answer_chars=settings.media_answer_chars,
            wait_seconds=settings.media_wait_seconds,
        )

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

    async def ask(
        self, chat_id: int, message_id: int, payload: dict[str, Any], question: str
    ) -> str | None:
        """Answer one question about a message's media, uncached.

        The stored note is one line an agent never chose the shape of;
        this is the second look it can ask for — what the sign says, what
        breed the dog is, which frame the cup falls in. The answer is not
        written to ``media_notes``: it answers a question rather than
        describing the file, and every later transcript would carry it.
        The file is fetched from Telegram anew — nothing of the first
        look was kept to reuse.

        A file already retired answers nothing, but a refusal here does
        not retire one: the model was asked a question about the file,
        and a no can be about either.

        Audio has no such second look — a transcription endpoint takes no
        question — so a voice message answers with its transcript, which
        is everything there is to know about it.
        """
        ref = media_ref(payload)
        if ref is None:
            return None
        try:
            if ref.source == "audio":
                return await self._note(chat_id, message_id, ref)
            async with self._file_lock(ref.file_unique_id):
                # Inside the lock, like the describing path's check: a
                # look and a question about the same file run together
                # often enough, and reading the flag outside means
                # downloading and sending a file the look just had
                # refused.
                if await self.db.media_refused(ref.file_unique_id):
                    return None
                async with self._jobs:
                    upload = await self._build(ref)
                    if upload is None:
                        return None
                try:
                    return await self._describe_image(ref, upload, question=question)
                except _Refusal as refusal:
                    # The call carried the agent's question as well as
                    # the file, so a no here does not name the file as
                    # the reason — "who is the person in this photo?"
                    # is refused over the asking. Retiring on it would
                    # cost the file its description, permanently, for a
                    # question nobody has to ask twice.
                    log.info(
                        "model would not answer about %s (%s): %s",
                        ref.kind,
                        ref.file_unique_id,
                        refusal,
                    )
                    return None
        except Exception:
            log.exception("answering about %s in chat %s failed", ref.kind, chat_id)
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
                if await self.db.media_refused(ref.file_unique_id):
                    return None
                try:
                    note = await self._describe(ref)
                except _Refusal as refusal:
                    await self._flag_refusal(ref, str(refusal))
                    return None
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

    async def _flag_refusal(self, ref: MediaRef, reason: str) -> None:
        """Retire a file a model said no to; it is never sent again."""
        log.warning(
            "model refused %s (%s), retiring it: %s",
            ref.kind,
            ref.file_unique_id,
            reason,
        )
        await self.db.save_media_refusal(
            ref.file_unique_id, ref.kind, self._model_for(ref)
        )

    async def _describe(self, ref: MediaRef) -> str | None:
        """Fetch and compress the file, then ask a model what it is."""
        if ref.source == "audio" and not self.transcribe_model:
            return None
        async with self._jobs:
            upload = await self._build(ref)
            if upload is None:
                return None
            if ref.source == "audio":
                return await self._transcribe(upload)
            return await self._describe_image(ref, upload)

    async def _build(self, ref: MediaRef) -> bytes | None:
        """Download the original and compress it, all in memory.

        The original exists only as an anonymous in-memory file between
        the Telegram download and the ffmpeg run; what comes back down
        ffmpeg's stdout is the small webp or opus the model is sent.
        Nothing is written to disk, so there is no half-written file to
        trust later and nothing to clean up after a crash — the memfd
        goes away with the ``with``, crash or not.

        A clip is probed and encoded off that one descriptor, so the
        bytes are held once however many runs read them.
        """
        if ref.size > MAX_DOWNLOAD_BYTES:
            log.info(
                "skipping %s: %d bytes is past Telegram's limit", ref.kind, ref.size
            )
            return None
        buffer = await self.bot.download(ref.file_id)
        original = buffer.read() if buffer is not None else b""
        if not original:
            return None
        with in_memory_file(original) as source:
            # The download's own copies are dead weight once the memfd
            # holds the bytes, and this path runs on machines where 20 MB
            # twice over is worth not carrying through an ffmpeg run.
            del buffer, original
            if ref.source == "video":
                duration = await probe_duration(source)
                command = frames_command(source, self.max_frames, duration)
            elif ref.source == "audio":
                command = audio_command(source)
            else:
                command = image_command(source)
            return await _run(command, source) or None

    async def _describe_image(
        self, ref: MediaRef, upload: bytes, question: str | None = None
    ) -> str | None:
        """Ask the vision model to put one small image into words.

        With a ``question`` the looser answering prompt is used and the
        answer may run longer: it is read once, by the agent that asked,
        rather than stored and carried by every later transcript.
        """
        data = base64.b64encode(upload).decode("ascii")
        hint = ref.hint
        if ref.source == "video":
            hint += (
                f" — the image is up to {self.max_frames} frames of it,"
                " tiled left to right in time order"
            )
        prompt = f"This is {hint}."
        if question is not None:
            # The question is the agent's own words, not a stranger's,
            # but it is still quoted rather than joined to the
            # instructions: the model is answering about the picture,
            # not taking orders from the turn that asked.
            asked = " ".join(question.split())
            prompt += f'\nAnswer this about it: "{asked}"'
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=self.answer_prompt if question else self.describe_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/webp;base64,{data}",
                                # Cheapest tier every provider offers:
                                # enough to say what is going on at this
                                # size.
                                "detail": "low",
                            },
                        ],
                    }
                ],
                store=False,
            )
        except BadRequestError as error:
            if _policy_error(error):
                raise _Refusal(str(error)) from error
            raise
        if reason := _refusal_reason(response):
            raise _Refusal(reason)
        cap = self.answer_chars if question else self.note_chars
        return self._clean(response.output_text, cap)

    async def _transcribe(self, upload: bytes) -> str | None:
        """Send one voice/audio recording to the speech-to-text endpoint.

        The upload is named ``.ogg`` rather than ``.opus`` — an endpoint
        reads the format off the filename it is handed, and ``.ogg`` is
        on every provider's list where ``.opus`` is on few. ``json``
        rather than ``text`` for the same reason: it is the one response
        format every transcription endpoint offers (OpenRouter rejects
        ``text`` outright), and the answer is read the same either way —
        some compatible endpoints hand back the bare string regardless.
        """
        try:
            response = await self.client.audio.transcriptions.create(
                model=self.transcribe_model or "",
                file=("voice.ogg", upload, "audio/ogg"),
                response_format="json",
            )
        except BadRequestError as error:
            if _policy_error(error):
                raise _Refusal(str(error)) from error
            raise
        text = response if isinstance(response, str) else getattr(response, "text", "")
        return self._clean(text)

    def _clean(self, text: str | None, cap: int | None = None) -> str | None:
        """Fold and cap a model's answer into a storable note.

        Notes are read back into a line-oriented transcript, so folding is
        not cosmetic: a note is stored as one line because the format has
        no way to say where a second one would end. An answer to a
        question is folded on the same rule though it is never stored —
        one line is what the tool result should be either way.
        """
        note = " ".join((text or "").split())
        if not note:
            return None
        limit = self.note_chars if cap is None else cap
        if len(note) > limit:
            note = note[:limit].rstrip() + "…"
        return note
