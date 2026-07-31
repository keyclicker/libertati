"""Read public Telegram channels via the t.me/s/<username> web preview.

Needs no API credentials: public channels with the web preview enabled expose
their most recent ~20 posts as plain HTML. Private chats/groups and channels
that disable the preview are not readable this way.
"""

from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
from typing import Any

import httpx

from ..logging import get_logger
from .base import IncomingMessage

log = get_logger("telegram.webpreview")


def _classes(attrs: dict[str, str | None]) -> set[str]:
    return set((attrs.get("class") or "").split())


class _PreviewParser(HTMLParser):
    """Extract posts (id, author, text, timestamp) from a t.me/s page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.posts: list[dict[str, Any]] = []
        self._div_depth = 0
        self._post: dict[str, Any] | None = None
        self._post_depth = 0
        self._text_depth: int | None = None
        self._in_name = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = _classes(a)
        if tag == "div":
            self._div_depth += 1
            if "tgme_widget_message" in cls and a.get("data-post"):
                self._post = {"post": a["data-post"], "author": None, "text": [], "ts": None}
                self._post_depth = self._div_depth
                self.posts.append(self._post)
            elif self._post is not None and "tgme_widget_message_text" in cls:
                self._text_depth = self._div_depth
        elif self._post is not None:
            if tag == "br" and self._text_depth is not None:
                self._post["text"].append("\n")
            elif tag == "time" and a.get("datetime"):
                self._post["ts"] = a["datetime"]
            elif tag == "a" and "tgme_widget_message_owner_name" in cls:
                self._in_name = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            if self._text_depth is not None and self._div_depth == self._text_depth:
                self._text_depth = None
            if self._post is not None and self._div_depth == self._post_depth:
                self._post = None
            self._div_depth = max(0, self._div_depth - 1)
        elif tag == "a":
            self._in_name = False

    def handle_data(self, data: str) -> None:
        if self._post is None:
            return
        if self._text_depth is not None:
            self._post["text"].append(data)
        elif self._in_name and not self._post["author"]:
            author = data.strip()
            if author:
                self._post["author"] = author


def _post_id(data_post: str) -> int:
    tail = data_post.rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _post_ts(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def parse_preview(html: str, channel: str, limit: int = 20) -> list[IncomingMessage]:
    """Parse a t.me/s page into messages, oldest first."""
    parser = _PreviewParser()
    parser.feed(html)
    out: list[IncomingMessage] = []
    for p in parser.posts[-limit:]:
        text = "".join(p["text"]).strip()
        if not text:
            continue
        out.append(
            IncomingMessage(
                chat_id=0,  # the web preview never reveals the numeric id
                message_id=_post_id(p["post"]),
                text=text,
                user_name=p["author"],
                is_group=True,
                chat_title=p["author"],
                chat_username=f"@{channel}",
                ts=_post_ts(p["ts"]),
            )
        )
    return out


class WebPreviewReader:
    """Channel reader used by the read_telegram tool and the browse job."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    async def read_source(self, source: str, limit: int = 20) -> list[IncomingMessage]:
        username = str(source).strip().lstrip("@")
        if not username or not username.replace("_", "").isalnum():
            log.info("webpreview: %r is not a channel @username", source)
            return []
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": "libertati/0.2"},
            follow_redirects=True,
        ) as client:
            try:
                resp = await client.get(f"https://t.me/s/{username}")
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("webpreview: fetch of @%s failed: %s", username, exc)
                return []
        # Channels without a preview redirect from /s/<name> to the bare profile page.
        if "/s/" not in str(resp.url):
            log.info("webpreview: @%s has no public preview", username)
            return []
        return parse_preview(resp.text, username, limit=limit)
