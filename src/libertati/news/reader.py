"""Read the news from configurable RSS/Atom feeds."""

from __future__ import annotations

from dataclasses import dataclass

import feedparser
import httpx

from ..logging import get_logger

log = get_logger("news")


@dataclass(slots=True)
class NewsItem:
    title: str
    summary: str
    link: str
    source: str


class NewsReader:
    def __init__(self, feeds: list[str], timeout: float = 10.0) -> None:
        self.feeds = feeds
        self.timeout = timeout

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> list[NewsItem]:
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("failed to fetch feed %s: %s", url, exc)
            return []
        parsed = feedparser.parse(resp.content)
        source = parsed.feed.get("title", url) if parsed.feed else url
        items: list[NewsItem] = []
        for entry in parsed.entries:
            items.append(
                NewsItem(
                    title=entry.get("title", "").strip(),
                    summary=_clean(entry.get("summary", "")),
                    link=entry.get("link", ""),
                    source=source,
                )
            )
        return items

    async def fetch(self, limit_per_feed: int = 5, topic: str | None = None) -> list[NewsItem]:
        items: list[NewsItem] = []
        async with httpx.AsyncClient(
            timeout=self.timeout, headers={"User-Agent": "libertati/0.2"}
        ) as client:
            for url in self.feeds:
                feed_items = await self._fetch(client, url)
                items.extend(feed_items[:limit_per_feed])
        if topic:
            needle = topic.lower()
            items = [i for i in items if needle in i.title.lower() or needle in i.summary.lower()]
        # de-dupe by title
        seen: set[str] = set()
        deduped: list[NewsItem] = []
        for item in items:
            key = item.title.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped


def _clean(text: str) -> str:
    # strip crude HTML tags feedparser sometimes leaves in summaries
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())[:400]


def format_digest(items: list[NewsItem], limit: int = 10) -> str:
    if not items:
        return "No news items fetched."
    lines = []
    for item in items[:limit]:
        lines.append(f"- {item.title} ({item.source})")
        if item.summary:
            lines.append(f"  {item.summary}")
    return "\n".join(lines)
