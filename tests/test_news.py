from __future__ import annotations

from libertati.news.reader import NewsReader, format_digest

SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Example News</title>
  <item><title>First headline</title><description>Some &lt;b&gt;bold&lt;/b&gt; summary</description><link>http://a</link></item>
  <item><title>Second headline</title><description>About libertarianism</description><link>http://b</link></item>
  <item><title>First headline</title><description>dupe</description><link>http://c</link></item>
</channel></rss>"""


class FakeReader(NewsReader):
    async def _fetch(self, client, url):  # override network
        import feedparser

        parsed = feedparser.parse(SAMPLE_RSS)
        from libertati.news.reader import NewsItem, _clean

        return [
            NewsItem(
                title=e.get("title", ""),
                summary=_clean(e.get("summary", "")),
                link=e.get("link", ""),
                source="Example News",
            )
            for e in parsed.entries
        ]


async def test_fetch_parses_and_dedupes():
    reader = FakeReader(feeds=["http://x"])
    items = await reader.fetch()
    titles = [i.title for i in items]
    assert titles == ["First headline", "Second headline"]  # dupe removed
    # HTML stripped from summary
    assert "<b>" not in items[0].summary
    assert "bold" in items[0].summary


async def test_fetch_topic_filter():
    reader = FakeReader(feeds=["http://x"])
    items = await reader.fetch(topic="libertarianism")
    assert len(items) == 1
    assert items[0].title == "Second headline"


def test_format_digest_empty():
    assert "No news" in format_digest([])


async def test_real_fetch_handles_http_error():
    # a feed URL that errors should be skipped, not raise
    reader = NewsReader(feeds=["http://127.0.0.1:1/nope"], timeout=0.5)
    items = await reader.fetch()
    assert items == []
