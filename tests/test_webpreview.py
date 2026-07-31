from __future__ import annotations

from libertati.telegram.webpreview import parse_preview

SAMPLE = """
<html><body>
<section class="tgme_channel_history js-message_history">
  <div class="tgme_widget_message_wrap js-widget_message_wrap">
    <div class="tgme_widget_message text_not_supported_wrap js-widget_message"
         data-post="somechan/101">
      <div class="tgme_widget_message_bubble">
        <div class="tgme_widget_message_author accent_color">
          <a class="tgme_widget_message_owner_name" href="https://t.me/somechan">
            <span dir="auto">Some Channel</span></a>
        </div>
        <div class="tgme_widget_message_text js-message_text" dir="auto">
          First post with a <a href="https://x.test">link</a> and
          <i class="emoji"><b>👍</b></i>
        </div>
        <div class="tgme_widget_message_footer compact js-message_footer">
          <a class="tgme_widget_message_date" href="https://t.me/somechan/101">
            <time datetime="2026-07-30T10:00:00+00:00" class="time">10:00</time></a>
        </div>
      </div>
    </div>
  </div>
  <div class="tgme_widget_message_wrap js-widget_message_wrap">
    <div class="tgme_widget_message text_not_supported_wrap js-widget_message"
         data-post="somechan/102">
      <div class="tgme_widget_message_bubble">
        <div class="tgme_widget_message_text js-message_text" dir="auto">
          Second post line one<br/>line two
        </div>
      </div>
    </div>
  </div>
  <div class="tgme_widget_message_wrap js-widget_message_wrap">
    <div class="tgme_widget_message js-widget_message" data-post="somechan/103">
      <div class="tgme_widget_message_bubble">
        <!-- photo-only post: no message_text div -->
      </div>
    </div>
  </div>
</section>
</body></html>
"""


def test_parse_preview_extracts_posts():
    msgs = parse_preview(SAMPLE, "somechan")
    assert len(msgs) == 2  # photo-only post has no text and is dropped
    first, second = msgs
    assert first.message_id == 101
    assert "First post with a link" in " ".join(first.text.split())
    assert "👍" in first.text
    assert first.user_name == "Some Channel"
    assert first.chat_username == "@somechan"
    assert first.ts is not None
    assert second.message_id == 102
    assert second.text == "Second post line one\nline two"


def test_parse_preview_respects_limit():
    # limit keeps the newest posts; the photo-only 103 has no text and is dropped
    msgs = parse_preview(SAMPLE, "somechan", limit=2)
    assert [m.message_id for m in msgs] == [102]


def test_parse_preview_empty_html():
    assert parse_preview("<html></html>", "x") == []
