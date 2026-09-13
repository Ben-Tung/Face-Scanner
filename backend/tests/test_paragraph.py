from app import paragraph
from app.config import Settings
from app.palettes import Swatch
from app.vision.season_classifier import SeasonClassification


def _classification() -> SeasonClassification:
    return SeasonClassification(
        season="Winter",
        undertone="cool",
        depth="deep",
        clarity="clear",
        avg_lab=(40.0, 10.0, -5.0),
        hue_deg=300.0,
        chroma=11.0,
    )


def _swatches() -> list[Swatch]:
    return [Swatch("True Red", "#D0103A"), Swatch("Emerald Green", "#00693E")]


def test_generate_paragraph_noops_without_api_key(monkeypatch):
    monkeypatch.setattr(paragraph, "get_settings", lambda: Settings(anthropic_api_key=None))

    def _fail_if_called():
        raise AssertionError("must not construct a client without an API key")

    monkeypatch.setattr(paragraph, "_client", _fail_if_called)

    assert paragraph.generate_paragraph(_classification(), _swatches()) is None


def test_generate_paragraph_swallows_client_errors(monkeypatch):
    monkeypatch.setattr(paragraph, "get_settings", lambda: Settings(anthropic_api_key="test-key"))

    class _RaisingMessages:
        def create(self, **kwargs):
            raise RuntimeError("network boom")

    class _FakeClient:
        messages = _RaisingMessages()

    monkeypatch.setattr(paragraph, "_client", lambda: _FakeClient())

    assert paragraph.generate_paragraph(_classification(), _swatches()) is None


def test_generate_paragraph_returns_extracted_text(monkeypatch):
    monkeypatch.setattr(paragraph, "get_settings", lambda: Settings(anthropic_api_key="test-key"))

    class _TextBlock:
        type = "text"
        text = "You're a Winter — cool, deep, and clear."

    class _FakeResponse:
        content = [_TextBlock()]

    class _FakeMessages:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(paragraph, "_client", lambda: _FakeClient())

    result = paragraph.generate_paragraph(_classification(), _swatches())
    assert result == "You're a Winter — cool, deep, and clear."
