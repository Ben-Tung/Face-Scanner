from app import confirmation_email
from app.config import Settings

_TO_EMAIL = "customer@example.com"
_RESULT_URL = "http://localhost:3000/result/11111111-1111-1111-1111-111111111111"


def test_send_confirmation_email_noops_without_api_key(monkeypatch):
    monkeypatch.setattr(confirmation_email, "get_settings", lambda: Settings(resend_api_key=None))

    def _fail_if_called(params):
        raise AssertionError("must not call Resend without an API key")

    monkeypatch.setattr(confirmation_email.resend.Emails, "send", _fail_if_called)

    assert confirmation_email.send_confirmation_email(_TO_EMAIL, _RESULT_URL) is False


def test_send_confirmation_email_swallows_send_errors(monkeypatch):
    monkeypatch.setattr(
        confirmation_email, "get_settings", lambda: Settings(resend_api_key="re_test_fake")
    )

    def _raise(params):
        raise RuntimeError("network boom")

    monkeypatch.setattr(confirmation_email.resend.Emails, "send", _raise)

    assert confirmation_email.send_confirmation_email(_TO_EMAIL, _RESULT_URL) is False


def test_send_confirmation_email_sends_with_expected_params(monkeypatch):
    monkeypatch.setattr(
        confirmation_email, "get_settings", lambda: Settings(resend_api_key="re_test_fake")
    )

    captured = {}

    def _fake_send(params):
        captured.update(params)
        return {"id": "email_123"}

    monkeypatch.setattr(confirmation_email.resend.Emails, "send", _fake_send)

    result = confirmation_email.send_confirmation_email(_TO_EMAIL, _RESULT_URL)

    assert result is True
    assert captured["to"] == _TO_EMAIL
    assert captured["from"] == confirmation_email._FROM_ADDRESS
    assert _RESULT_URL in captured["text"]
    assert _RESULT_URL in captured["html"]
