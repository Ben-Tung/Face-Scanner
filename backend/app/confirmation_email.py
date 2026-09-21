"""Best-effort purchase-confirmation email via Resend.

Sent once, right after a Stripe payment is confirmed (see the webhook
handler in app/routers/payments.py), to hand the customer their permanent
result link. Mirrors app/paragraph.py's best-effort contract: a confirmed
payment must never be treated as failed just because this can't reach
Resend or isn't configured. Every failure here is caught and logged;
callers get False and must not retry or fail the webhook because of it.
"""

from __future__ import annotations

import logging

import resend
from resend.exceptions import ResendError

from app.config import get_settings

logger = logging.getLogger(__name__)

# Resend's own sandbox sender — works without a verified domain, but Resend
# restricts delivery to the account's own signup email until a domain is
# verified. Swap for a verified no-reply@<domain> sender once that's set up
# (out of scope for this change).
_FROM_ADDRESS = "Palette <onboarding@resend.dev>"
_SUBJECT = "Your color season results are ready"

_TEXT_TEMPLATE = (
    "Thanks for your purchase! Your full color season report is ready.\n\n"
    "View it here: {result_url}"
)
_HTML_TEMPLATE = (
    "<p>Thanks for your purchase! Your full color season report is ready.</p>"
    '<p><a href="{result_url}">View your results</a></p>'
)


def send_confirmation_email(to_email: str, result_url: str) -> bool:
    """Send a short thank-you email with the permanent result link.

    Best-effort: returns False (never raises) if RESEND_API_KEY isn't
    configured or the send fails for any reason. Callers must not treat a
    confirmed payment as failed just because this returns False.
    """
    settings = get_settings()
    if not settings.resend_api_key:
        return False

    resend.api_key = settings.resend_api_key
    params = {
        "from": _FROM_ADDRESS,
        "to": to_email,
        "subject": _SUBJECT,
        "text": _TEXT_TEMPLATE.format(result_url=result_url),
        "html": _HTML_TEMPLATE.format(result_url=result_url),
    }

    try:
        resend.Emails.send(params)
    except ResendError:
        logger.warning("Resend rejected the confirmation email", exc_info=True)
        return False
    except Exception:
        logger.warning("Unexpected failure sending the confirmation email", exc_info=True)
        return False

    return True
