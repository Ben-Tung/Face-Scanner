"""Stripe Checkout session creation, payment status lookup, and webhook
handling for unlocking a scan's full report.

Payment confirmation is never trusted from the client. The `paid` state
returned by GET /api/scans/{scan_id} only ever flips because either (a) the
Stripe webhook below verified the event's signature and updated the scans
row itself, or (b) get_scan_state retrieved the Checkout Session directly
from Stripe's API and confirmed both its payment_status and that its
metadata actually names this scan. The client supplies a scan_id and,
after redirect, a session_id — never a "paid" boolean, never an amount.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import stripe
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from app.analytics import log_event
from app.config import get_settings
from app.scans_repo import ScanRow, get_scan, mark_scan_paid, set_checkout_session
from app.schemas import SwatchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["payments"])

# $2.99 — within CLAUDE.md's v1 pricing range, fixed entirely server-side.
# The client never supplies or influences this value.
_FULL_REPORT_PRICE_CENTS = 299

_FULL_REPORT_NOTE = "Full report content coming soon — you've unlocked payment successfully."


class FullReportResponse(BaseModel):
    swatches: list[SwatchResponse]
    paragraph: str | None
    note: str


class ScanStateResponse(BaseModel):
    scan_id: str
    season: str
    swatches: list[SwatchResponse]
    paragraph: str | None
    paid: bool
    full_report: FullReportResponse | None = None


class CheckoutResponse(BaseModel):
    checkout_url: str


def _parse_scan_id(scan_id: str) -> str:
    try:
        return str(uuid.UUID(scan_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Scan not found.")


def _swatches_response(row: ScanRow) -> list[SwatchResponse]:
    return [SwatchResponse(**swatch) for swatch in row.swatches]


def _scan_state_response(row: ScanRow, paid: bool) -> ScanStateResponse:
    swatches = _swatches_response(row)
    full_report = (
        FullReportResponse(swatches=swatches, paragraph=row.paragraph, note=_FULL_REPORT_NOTE)
        if paid
        else None
    )
    return ScanStateResponse(
        scan_id=row.id,
        season=row.season,
        swatches=swatches,
        paragraph=row.paragraph,
        paid=paid,
        full_report=full_report,
    )


def _verify_and_mark_paid(row: ScanRow, session_id: str) -> bool:
    """Fallback confirmation for the moment right after a Stripe redirect,
    in case the webhook hasn't landed yet. Retrieves the session from
    Stripe itself rather than trusting the client's own claim, and checks
    that it both paid *and* belongs to this exact scan — otherwise a
    session_id for someone else's paid checkout (a different scan) could
    unlock this one.
    """
    settings = get_settings()
    if not settings.stripe_secret_key:
        return False

    try:
        session = stripe.checkout.Session.retrieve(session_id, api_key=settings.stripe_secret_key)
    except stripe.error.StripeError:
        logger.warning(
            "Failed to verify checkout session %r for scan %r", session_id, row.id, exc_info=True
        )
        return False

    # Stripe's SDK objects deliberately block dict-style .get() (raises
    # "... is not a dict, use .to_dict()") — declared API fields are plain
    # attributes (session.payment_status, session.payment_intent), while
    # metadata's keys are user-defined, so those need getattr's safe
    # default rather than attribute access, which would raise on a key
    # that was never set.
    scan_id = getattr(session.metadata, "scan_id", None)
    if session.payment_status != "paid" or scan_id != row.id:
        return False

    mark_scan_paid(row.id, session.payment_intent)
    return True


@router.get("/scans/{scan_id}", response_model=ScanStateResponse)
def get_scan_state(scan_id: str, session_id: str | None = None) -> ScanStateResponse:
    scan_id = _parse_scan_id(scan_id)
    row = get_scan(scan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Scan not found.")

    paid = row.paid
    if not paid:
        log_event("paywall_viewed", {"scan_id": scan_id})
        if session_id:
            paid = _verify_and_mark_paid(row, session_id)

    return _scan_state_response(row, paid)


@router.post("/scans/{scan_id}/checkout", response_model=CheckoutResponse)
def create_checkout(scan_id: str) -> CheckoutResponse:
    scan_id = _parse_scan_id(scan_id)
    row = get_scan(scan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Scan not found.")
    if row.paid:
        raise HTTPException(status_code=409, detail="This scan has already been unlocked.")

    settings = get_settings()
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=500, detail="Stripe is not configured on this server.")

    # {CHECKOUT_SESSION_ID} is Stripe's own template literal, substituted by
    # Stripe when it builds the redirect — built via concatenation (not an
    # f-string) so Python's brace-interpolation can't mangle it.
    success_url = f"{settings.frontend_base_url}/result/{scan_id}?session_id=" + "{CHECKOUT_SESSION_ID}"
    cancel_url = f"{settings.frontend_base_url}/result/{scan_id}"

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": _FULL_REPORT_PRICE_CENTS,
                        "product_data": {"name": "Palette Full Report"},
                    },
                    "quantity": 1,
                }
            ],
            metadata={"scan_id": scan_id},
            success_url=success_url,
            cancel_url=cancel_url,
            api_key=settings.stripe_secret_key,
        )
    except stripe.error.StripeError:
        logger.exception("Failed to create checkout session for scan %r", scan_id)
        raise HTTPException(status_code=502, detail="We couldn't start checkout right now. Please try again.")

    set_checkout_session(scan_id, session.id)
    return CheckoutResponse(checkout_url=session.url)


@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request, stripe_signature: str = Header(None, alias="Stripe-Signature")
) -> dict[str, Any]:
    settings = get_settings()
    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, settings.stripe_webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid payload or signature.")

    # Card-only Checkout (see create_checkout) settles synchronously, so
    # checkout.session.completed means the charge is already captured —
    # there's no pending state and no need to also handle
    # checkout.session.async_payment_succeeded/_failed for v1. Any other
    # event type is acknowledged and ignored.
    if event.type == "checkout.session.completed":
        session = event.data.object
        # metadata's keys are user-defined (unlike declared API fields such
        # as .id/.payment_intent), so a missing scan_id needs getattr's
        # safe default rather than attribute access.
        scan_id = getattr(session.metadata, "scan_id", None)
        if not scan_id:
            logger.warning(
                "checkout.session.completed with no scan_id metadata (session %r)", session.id
            )
            return {"status": "ignored"}

        # Deliberately not wrapped in try/except: if this raises (e.g. the
        # database is unreachable), Stripe retries the webhook delivery on
        # its own backoff schedule, giving payment confirmation another
        # chance to land on top of (not instead of) get_scan_state's
        # fallback verification above.
        mark_scan_paid(scan_id, session.payment_intent)
        log_event(
            "purchase_completed",
            {"scan_id": scan_id, "payment_intent_id": session.payment_intent},
        )

    return {"status": "ok"}
