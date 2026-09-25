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
from collections.abc import Sequence
from typing import Any, Literal

import stripe
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from app.analytics import log_event
from app.beauty_guidance import GOLD_SWATCH, GUIDANCE_BY_SEASON, SILVER_SWATCH, BeautyGuidance
from app.config import Settings, get_settings
from app.confirmation_email import send_confirmation_email
from app.palettes import FULL_PALETTE_BY_SEASON, Swatch
from app.paragraph import generate_full_report_paragraph
from app.scans_repo import (
    ScanRow,
    get_scan,
    mark_scan_paid,
    set_checkout_session,
    set_full_report_paragraph,
)
from app.schemas import SwatchResponse, parse_scan_id_or_404, to_swatch_responses
from app.vision.season_classifier import Season

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["payments"])

# $2.99 — within CLAUDE.md's v1 pricing range, fixed entirely server-side.
# The client never supplies or influences this value.
_FULL_REPORT_PRICE_CENTS = 299


class PaletteSectionResponse(BaseModel):
    best: list[SwatchResponse]
    good: list[SwatchResponse]
    avoid: list[SwatchResponse]


class MakeupGuidanceResponse(BaseModel):
    foundation_undertone: str
    foundation_tip: str
    lip_shades: list[SwatchResponse]
    blush_shades: list[SwatchResponse]


class JewelryGuidanceResponse(BaseModel):
    metal: Literal["Gold", "Silver", "Both"]
    tip: str
    gold_swatch: SwatchResponse
    silver_swatch: SwatchResponse


class FullReportResponse(BaseModel):
    palette: PaletteSectionResponse
    makeup: MakeupGuidanceResponse
    jewelry: JewelryGuidanceResponse
    shopping_guidance: str
    paragraph: str | None


class ScanStateResponse(BaseModel):
    scan_id: str
    season: Season
    swatches: list[SwatchResponse]
    paragraph: str | None
    paid: bool
    retake_used: bool
    price_cents: int
    full_report: FullReportResponse | None = None


class CheckoutResponse(BaseModel):
    checkout_url: str


def _swatches_response(row: ScanRow) -> list[SwatchResponse]:
    return [SwatchResponse(**swatch) for swatch in row.swatches]


def _resolve_full_report_paragraph(
    row: ScanRow, season: Season, best_swatches: Sequence[Swatch], guidance: BeautyGuidance
) -> str | None:
    """Return the cached paid paragraph, or generate and cache one.

    The palette/guidance are cheap and deterministic, recomputed on every
    call from `row.season` alone. The AI paragraph is the one part that
    costs money and has latency, so it's generated once (lazily, on first
    paid view) and cached on the row for every read after that.
    """
    if row.full_report_paragraph:
        return row.full_report_paragraph

    paragraph = generate_full_report_paragraph(season, best_swatches, guidance)
    if paragraph:
        set_full_report_paragraph(row.id, paragraph)
    return paragraph


def _full_report_response(row: ScanRow) -> FullReportResponse:
    palette = FULL_PALETTE_BY_SEASON[row.season]
    guidance = GUIDANCE_BY_SEASON[row.season]

    paragraph = _resolve_full_report_paragraph(row, row.season, palette.best, guidance)

    return FullReportResponse(
        palette=PaletteSectionResponse(
            best=to_swatch_responses(palette.best),
            good=to_swatch_responses(palette.good),
            avoid=to_swatch_responses(palette.avoid),
        ),
        makeup=MakeupGuidanceResponse(
            foundation_undertone=guidance.makeup.foundation_undertone,
            foundation_tip=guidance.makeup.foundation_tip,
            lip_shades=to_swatch_responses(guidance.makeup.lip_shades),
            blush_shades=to_swatch_responses(guidance.makeup.blush_shades),
        ),
        jewelry=JewelryGuidanceResponse(
            metal=guidance.jewelry.metal,
            tip=guidance.jewelry.tip,
            gold_swatch=SwatchResponse(name=GOLD_SWATCH.name, hex=GOLD_SWATCH.hex),
            silver_swatch=SwatchResponse(name=SILVER_SWATCH.name, hex=SILVER_SWATCH.hex),
        ),
        shopping_guidance=guidance.shopping_guidance,
        paragraph=paragraph,
    )


def _scan_state_response(row: ScanRow, paid: bool) -> ScanStateResponse:
    swatches = _swatches_response(row)
    full_report = _full_report_response(row) if paid else None
    return ScanStateResponse(
        scan_id=row.id,
        season=row.season,
        swatches=swatches,
        paragraph=row.paragraph,
        paid=paid,
        retake_used=row.retake_used,
        price_cents=_FULL_REPORT_PRICE_CENTS,
        full_report=full_report,
    )


def _finalize_paid_scan(scan_id: str, session: Any, settings: Settings) -> None:
    """Marks a scan paid, logs purchase_completed, and sends the
    confirmation email -- shared by the webhook and the post-redirect
    fallback verify below, since either one might be the first to observe
    a given payment, and a customer should get their confirmation email
    regardless of which path gets there first.
    """
    if not mark_scan_paid(scan_id, session.payment_intent):
        return

    log_event(
        "purchase_completed",
        {"scan_id": scan_id, "payment_intent_id": session.payment_intent},
    )

    # Stripe Checkout collects this by default; not persisted anywhere, so
    # this is the only place it's ever available. customer_details is an
    # optional nested object on the session -- same defensive getattr style
    # as metadata.scan_id above. Falls back to the session's top-level
    # customer_email (populated when Stripe already knew the customer's
    # email before checkout) so a null/absent customer_details.email
    # doesn't silently drop the email.
    customer_email = getattr(
        getattr(session, "customer_details", None), "email", None
    ) or getattr(session, "customer_email", None)
    if customer_email:
        result_url = f"{settings.frontend_base_url}/result/{scan_id}"
        send_confirmation_email(customer_email, result_url)
    else:
        logger.warning(
            "checkout.session.completed with no customer email (session %r, scan %r)",
            session.id,
            scan_id,
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

    _finalize_paid_scan(row.id, session, settings)
    return True


@router.get("/scans/{scan_id}", response_model=ScanStateResponse)
def get_scan_state(scan_id: str, session_id: str | None = None) -> ScanStateResponse:
    scan_id = parse_scan_id_or_404(scan_id)
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
    scan_id = parse_scan_id_or_404(scan_id)
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
        _finalize_paid_scan(scan_id, session, settings)

    return {"status": "ok"}
