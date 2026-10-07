from __future__ import annotations

import logging
import math
import threading
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from app.analytics import log_event
from app.config import get_settings
from app.palettes import SWATCHES_BY_SEASON
from app.paragraph import generate_paragraph
from app.rate_limit import limiter, scan_limit_value
from app.scans_repo import consume_retake, create_scan, get_scan
from app.schemas import SwatchResponse, parse_scan_id_or_404, to_swatch_responses
from app.vision.image_decode import decode_image
from app.vision.season_classifier import Season, SeasonClassificationResult, classify_season, rgb_to_lab
from app.vision.skin_sampling import RGB, AnchorPoints, SkinSampleResult, sample_skin_regions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["scan"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — generous for a phone camera selfie
# Upper bound on a manually submitted patch_half_size, as a fraction of the
# image's shorter side. The frontend only ever echoes back the automatic
# value (9% of interocular distance - a few percent of the short side), so
# this leaves wide headroom while stopping one patch from swallowing the
# whole photo, background included.
_MAX_PATCH_HALF_SIZE_RATIO = 0.25

# The scan handlers are plain `def`, so FastAPI runs them on its threadpool
# and the event loop stays free while a scan waits on Anthropic or Postgres
# (as `async def` handlers calling those blocking clients, one scan's
# multi-second paragraph call used to stall every other request on the
# server). That threadpool would also let many scans decode full-resolution
# photos at once, so this lock keeps it to one photo's working set at a
# time - the same peak memory as before. Decode + sampling takes ~0.1-0.3s,
# so serializing just that part costs little; the slow I/O runs outside it.
# Could become a BoundedSemaphore(n) once the host's memory headroom is known.
_IMAGE_PIPELINE_LOCK = threading.Lock()

_SAMPLE_ERROR_MESSAGES: dict[str, str] = {
    "no_face_detected": "We couldn't find a face in that photo. Try again with your face centered and well-lit.",
    "face_out_of_frame": (
        "Your face is too close to the edge of the frame — drag the boxes below to "
        "reposition them, or retake with your face more centered."
    ),
    "patch_clipped": (
        "That photo is too bright in places — drag the boxes below to fine-tune "
        "where we sample, or retake it in softer light."
    ),
    # Sent as a plain-string detail, never the structured adjust-the-boxes
    # body: the face is blown out across most of the boxes, so there's no
    # usable skin to drag them onto.
    "overexposed": (
        "That photo is too bright to read your skin tone. Try again in softer, indirect "
        "light, facing a window rather than direct sun."
    ),
}
_CLASSIFY_ERROR_MESSAGES: dict[str, str] = {
    "inconsistent_patches": (
        "We got mixed color readings from your forehead and cheeks — usually hair, "
        "a shadow, or a bright reflection over one of the boxes. Drag the boxes onto "
        "clear skin, or retake with your hair pulled back."
    ),
    # Sent as a plain-string detail, never the structured adjust-the-boxes
    # body: no placement of the boxes can fix the photo's lighting.
    "color_cast": (
        "The lighting in that photo has a strong color tint (like a warm indoor bulb), so we "
        "can't read your skin tone accurately. Try again facing a window in daylight."
    ),
}
# The retake screen has no drag-the-boxes adjuster (see retake-capture.tsx),
# so its low-confidence messages only ever point at trying another photo.
_RETAKE_ERROR_MESSAGES: dict[str, str] = {
    "face_out_of_frame": (
        "Your face is too close to the edge of that photo. Try another with your face centered in the frame."
    ),
    "patch_clipped": "That photo is too bright in places. Try another in softer, indirect light.",
    "overexposed": _SAMPLE_ERROR_MESSAGES["overexposed"],
    "inconsistent_patches": (
        "We got mixed color readings from your forehead and cheeks — usually hair, a shadow, "
        "or a bright reflection. Try another photo with your hair pulled back, in even light."
    ),
    "color_cast": _CLASSIFY_ERROR_MESSAGES["color_cast"],
}


class ScanResponse(BaseModel):
    scan_id: str
    season: Season
    swatches: list[SwatchResponse]
    paragraph: str | None = None


class PatchPoint(BaseModel):
    x: float
    y: float


class PatchAnchorsPayload(BaseModel):
    forehead: PatchPoint
    left_cheek: PatchPoint
    right_cheek: PatchPoint
    patch_half_size: float


class ScanImage(BaseModel):
    width: int
    height: int


class LowConfidenceDetail(BaseModel):
    reason: Literal["patch_clipped", "inconsistent_patches", "face_out_of_frame"]
    message: str
    patches: PatchAnchorsPayload
    image: ScanImage


def _read_upload(photo: UploadFile) -> bytes:
    """Validate and read an upload's raw bytes.

    Never retains the uploaded photo beyond the caller's request — nothing
    here writes it to disk or a database (see CLAUDE.md's retention rule).
    """
    if photo.content_type and not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Please upload an image file.")

    contents = photo.file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Please upload an image file.")
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="That photo is too large (max 10MB). Try a smaller photo.")
    return contents


def _decode_or_422(contents: bytes) -> np.ndarray:
    image_bgr = decode_image(contents)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="We couldn't read that image. Try a different photo.")
    return image_bgr


def _persist_scan(season: Season, swatches: list[SwatchResponse], paragraph: str | None) -> str:
    """Save the free-result fields and hand back a scan id.

    Unlike log_event's best-effort contract, a scan that can't be persisted
    can never legitimately be sold, so a failure here must surface to the
    caller rather than being swallowed.
    """
    scan_id = str(uuid4())
    try:
        create_scan(scan_id, season, [s.model_dump() for s in swatches], paragraph)
    except Exception:
        logger.exception("Failed to persist scan %r", scan_id)
        raise HTTPException(
            status_code=503, detail="We couldn't save your scan right now. Please try again."
        )
    return scan_id


def _sclera_rgb(sample: SkinSampleResult) -> RGB | None:
    """This photo's sclera lighting reference for classify_season, or None
    (classify on uncorrected color) when it couldn't be reliably read."""
    return sample.sclera.sclera_rgb if sample.sclera is not None and sample.sclera.success else None


def _classify(sample: SkinSampleResult) -> SeasonClassificationResult:
    """Classify a successful sample with the same lighting corrections on
    every endpoint, so the same boxes on the same photo land on the same
    season whichever path produced them."""
    return classify_season(
        sample.forehead_rgb,
        sample.left_cheek_rgb,
        sample.right_cheek_rgb,
        sclera_rgb=_sclera_rgb(sample),
        correct_depth=get_settings().sclera_depth_correction,
    )


def _scan_measurements(sample: SkinSampleResult, result: SeasonClassificationResult) -> dict:
    """Event metadata carrying the numbers this scan was judged on, so
    real-world values can be compared against the classifier's thresholds
    over time: cross-patch ΔH*, the lighting's color cast, whether a shaded
    forehead was dropped, the sclera reading itself, and - on success - the
    skin color and the hue/chroma/depth the season was decided from.

    Numbers, flags and short reason strings only - never pixels, and
    nothing that identifies the person."""

    def _rounded(value: float | None) -> float | None:
        return round(value, 2) if value is not None else None

    sclera = sample.sclera
    sclera_rgb = _sclera_rgb(sample)
    sclera_lab = rgb_to_lab(sclera_rgb) if sclera_rgb is not None else (None, None, None)
    measurements = {
        "max_hue_difference": _rounded(result.max_hue_difference),
        "color_cast": _rounded(result.color_cast),
        "forehead_dropped": result.forehead_dropped,
        "sclera_L": _rounded(sclera_lab[0]),
        "sclera_a": _rounded(sclera_lab[1]),
        "sclera_b": _rounded(sclera_lab[2]),
        "sclera_pixel_count": sclera.pixel_count if sclera is not None else None,
        "sclera_error": sclera.error if sclera is not None else None,
    }

    classification = result.classification
    if classification is not None:
        measurements |= {
            "skin_L": _rounded(classification.avg_lab[0]),
            "skin_a": _rounded(classification.avg_lab[1]),
            "skin_b": _rounded(classification.avg_lab[2]),
            "hue_deg": _rounded(classification.hue_deg),
            "chroma": _rounded(classification.chroma),
            "depth_lightness": _rounded(classification.depth_lightness),
            # Replaces depth_normalized, which meant "a sclera reading was
            # used" back when it gated both corrections; older rows'
            # depth_normalized equals color_corrected.
            "color_corrected": sclera_rgb is not None,
            "depth_corrected": classification.depth_corrected,
        }
    return measurements


def _patch_anchors_payload(anchors: AnchorPoints) -> PatchAnchorsPayload:
    return PatchAnchorsPayload(
        forehead=PatchPoint(x=float(anchors.forehead[0]), y=float(anchors.forehead[1])),
        left_cheek=PatchPoint(x=float(anchors.left_cheek[0]), y=float(anchors.left_cheek[1])),
        right_cheek=PatchPoint(x=float(anchors.right_cheek[0]), y=float(anchors.right_cheek[1])),
        patch_half_size=anchors.patch_half_size,
    )


def _low_confidence_detail(
    reason: Literal["patch_clipped", "inconsistent_patches", "face_out_of_frame"],
    message: str,
    anchors: AnchorPoints,
    width: int,
    height: int,
) -> dict:
    return LowConfidenceDetail(
        reason=reason,
        message=message,
        patches=_patch_anchors_payload(anchors),
        image=ScanImage(width=width, height=height),
    ).model_dump()


@router.post("/scan", response_model=ScanResponse)
@limiter.limit(scan_limit_value)
def scan(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    photo: UploadFile = File(...),
) -> ScanResponse:
    """Detect a face in the uploaded photo and classify its color season.

    Never retains the uploaded photo beyond this request — nothing here
    writes it to disk or a database (see CLAUDE.md's retention rule).
    """
    contents = _read_upload(photo)
    with _IMAGE_PIPELINE_LOCK:
        image_bgr = _decode_or_422(contents)
        height, width = image_bgr.shape[:2]
        sample = sample_skin_regions(image_bgr)
        del image_bgr  # free the full-resolution array before releasing the lock

    # Called directly, not via background_tasks: FastAPI only attaches queued
    # background tasks to a successful Response, never to the response an
    # HTTPException raised further down would build — so a backgrounded call
    # here would silently never fire on a rejected/failed scan. log_event()
    # is itself try/except-wrapped and timeout-bounded, so this stays cheap.
    log_event("scan_started")

    if not sample.success:
        if sample.error in ("patch_clipped", "face_out_of_frame"):
            assert sample.anchors is not None  # both reasons always carry anchors
            log_event("scan_low_confidence", {"reason": sample.error})
            raise HTTPException(
                status_code=422,
                detail=_low_confidence_detail(
                    sample.error, _SAMPLE_ERROR_MESSAGES[sample.error], sample.anchors, width, height
                ),
            )
        if sample.error == "overexposed":
            log_event("scan_low_confidence", {"reason": "overexposed"})
        raise HTTPException(status_code=422, detail=_SAMPLE_ERROR_MESSAGES[sample.error])

    result = _classify(sample)
    if not result.success:
        log_event("scan_low_confidence", {"reason": result.error, **_scan_measurements(sample, result)})
        if result.error == "color_cast":
            raise HTTPException(status_code=422, detail=_CLASSIFY_ERROR_MESSAGES["color_cast"])
        assert sample.anchors is not None  # sampling succeeded, so anchors are always set
        raise HTTPException(
            status_code=422,
            detail=_low_confidence_detail(
                "inconsistent_patches",
                _CLASSIFY_ERROR_MESSAGES["inconsistent_patches"],
                sample.anchors,
                width,
                height,
            ),
        )

    season = result.classification.season
    swatches = to_swatch_responses(SWATCHES_BY_SEASON[season])
    paragraph = generate_paragraph(result.classification, SWATCHES_BY_SEASON[season])
    scan_id = _persist_scan(season, swatches, paragraph)
    background_tasks.add_task(
        log_event,
        "scan_completed",
        {"season": season, "scan_id": scan_id, **_scan_measurements(sample, result)},
    )
    return ScanResponse(scan_id=scan_id, season=season, swatches=swatches, paragraph=paragraph)


@router.post("/scan/manual", response_model=ScanResponse)
@limiter.limit(scan_limit_value)
def scan_manual(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    photo: UploadFile = File(...),
    forehead_x: float = Form(...),
    forehead_y: float = Form(...),
    left_cheek_x: float = Form(...),
    left_cheek_y: float = Form(...),
    right_cheek_x: float = Form(...),
    right_cheek_y: float = Form(...),
    patch_half_size: float = Form(...),
) -> ScanResponse:
    """Re-classify against manually adjusted patch coordinates.

    Recovery path for a low-confidence /scan result: the caller resends the
    same photo bytes (nothing here persists the image either, same as
    /scan) alongside forehead/cheek centers the user dragged into place.
    The face is re-detected on those bytes only to read the sclera, so the
    result gets the same lighting correction /scan applies - the same boxes
    on the same photo must classify the same way through either endpoint.
    """
    contents = _read_upload(photo)
    with _IMAGE_PIPELINE_LOCK:
        image_bgr = _decode_or_422(contents)
        height, width = image_bgr.shape[:2]

        coords = {
            "forehead": (forehead_x, forehead_y),
            "left_cheek": (left_cheek_x, left_cheek_y),
            "right_cheek": (right_cheek_x, right_cheek_y),
        }
        form_values = (forehead_x, forehead_y, left_cheek_x, left_cheek_y, right_cheek_x, right_cheek_y, patch_half_size)
        # isfinite first: NaN compares False against everything, so it would
        # slip through the range checks below and crash patch sampling.
        if (
            not all(math.isfinite(v) for v in form_values)
            or patch_half_size <= 0
            or patch_half_size > min(width, height) * _MAX_PATCH_HALF_SIZE_RATIO
            or any(not (0 <= x <= width and 0 <= y <= height) for x, y in coords.values())
        ):
            raise HTTPException(status_code=400, detail="Those patch positions are out of bounds for this image.")

        anchors = AnchorPoints(
            forehead=np.array([forehead_x, forehead_y]),
            left_cheek=np.array([left_cheek_x, left_cheek_y]),
            right_cheek=np.array([right_cheek_x, right_cheek_y]),
            patch_half_size=patch_half_size,
        )
        sample = sample_skin_regions(image_bgr, anchors=anchors)
        del image_bgr  # free the full-resolution array before releasing the lock

    log_event("scan_manual_started")

    if not sample.success:
        if sample.error == "patch_clipped":
            log_event("scan_low_confidence", {"reason": "patch_clipped", "manual": True})
            raise HTTPException(
                status_code=422,
                detail=_low_confidence_detail(
                    "patch_clipped", _SAMPLE_ERROR_MESSAGES["patch_clipped"], anchors, width, height
                ),
            )
        if sample.error == "overexposed":
            log_event("scan_low_confidence", {"reason": "overexposed", "manual": True})
        # face_out_of_frame is unreachable in practice here: the bounds check
        # above guarantees every patch overlaps the image by at least one
        # pixel, which is all sample_patch_rgb needs to return a value. Kept
        # only for defensive completeness.
        raise HTTPException(status_code=422, detail=_SAMPLE_ERROR_MESSAGES[sample.error])

    result = _classify(sample)
    if not result.success:
        log_event(
            "scan_low_confidence",
            {"reason": result.error, "manual": True, **_scan_measurements(sample, result)},
        )
        if result.error == "color_cast":
            raise HTTPException(status_code=422, detail=_CLASSIFY_ERROR_MESSAGES["color_cast"])
        raise HTTPException(
            status_code=422,
            detail=_low_confidence_detail(
                "inconsistent_patches", _CLASSIFY_ERROR_MESSAGES["inconsistent_patches"], anchors, width, height
            ),
        )

    season = result.classification.season
    swatches = to_swatch_responses(SWATCHES_BY_SEASON[season])
    paragraph = generate_paragraph(result.classification, SWATCHES_BY_SEASON[season])
    scan_id = _persist_scan(season, swatches, paragraph)
    background_tasks.add_task(
        log_event,
        "scan_manual_completed",
        {"season": season, "scan_id": scan_id, **_scan_measurements(sample, result)},
    )
    return ScanResponse(scan_id=scan_id, season=season, swatches=swatches, paragraph=paragraph)


@router.post("/scans/{scan_id}/retake", response_model=ScanResponse)
def retake_scan(
    background_tasks: BackgroundTasks, scan_id: str, photo: UploadFile = File(...)
) -> ScanResponse:
    """Re-run the full pipeline against a new photo for a paid scan's one
    included free retake, overwriting that scan's stored result in place.

    The upfront paid/retake_used checks below are a UX fast-fail only — they
    let an obviously-doomed request fail before spending time decoding the
    photo and running the pipeline. They are NOT the concurrency guard: two
    requests can both pass them and both run the pipeline, so the actual
    safety property (at most one retake ever gets consumed) comes entirely
    from consume_retake's atomic conditional UPDATE, re-checked fresh after
    the pipeline finishes. A face-detection or classification failure (422)
    returns before consume_retake is ever called, so a failed attempt never
    consumes the retake and never touches the scan's existing result.
    """
    scan_id = parse_scan_id_or_404(scan_id)
    row = get_scan(scan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Scan not found.")
    if not row.paid:
        raise HTTPException(status_code=403, detail="This scan hasn't been unlocked yet.")
    if row.retake_used:
        raise HTTPException(status_code=409, detail="You've already used your free retake for this scan.")

    contents = _read_upload(photo)
    with _IMAGE_PIPELINE_LOCK:
        image_bgr = _decode_or_422(contents)
        sample = sample_skin_regions(image_bgr)
        del image_bgr  # free the full-resolution array before releasing the lock

    log_event("retake_started", {"scan_id": scan_id})

    if not sample.success:
        if sample.error in _RETAKE_ERROR_MESSAGES:
            log_event("scan_low_confidence", {"reason": sample.error, "retake": True})
            raise HTTPException(status_code=422, detail=_RETAKE_ERROR_MESSAGES[sample.error])
        raise HTTPException(status_code=422, detail=_SAMPLE_ERROR_MESSAGES[sample.error])

    result = _classify(sample)
    if not result.success:
        log_event(
            "scan_low_confidence",
            {"reason": result.error, "retake": True, **_scan_measurements(sample, result)},
        )
        raise HTTPException(status_code=422, detail=_RETAKE_ERROR_MESSAGES[result.error])

    season = result.classification.season
    swatches = to_swatch_responses(SWATCHES_BY_SEASON[season])
    paragraph = generate_paragraph(result.classification, SWATCHES_BY_SEASON[season])

    try:
        consumed = consume_retake(scan_id, season, [s.model_dump() for s in swatches], paragraph)
    except Exception:
        logger.exception("Failed to persist retake for scan %r", scan_id)
        raise HTTPException(
            status_code=503, detail="We couldn't save your retake right now. Please try again."
        )
    if not consumed:
        raise HTTPException(status_code=409, detail="You've already used your free retake for this scan.")

    background_tasks.add_task(
        log_event,
        "retake_completed",
        {"season": season, "scan_id": scan_id, **_scan_measurements(sample, result)},
    )
    return ScanResponse(scan_id=scan_id, season=season, swatches=swatches, paragraph=paragraph)
