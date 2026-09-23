from __future__ import annotations

import logging
from typing import Literal
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from app.analytics import log_event
from app.palettes import SWATCHES_BY_SEASON
from app.paragraph import generate_paragraph
from app.rate_limit import limiter, scan_limit_value
from app.scans_repo import consume_retake, create_scan, get_scan
from app.schemas import SwatchResponse, parse_scan_id_or_404, to_swatch_responses
from app.vision.season_classifier import Season, classify_season
from app.vision.skin_sampling import AnchorPoints, sample_at_anchors, sample_skin_regions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["scan"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — generous for a phone camera selfie

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
}
_CLASSIFY_ERROR_MESSAGES: dict[str, str] = {
    "inconsistent_patches": (
        "Lighting looks uneven across your face — drag the boxes below to adjust, "
        "or retake the photo in even light."
    ),
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


async def _read_and_decode_photo(photo: UploadFile) -> np.ndarray:
    """Validate, read, and decode an upload into a BGR array.

    Never retains the uploaded photo beyond the caller's request — nothing
    here writes it to disk or a database (see CLAUDE.md's retention rule).
    """
    if photo.content_type and not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Please upload an image file.")

    contents = await photo.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Please upload an image file.")
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="That photo is too large (max 10MB). Try a smaller photo.")

    image_bgr = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
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
async def scan(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    photo: UploadFile = File(...),
) -> ScanResponse:
    """Detect a face in the uploaded photo and classify its color season.

    Never retains the uploaded photo beyond this request — nothing here
    writes it to disk or a database (see CLAUDE.md's retention rule).
    """
    image_bgr = await _read_and_decode_photo(photo)

    # Called directly, not via background_tasks: FastAPI only attaches queued
    # background tasks to a successful Response, never to the response an
    # HTTPException raised further down would build — so a backgrounded call
    # here would silently never fire on a rejected/failed scan. log_event()
    # is itself try/except-wrapped and timeout-bounded, so this stays cheap.
    log_event("scan_started")

    height, width = image_bgr.shape[:2]

    sample = sample_skin_regions(image_bgr)
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
        raise HTTPException(status_code=422, detail=_SAMPLE_ERROR_MESSAGES[sample.error])

    sclera_rgb = sample.sclera.sclera_rgb if sample.sclera is not None and sample.sclera.success else None
    result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sclera_rgb)
    if not result.success:
        assert sample.anchors is not None  # sampling succeeded, so anchors are always set
        log_event("scan_low_confidence", {"reason": "inconsistent_patches"})
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
        {"season": season, "scan_id": scan_id, "depth_normalized": sclera_rgb is not None},
    )
    return ScanResponse(scan_id=scan_id, season=season, swatches=swatches, paragraph=paragraph)


@router.post("/scan/manual", response_model=ScanResponse)
@limiter.limit(scan_limit_value)
async def scan_manual(
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
    """
    image_bgr = await _read_and_decode_photo(photo)
    height, width = image_bgr.shape[:2]

    coords = {
        "forehead": (forehead_x, forehead_y),
        "left_cheek": (left_cheek_x, left_cheek_y),
        "right_cheek": (right_cheek_x, right_cheek_y),
    }
    if patch_half_size <= 0 or any(
        not (0 <= x <= width and 0 <= y <= height) for x, y in coords.values()
    ):
        raise HTTPException(status_code=400, detail="Those patch positions are out of bounds for this image.")

    anchors = AnchorPoints(
        forehead=np.array([forehead_x, forehead_y]),
        left_cheek=np.array([left_cheek_x, left_cheek_y]),
        right_cheek=np.array([right_cheek_x, right_cheek_y]),
        patch_half_size=patch_half_size,
    )

    log_event("scan_manual_started")

    sample = sample_at_anchors(image_bgr, anchors)
    if not sample.success:
        if sample.error == "patch_clipped":
            log_event("scan_low_confidence", {"reason": "patch_clipped", "manual": True})
            raise HTTPException(
                status_code=422,
                detail=_low_confidence_detail(
                    "patch_clipped", _SAMPLE_ERROR_MESSAGES["patch_clipped"], anchors, width, height
                ),
            )
        # face_out_of_frame is unreachable in practice here: the bounds check
        # above guarantees every patch overlaps the image by at least one
        # pixel, which is all sample_patch_rgb needs to return a value. Kept
        # only for defensive completeness.
        raise HTTPException(status_code=422, detail=_SAMPLE_ERROR_MESSAGES[sample.error])

    result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb)
    if not result.success:
        log_event("scan_low_confidence", {"reason": "inconsistent_patches", "manual": True})
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
        log_event, "scan_manual_completed", {"season": season, "scan_id": scan_id}
    )
    return ScanResponse(scan_id=scan_id, season=season, swatches=swatches, paragraph=paragraph)


@router.post("/scans/{scan_id}/retake", response_model=ScanResponse)
async def retake_scan(
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

    image_bgr = await _read_and_decode_photo(photo)
    log_event("retake_started", {"scan_id": scan_id})
    height, width = image_bgr.shape[:2]

    sample = sample_skin_regions(image_bgr)
    if not sample.success:
        if sample.error == "patch_clipped":
            assert sample.anchors is not None  # patch_clipped always carries anchors
            log_event("scan_low_confidence", {"reason": "patch_clipped", "retake": True})
            raise HTTPException(
                status_code=422,
                detail=_low_confidence_detail(
                    "patch_clipped", _SAMPLE_ERROR_MESSAGES["patch_clipped"], sample.anchors, width, height
                ),
            )
        raise HTTPException(status_code=422, detail=_SAMPLE_ERROR_MESSAGES[sample.error])

    sclera_rgb = sample.sclera.sclera_rgb if sample.sclera is not None and sample.sclera.success else None
    result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sclera_rgb)
    if not result.success:
        assert sample.anchors is not None  # sampling succeeded, so anchors are always set
        log_event("scan_low_confidence", {"reason": "inconsistent_patches", "retake": True})
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

    try:
        consumed = consume_retake(scan_id, season, [s.model_dump() for s in swatches], paragraph)
    except Exception:
        logger.exception("Failed to persist retake for scan %r", scan_id)
        raise HTTPException(
            status_code=503, detail="We couldn't save your retake right now. Please try again."
        )
    if not consumed:
        raise HTTPException(status_code=409, detail="You've already used your free retake for this scan.")

    background_tasks.add_task(log_event, "retake_completed", {"season": season, "scan_id": scan_id})
    return ScanResponse(scan_id=scan_id, season=season, swatches=swatches, paragraph=paragraph)
