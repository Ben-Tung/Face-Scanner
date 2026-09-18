"""Response models shared across routers.

Split out of app/routers/scan.py so app/routers/payments.py can reuse the
same swatch shape without one router importing from another.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Protocol

from fastapi import HTTPException
from pydantic import BaseModel


class SwatchResponse(BaseModel):
    name: str
    hex: str


class _NamedHexSwatch(Protocol):
    name: str
    hex: str


def to_swatch_responses(swatches: Sequence[_NamedHexSwatch]) -> list[SwatchResponse]:
    return [SwatchResponse(name=s.name, hex=s.hex) for s in swatches]


def parse_scan_id_or_404(scan_id: str) -> str:
    try:
        return str(uuid.UUID(scan_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Scan not found.")
