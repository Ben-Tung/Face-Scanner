"""Response models shared across routers.

Split out of app/routers/scan.py so app/routers/payments.py can reuse the
same swatch shape without one router importing from another.
"""

from __future__ import annotations

from pydantic import BaseModel


class SwatchResponse(BaseModel):
    name: str
    hex: str
