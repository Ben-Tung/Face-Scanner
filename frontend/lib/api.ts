const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Build an absolute URL for a backend API path, e.g. apiUrl("/api/health"). */
export function apiUrl(path: string): string {
  return `${API_BASE_URL.replace(/\/$/, "")}${path.startsWith("/") ? path : `/${path}`}`;
}

export type Season = "Spring" | "Summer" | "Autumn" | "Winter";

export type ScanSwatch = {
  name: string;
  hex: string;
};

export type ScanResult = {
  scanId: string;
  season: Season;
  swatches: ScanSwatch[];
  paragraph: string | null;
};

export type SeasonPalette = {
  best: ScanSwatch[];
  good: ScanSwatch[];
  avoid: ScanSwatch[];
};

export type MakeupGuidance = {
  foundationUndertone: string;
  foundationTip: string;
  lipShades: ScanSwatch[];
  blushShades: ScanSwatch[];
};

export type JewelryMetal = "Gold" | "Silver" | "Both";

export type JewelryGuidance = {
  metal: JewelryMetal;
  tip: string;
  goldSwatch: ScanSwatch;
  silverSwatch: ScanSwatch;
};

export type FullReport = {
  palette: SeasonPalette;
  makeup: MakeupGuidance;
  jewelry: JewelryGuidance;
  shoppingGuidance: string;
  paragraph: string | null;
};

export type ScanState = {
  scanId: string;
  season: Season;
  swatches: ScanSwatch[];
  paragraph: string | null;
  paid: boolean;
  retakeUsed: boolean;
  priceCents: number;
  fullReport: FullReport | null;
};

export type PatchPoint = {
  x: number;
  y: number;
};

export type PatchAnchors = {
  forehead: PatchPoint;
  leftCheek: PatchPoint;
  rightCheek: PatchPoint;
  patchHalfSize: number;
};

export type ScanImageInfo = {
  width: number;
  height: number;
};

export type LowConfidenceReason = "patch_clipped" | "inconsistent_patches";

export class ScanError extends Error {}

/** A scan id that doesn't exist (bad link, or the scan was never
 * persisted) — kept distinct from ScanError so the result page can show a
 * friendlier "we couldn't find that scan" message instead of a generic
 * error. */
export class ScanNotFoundError extends ScanError {}

/** A recoverable low-confidence result: carries the auto-detected patch
 * coordinates so the caller can offer manual adjustment instead of just
 * asking the user to retake the photo. */
export class LowConfidenceScanError extends ScanError {
  constructor(
    message: string,
    public readonly reason: LowConfidenceReason,
    public readonly patches: PatchAnchors,
    public readonly image: ScanImageInfo,
  ) {
    super(message);
  }
}

type LowConfidenceDetailBody = {
  reason: LowConfidenceReason;
  message: string;
  patches: {
    forehead: PatchPoint;
    left_cheek: PatchPoint;
    right_cheek: PatchPoint;
    patch_half_size: number;
  };
  image: ScanImageInfo;
};

async function parseScanErrorBody(response: Response): Promise<Error> {
  const fallback = "Something went wrong scanning that photo. Please try again.";
  let body: { detail?: unknown };
  try {
    body = (await response.json()) as { detail?: unknown };
  } catch {
    return new ScanError(fallback);
  }

  const detail = body.detail;
  if (typeof detail === "string") return new ScanError(detail);

  // FastAPI's own request-validation errors return `detail` as an array of
  // objects rather than our structured shape — only match on the fields we
  // actually send for a low-confidence result.
  if (detail && typeof detail === "object" && "patches" in detail && "reason" in detail) {
    const d = detail as LowConfidenceDetailBody;
    return new LowConfidenceScanError(
      d.message,
      d.reason,
      {
        forehead: d.patches.forehead,
        leftCheek: d.patches.left_cheek,
        rightCheek: d.patches.right_cheek,
        patchHalfSize: d.patches.patch_half_size,
      },
      d.image,
    );
  }

  return new ScanError(fallback);
}

type ScanResultBody = {
  scan_id: string;
  season: Season;
  swatches: ScanSwatch[];
  paragraph: string | null;
};

function mapScanResult(body: ScanResultBody): ScanResult {
  return {
    scanId: body.scan_id,
    season: body.season,
    swatches: body.swatches,
    paragraph: body.paragraph,
  };
}

/** Upload a selfie to the backend and get back its color season + example swatches. */
export async function scanPhoto(file: File): Promise<ScanResult> {
  const formData = new FormData();
  formData.append("photo", file);

  // No Content-Type header here on purpose — the browser sets the
  // multipart boundary itself when the body is a FormData instance.
  const response = await fetch(apiUrl("/api/scan"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw await parseScanErrorBody(response);
  }

  return mapScanResult((await response.json()) as ScanResultBody);
}

/** Re-run classification against manually adjusted patch coordinates, for
 * the low-confidence recovery flow. Resends the original file since the
 * backend never retains an uploaded photo between requests. */
export async function scanPhotoWithPatches(file: File, patches: PatchAnchors): Promise<ScanResult> {
  const formData = new FormData();
  formData.append("photo", file);
  formData.append("forehead_x", String(patches.forehead.x));
  formData.append("forehead_y", String(patches.forehead.y));
  formData.append("left_cheek_x", String(patches.leftCheek.x));
  formData.append("left_cheek_y", String(patches.leftCheek.y));
  formData.append("right_cheek_x", String(patches.rightCheek.x));
  formData.append("right_cheek_y", String(patches.rightCheek.y));
  formData.append("patch_half_size", String(patches.patchHalfSize));

  const response = await fetch(apiUrl("/api/scan/manual"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw await parseScanErrorBody(response);
  }

  return mapScanResult((await response.json()) as ScanResultBody);
}

type FullReportBody = {
  palette: { best: ScanSwatch[]; good: ScanSwatch[]; avoid: ScanSwatch[] };
  makeup: {
    foundation_undertone: string;
    foundation_tip: string;
    lip_shades: ScanSwatch[];
    blush_shades: ScanSwatch[];
  };
  jewelry: { metal: JewelryMetal; tip: string; gold_swatch: ScanSwatch; silver_swatch: ScanSwatch };
  shopping_guidance: string;
  paragraph: string | null;
};

type ScanStateBody = {
  scan_id: string;
  season: Season;
  swatches: ScanSwatch[];
  paragraph: string | null;
  paid: boolean;
  retake_used: boolean;
  price_cents: number;
  full_report: FullReportBody | null;
};

function mapFullReport(body: FullReportBody): FullReport {
  return {
    palette: body.palette,
    makeup: {
      foundationUndertone: body.makeup.foundation_undertone,
      foundationTip: body.makeup.foundation_tip,
      lipShades: body.makeup.lip_shades,
      blushShades: body.makeup.blush_shades,
    },
    jewelry: {
      metal: body.jewelry.metal,
      tip: body.jewelry.tip,
      goldSwatch: body.jewelry.gold_swatch,
      silverSwatch: body.jewelry.silver_swatch,
    },
    shoppingGuidance: body.shopping_guidance,
    paragraph: body.paragraph,
  };
}

/** Fetch a scan's current state by id — the free result plus whether it's
 * been paid for. Pass sessionId (Stripe's session_id query param on the
 * redirect back from Checkout) so the backend can self-heal `paid` from
 * Stripe directly if the webhook hasn't landed yet. */
export async function getScan(scanId: string, sessionId?: string): Promise<ScanState> {
  const url = new URL(apiUrl(`/api/scans/${scanId}`));
  if (sessionId) url.searchParams.set("session_id", sessionId);

  const response = await fetch(url);

  if (response.status === 404) {
    throw new ScanNotFoundError("We couldn't find that scan.");
  }
  if (!response.ok) {
    throw new ScanError("Something went wrong loading that result. Please try again.");
  }

  const body = (await response.json()) as ScanStateBody;
  return {
    scanId: body.scan_id,
    season: body.season,
    swatches: body.swatches,
    paragraph: body.paragraph,
    paid: body.paid,
    retakeUsed: body.retake_used,
    priceCents: body.price_cents,
    fullReport: body.full_report ? mapFullReport(body.full_report) : null,
  };
}

/** Consume a paid scan's one free retake: re-run the pipeline against a new
 * photo and overwrite that scan's stored result. The retake endpoint itself
 * only returns the free-tier shape (season/swatches/paragraph) — the paid
 * full report is entirely season-derived server-side, so this re-fetches
 * getScan afterward to get a fully consistent result, including a freshly
 * regenerated (not stale) AI paragraph for the new season. */
export async function retakeScan(scanId: string, file: File): Promise<ScanState> {
  const formData = new FormData();
  formData.append("photo", file);

  const response = await fetch(apiUrl(`/api/scans/${scanId}/retake`), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw await parseScanErrorBody(response);
  }

  return getScan(scanId);
}

/** Start a one-time Stripe Checkout session to unlock a scan's full
 * report. Returns the hosted Checkout URL to redirect the browser to. */
export async function createCheckoutSession(scanId: string): Promise<{ checkoutUrl: string }> {
  const response = await fetch(apiUrl(`/api/scans/${scanId}/checkout`), { method: "POST" });

  if (!response.ok) {
    throw new ScanError("We couldn't start checkout right now. Please try again.");
  }

  const body = (await response.json()) as { checkout_url: string };
  return { checkoutUrl: body.checkout_url };
}
