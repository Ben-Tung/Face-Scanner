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
  season: Season;
  swatches: ScanSwatch[];
  paragraph: string | null;
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

  return (await response.json()) as ScanResult;
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

  return (await response.json()) as ScanResult;
}
