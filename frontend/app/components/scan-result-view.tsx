"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import {
  ScanError,
  ScanNotFoundError,
  createCheckoutSession,
  getScan,
  type ScanState,
  type ScanSwatch,
} from "@/lib/api";
import { ColorChip } from "./color-chip";
import { FullReportView } from "./full-report-view";
import { RetakeCapture } from "./retake-capture";
import { PRIMARY_BUTTON_CLASS, SPINNER_CLASS } from "./ui";

type ViewState =
  | { kind: "loading" }
  | { kind: "loaded"; scan: ScanState }
  | { kind: "error"; message: string };

function SwatchGrid({ swatches }: { swatches: ScanSwatch[] }) {
  return (
    <div className="flex flex-wrap justify-center gap-4">
      {swatches.map((swatch) => (
        <ColorChip key={swatch.hex} swatch={swatch} />
      ))}
    </div>
  );
}

const START_OVER_LINK = (
  <Link
    href="/"
    className="block w-full rounded-xl border border-black/10 px-4 py-3 text-center text-sm font-medium dark:border-white/15"
  >
    Scan another photo
  </Link>
);

export function ScanResultView() {
  const params = useParams<{ scanId: string }>();
  const searchParams = useSearchParams();
  const scanId = params.scanId;
  const sessionId = searchParams.get("session_id");

  const [state, setState] = useState<ViewState>({ kind: "loading" });
  const [isRedirecting, setIsRedirecting] = useState(false);

  useEffect(() => {
    // No synchronous setState("loading") here on purpose — initial state is
    // already "loading", and this component is always freshly mounted for
    // a given scanId (router.push from a scan, or a fresh page load from
    // Stripe's redirect), so there's no stale-state case to reset from.
    let cancelled = false;

    getScan(scanId, sessionId ?? undefined)
      .then((scan) => {
        if (!cancelled) setState({ kind: "loaded", scan });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const message =
          error instanceof ScanNotFoundError
            ? "We couldn't find that scan. It may have expired, or the link is wrong."
            : error instanceof ScanError
              ? error.message
              : "Something went wrong loading that result. Please try again.";
        setState({ kind: "error", message });
      });

    return () => {
      cancelled = true;
    };
  }, [scanId, sessionId]);

  async function handleUnlock() {
    setIsRedirecting(true);
    try {
      const { checkoutUrl } = await createCheckoutSession(scanId);
      // Hard navigation to Stripe's hosted page, not router.push — this is
      // leaving the app entirely.
      window.location.href = checkoutUrl;
    } catch {
      setIsRedirecting(false);
      setState({
        kind: "error",
        message: "We couldn't start checkout right now. Please try again.",
      });
    }
  }

  if (state.kind === "loading") {
    return (
      <div className="flex justify-center py-12">
        <span aria-hidden className={SPINNER_CLASS} />
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="animate-fade-in space-y-4">
        <div
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-300"
        >
          {state.message}
        </div>
        {START_OVER_LINK}
      </div>
    );
  }

  const { scan } = state;

  return (
    <div className="animate-fade-in space-y-5">
      <div className="space-y-1">
        <p className="text-sm font-medium text-black/60 dark:text-white/60">
          {scan.paid ? "Full report unlocked — your color season is" : "Your color season is"}
        </p>
        <h2 className="text-3xl font-semibold tracking-tight">{scan.season}</h2>
      </div>

      {scan.paid && scan.fullReport ? (
        <>
          <FullReportView fullReport={scan.fullReport} />
          {!scan.retakeUsed && (
            <RetakeCapture
              scanId={scan.scanId}
              onSuccess={(updatedScan) => setState({ kind: "loaded", scan: updatedScan })}
            />
          )}
        </>
      ) : (
        <>
          {scan.paragraph && (
            <p className="text-sm leading-relaxed text-black/70 dark:text-white/70">{scan.paragraph}</p>
          )}
          <SwatchGrid swatches={scan.swatches} />
          <button
            type="button"
            onClick={handleUnlock}
            disabled={isRedirecting}
            className={PRIMARY_BUTTON_CLASS}
          >
            {isRedirecting
              ? "Redirecting to checkout…"
              : `Unlock full report — $${(scan.priceCents / 100).toFixed(2)}`}
          </button>
        </>
      )}

      {START_OVER_LINK}
    </div>
  );
}
