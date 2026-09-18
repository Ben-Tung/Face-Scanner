"use client";

import { useEffect, useState } from "react";

// Devices with a coarse (touch) primary pointer get the OS camera app via
// the file input's capture="user" hint — it's the native, well-supported
// path there. Devices with a fine pointer (typically no touch camera-app
// hint support at all, e.g. desktop browsers) get an in-page live capture
// instead, falling back to it too if getUserMedia simply isn't available.
export function prefersNativeCameraCapture(): boolean {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) return true;
  if (typeof window === "undefined" || !window.matchMedia) return true;
  return window.matchMedia("(pointer: coarse)").matches;
}

export function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Face detection and color classification (backend/app/vision) are fast and
// close to constant-time regardless of the photo. What actually makes scans
// take anywhere from ~1s to ~15s is the personalized write-up: the scan,
// rescan, and retake endpoints all await a live Anthropic API call inline
// before responding, and that call's latency varies with the API itself
// (worst case: an 8s timeout plus one retry). These messages reflect that —
// a fixed sequence for the fast, true stages, then an honest indefinite pool
// once we're plausibly waiting on the write-up.
export const EARLY_SCAN_MESSAGES = [
  "Finding your face…",
  "Reading your undertones…",
  "Matching your season…",
] as const;
export const EARLY_INTERVAL_MS = 1100;

export const RESCAN_MESSAGES = ["Rechecking your adjusted spots…"] as const;
export const RESCAN_INTERVAL_MS = 1400;

const LATER_STATUS_MESSAGES = [
  "Writing something personal about your colors…",
  "Almost there…",
  "Just putting on the finishing touches…",
] as const;
const LATER_INTERVAL_MS = 2600;

// Held for at least as long as the fast-stage messages get a full turn, so
// the overlay never flashes even when the backend short-circuits quickly
// (e.g. a low-confidence photo, which skips the AI call entirely).
export const MIN_SCANNING_MS = EARLY_SCAN_MESSAGES.length * EARLY_INTERVAL_MS;

// Walks through `leadMessages` at `leadIntervalMs` once, then loops
// LATER_STATUS_MESSAGES indefinitely at the slower LATER_INTERVAL_MS — the
// lead messages match real, fast, near-constant backend stages, so they
// only need to play once; the later pool covers however long the AI
// write-up call ends up taking, without ever implying it's stuck or restart
// the story from the top.
function useStatusMessage(leadMessages: readonly string[], leadIntervalMs: number): string {
  const [index, setIndex] = useState(0);
  const leadCount = leadMessages.length;
  const totalCount = leadCount + LATER_STATUS_MESSAGES.length;

  useEffect(() => {
    const holdMs = index < leadCount ? leadIntervalMs : LATER_INTERVAL_MS;
    const id = setTimeout(() => {
      setIndex((i) => (i + 1 < totalCount ? i + 1 : leadCount));
    }, holdMs);
    return () => clearTimeout(id);
  }, [index, leadCount, totalCount, leadIntervalMs]);

  return index < leadCount ? leadMessages[index] : LATER_STATUS_MESSAGES[index - leadCount];
}

// A fresh key (the scan's previewUrl, always unique) mounts a new instance
// of this per scan, so the cycling status naturally restarts at index 0
// without needing to reset state from inside the effect.
function ScanStatusMessage({
  leadMessages,
  leadIntervalMs,
}: {
  leadMessages: readonly string[];
  leadIntervalMs: number;
}) {
  const message = useStatusMessage(leadMessages, leadIntervalMs);
  return (
    <p key={message} className="animate-fade-in text-sm font-medium" aria-live="polite">
      {message}
    </p>
  );
}

// A simulated, not backend-driven, progress indicator: the backend has no
// streaming/progress signal today (a single plain request/response), and
// its one genuinely slow, variable step (the inline AI write-up call) isn't
// otherwise observable mid-flight. Easing up toward — but deliberately
// never reaching — full over several seconds reads as real, ongoing
// progress without claiming a precision we don't have; it settles near-full
// and just holds there until the real response arrives and this unmounts.
function ScanProgressBar() {
  const [filled, setFilled] = useState(false);

  useEffect(() => {
    const id = setTimeout(() => setFilled(true), 50);
    return () => clearTimeout(id);
  }, []);

  return (
    <div
      aria-hidden
      className="h-1.5 w-full overflow-hidden rounded-full bg-black/10 dark:bg-white/10"
    >
      <div
        className="h-full rounded-full bg-black transition-[width] duration-[8000ms] ease-out dark:bg-white"
        style={{ width: filled ? "92%" : "6%" }}
      />
    </div>
  );
}

// Shared blurred-preview-plus-progress treatment for every flow that awaits
// the backend's slow, variable AI write-up call (initial scan, a patch-
// adjustment resubmit, and a paid scan's retake) — all get the same easing
// progress bar and the same "fast lead messages, then an honest indefinite
// pool" messaging, just with different lead messages per case.
export function ScanningOverlay({
  previewUrl,
  leadMessages,
  leadIntervalMs = EARLY_INTERVAL_MS,
}: {
  previewUrl: string;
  leadMessages: readonly string[];
  leadIntervalMs?: number;
}) {
  return (
    <div className="relative aspect-square w-full overflow-hidden rounded-xl">
      {/* eslint-disable-next-line @next/next/no-img-element -- transient local object URL, not worth Image's remote-optimization pipeline */}
      <img
        src={previewUrl}
        alt=""
        aria-hidden
        className="h-full w-full scale-105 object-cover opacity-40 blur-sm"
      />
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
        <div className="w-full max-w-[220px]">
          <ScanProgressBar key={previewUrl} />
        </div>
        <ScanStatusMessage
          key={previewUrl}
          leadMessages={leadMessages}
          leadIntervalMs={leadIntervalMs}
        />
      </div>
    </div>
  );
}
