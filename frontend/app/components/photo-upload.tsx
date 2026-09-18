"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CameraCapture } from "@/app/components/camera-capture";
import { HealthCheck } from "@/app/components/health-check";
import { PatchAdjuster } from "@/app/components/patch-adjuster";
import { CARD_CLASS, PRIMARY_BUTTON_CLASS, SECONDARY_BUTTON_CLASS } from "@/app/components/ui";
import {
  LowConfidenceScanError,
  ScanError,
  scanPhoto,
  scanPhotoWithPatches,
  type LowConfidenceReason,
  type PatchAnchors,
  type ScanImageInfo,
} from "@/lib/api";

// Dev/docker-compose wiring check only (see README) — never render backend
// internals (service/version/environment) to real users.
const SHOW_HEALTH_CHECK = process.env.NODE_ENV !== "production";

const LIGHTING_TIPS = [
  "Face a window in daylight — indirect light, not direct sun on your face",
  "Turn off camera flash and skip overhead room lights",
  "Take off glasses, hats, and heavy foundation or bronzer",
  "Pull hair back off your forehead and cheeks",
  "Hold the phone at eye level, about an arm's length away",
] as const;

// Face detection and color classification (backend/app/vision) are fast and
// close to constant-time regardless of the photo. What actually makes scans
// take anywhere from ~1s to ~15s is the personalized write-up: both the
// fresh-scan and rescan endpoints await a live Anthropic API call inline
// before responding, and that call's latency varies with the API itself
// (worst case: an 8s timeout plus one retry). These messages reflect that —
// a fixed sequence for the fast, true stages, then an honest indefinite pool
// once we're plausibly waiting on the write-up.
const EARLY_SCAN_MESSAGES = [
  "Finding your face…",
  "Reading your undertones…",
  "Matching your season…",
] as const;
const EARLY_INTERVAL_MS = 1100;

const RESCAN_MESSAGES = ["Rechecking your adjusted spots…"] as const;
const RESCAN_INTERVAL_MS = 1400;

const LATER_STATUS_MESSAGES = [
  "Writing something personal about your colors…",
  "Almost there…",
  "Just putting on the finishing touches…",
] as const;
const LATER_INTERVAL_MS = 2600;

// Held for at least as long as the fast-stage messages get a full turn, so
// the overlay never flashes even when the backend short-circuits quickly
// (e.g. a low-confidence photo, which skips the AI call entirely).
const MIN_SCANNING_MS = EARLY_SCAN_MESSAGES.length * EARLY_INTERVAL_MS;

type State =
  | { kind: "empty" }
  | { kind: "camera" }
  | { kind: "scanning"; previewUrl: string }
  | {
      kind: "review";
      file: File;
      previewUrl: string;
      reason: LowConfidenceReason;
      message: string;
      patches: PatchAnchors;
      image: ScanImageInfo;
    }
  | { kind: "rescanning"; file: File; previewUrl: string }
  | { kind: "error"; message: string; previewUrl: string };

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Devices with a coarse (touch) primary pointer get the OS camera app via
// the file input's capture="user" hint — it's the native, well-supported
// path there. Devices with a fine pointer (typically no touch camera-app
// hint support at all, e.g. desktop browsers) get an in-page live capture
// instead, falling back to it too if getUserMedia simply isn't available.
function prefersNativeCameraCapture(): boolean {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) return true;
  if (typeof window === "undefined" || !window.matchMedia) return true;
  return window.matchMedia("(pointer: coarse)").matches;
}

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

// Shared blurred-preview-plus-progress treatment for both the initial scan
// and a patch-adjustment resubmit — both hit the same slow, variable AI
// write-up call server-side, so both get the same easing progress bar and
// the same "fast lead messages, then an honest indefinite pool" messaging,
// just with different lead messages for the two cases.
function ScanningOverlay({
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

export function PhotoUpload() {
  const [state, setState] = useState<State>({ kind: "empty" });
  const selfieInputRef = useRef<HTMLInputElement>(null);
  const libraryInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  async function processFile(file: File) {
    if (state.kind !== "empty" && state.kind !== "camera") {
      URL.revokeObjectURL(state.previewUrl);
    }
    const previewUrl = URL.createObjectURL(file);
    setState({ kind: "scanning", previewUrl });

    try {
      const [result] = await Promise.all([scanPhoto(file), wait(MIN_SCANNING_MS)]);
      URL.revokeObjectURL(previewUrl);
      router.push(`/result/${result.scanId}`);
    } catch (error) {
      if (error instanceof LowConfidenceScanError) {
        setState({
          kind: "review",
          file,
          previewUrl,
          reason: error.reason,
          message: error.message,
          patches: error.patches,
          image: error.image,
        });
        return;
      }
      const message =
        error instanceof ScanError ? error.message : "Something went wrong. Please try again.";
      setState({ kind: "error", message, previewUrl });
    }
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = ""; // allow re-selecting the same file later
    if (!file) return;
    await processFile(file);
  }

  function triggerSelfieCapture() {
    if (prefersNativeCameraCapture()) {
      selfieInputRef.current?.click();
    } else {
      setState({ kind: "camera" });
    }
  }

  async function handlePatchSubmit(patches: PatchAnchors) {
    if (state.kind !== "review") return;
    const { file, previewUrl } = state;
    setState({ kind: "rescanning", file, previewUrl });

    try {
      const [result] = await Promise.all([
        scanPhotoWithPatches(file, patches),
        wait(MIN_SCANNING_MS),
      ]);
      URL.revokeObjectURL(previewUrl);
      router.push(`/result/${result.scanId}`);
    } catch (error) {
      if (error instanceof LowConfidenceScanError) {
        setState({
          kind: "review",
          file,
          previewUrl,
          reason: error.reason,
          message: error.message,
          patches: error.patches,
          image: error.image,
        });
        return;
      }
      const message =
        error instanceof ScanError ? error.message : "Something went wrong. Please try again.";
      setState({ kind: "error", message, previewUrl });
    }
  }

  return (
    <div className="space-y-4">
      <input
        ref={selfieInputRef}
        type="file"
        accept="image/*"
        capture="user"
        onChange={handleFileChange}
        className="hidden"
      />
      <input
        ref={libraryInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {state.kind === "empty" && (
        <div className="space-y-4">
          <div className={CARD_CLASS}>
            <p className="text-sm font-medium">Before you scan</p>
            <ul className="mt-2 space-y-1.5 text-sm text-black/60 dark:text-white/60">
              {LIGHTING_TIPS.map((tip) => (
                <li key={tip} className="flex gap-2">
                  <span aria-hidden className="text-black/30 dark:text-white/30">
                    &middot;
                  </span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-2">
            <button type="button" onClick={triggerSelfieCapture} className={PRIMARY_BUTTON_CLASS}>
              Take a selfie
            </button>
            <button
              type="button"
              onClick={() => libraryInputRef.current?.click()}
              className={SECONDARY_BUTTON_CLASS}
            >
              Choose from library
            </button>
          </div>

          {SHOW_HEALTH_CHECK && <HealthCheck />}
        </div>
      )}

      {state.kind === "camera" && (
        <CameraCapture onCapture={processFile} onCancel={() => setState({ kind: "empty" })} />
      )}

      {state.kind === "scanning" && (
        <ScanningOverlay previewUrl={state.previewUrl} leadMessages={EARLY_SCAN_MESSAGES} />
      )}

      {state.kind === "review" && (
        <PatchAdjuster
          previewUrl={state.previewUrl}
          image={state.image}
          initialPatches={state.patches}
          message={state.message}
          onSubmit={handlePatchSubmit}
          onRetakeSelfie={triggerSelfieCapture}
          onRetakeLibrary={() => libraryInputRef.current?.click()}
        />
      )}

      {state.kind === "rescanning" && (
        <ScanningOverlay
          previewUrl={state.previewUrl}
          leadMessages={RESCAN_MESSAGES}
          leadIntervalMs={RESCAN_INTERVAL_MS}
        />
      )}

      {state.kind === "error" && (
        <div className="animate-fade-in space-y-4">
          <div
            role="alert"
            className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-300"
          >
            {state.message}
          </div>
          <div className="space-y-2">
            <button type="button" onClick={triggerSelfieCapture} className={PRIMARY_BUTTON_CLASS}>
              Take a selfie
            </button>
            <button
              type="button"
              onClick={() => libraryInputRef.current?.click()}
              className={SECONDARY_BUTTON_CLASS}
            >
              Choose from library
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
