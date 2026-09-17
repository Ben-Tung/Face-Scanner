"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CameraCapture } from "@/app/components/camera-capture";
import { HealthCheck } from "@/app/components/health-check";
import { PatchAdjuster } from "@/app/components/patch-adjuster";
import {
  CARD_CLASS,
  PRIMARY_BUTTON_CLASS,
  SECONDARY_BUTTON_CLASS,
  SPINNER_CLASS,
} from "@/app/components/ui";
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

const SCAN_STATUS_MESSAGES = [
  "Finding your face…",
  "Reading your undertones…",
  "Matching your season…",
] as const;
const STATUS_INTERVAL_MS = 650;
// Held for at least as long as every status message gets a turn, so the
// animation never flashes even though the actual classification is fast.
const MIN_SCANNING_MS = SCAN_STATUS_MESSAGES.length * STATUS_INTERVAL_MS;

const RESCAN_STATUS_MESSAGE = "Rechecking your adjusted spots…";

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

// A fresh key (the scan's previewUrl, always unique) mounts a new instance
// of this per scan, so the cycling status naturally restarts at index 0
// without needing to reset state from inside the effect.
function ScanStatusMessage() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(
      () => setIndex((i) => (i + 1) % SCAN_STATUS_MESSAGES.length),
      STATUS_INTERVAL_MS,
    );
    return () => clearInterval(id);
  }, []);

  const message = SCAN_STATUS_MESSAGES[index];
  return (
    <p key={message} className="animate-fade-in text-sm font-medium" aria-live="polite">
      {message}
    </p>
  );
}

// Shared blurred-preview-plus-spinner treatment for both the initial scan
// and a patch-adjustment resubmit. A static `statusMessage` (the resubmit
// case — we already know where the face is, so the "Finding your face…"
// cycle doesn't fit) skips the cycling text; omitting it keeps the
// original cycling behavior for a fresh scan.
function ScanningOverlay({
  previewUrl,
  statusMessage,
}: {
  previewUrl: string;
  statusMessage?: string;
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
        <span className={SPINNER_CLASS} />
        {statusMessage ? (
          <p className="animate-fade-in text-sm font-medium" aria-live="polite">
            {statusMessage}
          </p>
        ) : (
          <ScanStatusMessage key={previewUrl} />
        )}
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

      {state.kind === "scanning" && <ScanningOverlay previewUrl={state.previewUrl} />}

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
        <ScanningOverlay previewUrl={state.previewUrl} statusMessage={RESCAN_STATUS_MESSAGE} />
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
