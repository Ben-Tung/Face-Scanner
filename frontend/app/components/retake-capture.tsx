"use client";

import { useRef, useState } from "react";

import { CameraCapture } from "@/app/components/camera-capture";
import {
  EARLY_SCAN_MESSAGES,
  MIN_SCANNING_MS,
  ScanningOverlay,
  prefersNativeCameraCapture,
  wait,
} from "@/app/components/scanning-overlay";
import { CARD_CLASS, PRIMARY_BUTTON_CLASS, SECONDARY_BUTTON_CLASS } from "@/app/components/ui";
import { ScanError, retakeScan, type ScanState } from "@/lib/api";

type State =
  | { kind: "idle" }
  | { kind: "camera" }
  | { kind: "submitting"; previewUrl: string }
  | { kind: "error"; message: string };

// A low-confidence photo here just surfaces as a plain error (below) rather
// than the full drag-to-adjust PatchAdjuster recovery flow the original scan
// gets — the retake endpoint never consumes the free retake on a failed
// attempt, so the user can simply pick another photo and try again.
export function RetakeCapture({
  scanId,
  onSuccess,
}: {
  scanId: string;
  onSuccess: (scan: ScanState) => void;
}) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const selfieInputRef = useRef<HTMLInputElement>(null);
  const libraryInputRef = useRef<HTMLInputElement>(null);

  async function processFile(file: File) {
    const previewUrl = URL.createObjectURL(file);
    setState({ kind: "submitting", previewUrl });

    try {
      const [scan] = await Promise.all([retakeScan(scanId, file), wait(MIN_SCANNING_MS)]);
      URL.revokeObjectURL(previewUrl);
      onSuccess(scan);
    } catch (error) {
      URL.revokeObjectURL(previewUrl);
      const message =
        error instanceof ScanError ? error.message : "Something went wrong. Please try again.";
      setState({ kind: "error", message });
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

  return (
    <div className="space-y-3">
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

      {state.kind === "idle" && (
        <div className={`space-y-3 ${CARD_CLASS}`}>
          <div className="space-y-1">
            <p className="text-sm font-medium">Want a different result?</p>
            <p className="text-sm text-black/60 dark:text-white/60">
              Retake your photo — one free retake included, no matter the reason.
            </p>
          </div>
          <div className="space-y-2">
            <button type="button" onClick={triggerSelfieCapture} className={PRIMARY_BUTTON_CLASS}>
              Retake with camera
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

      {state.kind === "camera" && (
        <CameraCapture onCapture={processFile} onCancel={() => setState({ kind: "idle" })} />
      )}

      {state.kind === "submitting" && (
        <ScanningOverlay previewUrl={state.previewUrl} leadMessages={EARLY_SCAN_MESSAGES} />
      )}

      {state.kind === "error" && (
        <div className="animate-fade-in space-y-3">
          <div
            role="alert"
            className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900/40 dark:bg-red-950/30 dark:text-red-300"
          >
            {state.message}
          </div>
          <button
            type="button"
            onClick={() => setState({ kind: "idle" })}
            className={SECONDARY_BUTTON_CLASS}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
