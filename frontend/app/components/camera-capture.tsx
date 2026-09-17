"use client";

import { useEffect, useRef, useState } from "react";

import { PRIMARY_BUTTON_CLASS, SECONDARY_BUTTON_CLASS, SPINNER_CLASS } from "@/app/components/ui";

const CAPTURE_MIME = "image/jpeg";
const CAPTURE_QUALITY = 0.92;

type CameraState = "starting" | "ready" | "error";

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") {
      return "Camera access was denied. Allow camera access in your browser, or choose a photo from your library instead.";
    }
    if (error.name === "NotFoundError" || error.name === "OverconstrainedError") {
      return "No camera was found on this device. Choose a photo from your library instead.";
    }
  }
  return "Couldn't access your camera. Choose a photo from your library instead.";
}

// In-page live capture for devices that can't launch a native camera app via
// a file input's capture attribute (desktop browsers, mainly). Callers are
// expected to only mount this where navigator.mediaDevices.getUserMedia
// exists — see prefersNativeCameraCapture in photo-upload.tsx.
export function CameraCapture({
  onCapture,
  onCancel,
}: {
  onCapture: (file: File) => void;
  onCancel: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CameraState>("starting");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: "user", width: { ideal: 1024 }, height: { ideal: 1024 } },
        audio: false,
      })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setState("ready");
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setErrorMessage(cameraErrorMessage(error));
        setState("error");
      });

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  function handleCapture() {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        onCapture(new File([blob], "selfie.jpg", { type: CAPTURE_MIME }));
      },
      CAPTURE_MIME,
      CAPTURE_QUALITY,
    );
  }

  return (
    <div className="animate-fade-in space-y-4">
      <div className="relative aspect-square w-full overflow-hidden rounded-xl bg-black/5 dark:bg-white/5">
        {state === "starting" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={SPINNER_CLASS} />
          </div>
        )}
        {state === "error" && (
          <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-black/60 dark:text-white/60">
            {errorMessage}
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className={`h-full w-full scale-x-[-1] object-cover ${state === "ready" ? "" : "invisible"}`}
        />
      </div>

      <div className="space-y-2">
        {state === "ready" && (
          <button type="button" onClick={handleCapture} className={PRIMARY_BUTTON_CLASS}>
            Capture
          </button>
        )}
        <button type="button" onClick={onCancel} className={SECONDARY_BUTTON_CLASS}>
          Cancel
        </button>
      </div>
    </div>
  );
}
