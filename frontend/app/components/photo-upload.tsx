"use client";

import { useRef, useState } from "react";

const LIGHTING_TIPS = [
  "Face a window in daylight — indirect light, not direct sun on your face",
  "Turn off camera flash and skip overhead room lights",
  "Take off glasses, hats, and heavy foundation or bronzer",
  "Pull hair back off your forehead and cheeks",
  "Hold the phone at eye level, about an arm's length away",
] as const;

type State =
  | { kind: "empty" }
  | { kind: "selected"; file: File; previewUrl: string };

export function PhotoUpload() {
  const [state, setState] = useState<State>({ kind: "empty" });
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (state.kind === "selected") {
      URL.revokeObjectURL(state.previewUrl);
    }
    setState({ kind: "selected", file, previewUrl: URL.createObjectURL(file) });
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-black/10 p-4 dark:border-white/15">
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

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="user"
        onChange={handleFileChange}
        className="hidden"
      />

      {state.kind === "empty" && (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="w-full rounded-xl bg-black px-4 py-3 text-sm font-medium text-white dark:bg-white dark:text-black"
        >
          Take or choose a selfie
        </button>
      )}

      {state.kind === "selected" && (
        <div className="space-y-3">
          {/* eslint-disable-next-line @next/next/no-img-element -- transient local object URL, not worth Image's remote-optimization pipeline */}
          <img
            src={state.previewUrl}
            alt="Selected selfie preview"
            className="aspect-square w-full rounded-xl object-cover"
          />
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="w-full rounded-xl border border-black/10 px-4 py-3 text-sm font-medium dark:border-white/15"
          >
            Choose a different photo
          </button>
        </div>
      )}
    </div>
  );
}
