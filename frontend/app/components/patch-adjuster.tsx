"use client";

import { useRef, useState } from "react";

import type { PatchAnchors, PatchPoint, ScanImageInfo } from "@/lib/api";

import { PRIMARY_BUTTON_CLASS } from "./ui";

type PatchKey = "forehead" | "leftCheek" | "rightCheek";

const PATCH_LABELS: Record<PatchKey, string> = {
  forehead: "Forehead",
  leftCheek: "Left cheek",
  rightCheek: "Right cheek",
};

// Meets the standard mobile minimum touch-target size, independent of how
// small the actual (true-to-scale) sampled patch is on screen.
const HIT_AREA_PX = 44;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

type PatchAdjusterProps = {
  previewUrl: string;
  image: ScanImageInfo;
  initialPatches: PatchAnchors;
  message: string;
  onSubmit: (patches: PatchAnchors) => void;
  onRetake: () => void;
};

// Seeding state from `initialPatches` only on mount (not re-synced on prop
// changes) is safe here because the caller always interposes a "rescanning"
// render — which unmounts this component — between one review screen and
// the next, so a new `initialPatches` only ever arrives via a fresh mount.
export function PatchAdjuster({
  previewUrl,
  image,
  initialPatches,
  message,
  onSubmit,
  onRetake,
}: PatchAdjusterProps) {
  const [patches, setPatches] = useState<PatchAnchors>(initialPatches);
  const imgRef = useRef<HTMLImageElement>(null);

  function movePatch(key: PatchKey, clientX: number, clientY: number) {
    const rect = imgRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return;

    const half = patches.patchHalfSize;
    const x = clamp(((clientX - rect.left) / rect.width) * image.width, half, image.width - half);
    const y = clamp(((clientY - rect.top) / rect.height) * image.height, half, image.height - half);
    setPatches((prev) => ({ ...prev, [key]: { x, y } }));
  }

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>, key: PatchKey) {
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    movePatch(key, event.clientX, event.clientY);
  }

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>, key: PatchKey) {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    movePatch(key, event.clientX, event.clientY);
  }

  function handlePointerEnd(event: React.PointerEvent<HTMLDivElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const patchEntries: [PatchKey, PatchPoint][] = [
    ["forehead", patches.forehead],
    ["leftCheek", patches.leftCheek],
    ["rightCheek", patches.rightCheek],
  ];

  return (
    <div className="animate-fade-in space-y-4">
      <p className="text-sm text-black/60 dark:text-white/60">
        Drag each box onto an evenly lit patch of skin, then rescan.
      </p>

      <div className="relative w-full overflow-hidden rounded-xl border border-black/10 dark:border-white/15">
        {/* eslint-disable-next-line @next/next/no-img-element -- transient local object URL, not worth Image's remote-optimization pipeline */}
        <img
          ref={imgRef}
          src={previewUrl}
          alt="Your photo"
          draggable={false}
          className="block h-auto w-full select-none"
        />

        {patchEntries.map(([key, point]) => {
          const leftPct = (point.x / image.width) * 100;
          const topPct = (point.y / image.height) * 100;
          const widthPct = ((patches.patchHalfSize * 2) / image.width) * 100;
          const heightPct = ((patches.patchHalfSize * 2) / image.height) * 100;

          return (
            <div key={key}>
              {/* Visual box: true-to-scale with the area actually sampled. */}
              <div
                aria-hidden
                className="pointer-events-none absolute rounded-sm border-2 border-white/90 shadow-[0_0_0_1px_rgba(0,0,0,0.5)]"
                style={{
                  left: `${leftPct}%`,
                  top: `${topPct}%`,
                  width: `${widthPct}%`,
                  height: `${heightPct}%`,
                  transform: "translate(-50%, -50%)",
                }}
              />
              {/* Hit area: fixed touch-friendly size, carries the drag handlers.
                  Pointer-drag only (no keyboard equivalent), so this is
                  labeled as a plain interactive control rather than a
                  role="slider" — that role implies keyboard-adjustable
                  numeric semantics this doesn't have. */}
              <div
                aria-label={`${PATCH_LABELS[key]} sample area, currently at ${Math.round(point.x)}, ${Math.round(point.y)}. Drag to reposition.`}
                tabIndex={0}
                className="absolute flex touch-none items-center justify-center rounded-full active:bg-black/10 dark:active:bg-white/10"
                style={{
                  left: `${leftPct}%`,
                  top: `${topPct}%`,
                  width: HIT_AREA_PX,
                  height: HIT_AREA_PX,
                  transform: "translate(-50%, -50%)",
                }}
                onPointerDown={(event) => handlePointerDown(event, key)}
                onPointerMove={(event) => handlePointerMove(event, key)}
                onPointerUp={handlePointerEnd}
                onPointerCancel={handlePointerEnd}
              >
                <span className="rounded-full bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-white">
                  {PATCH_LABELS[key]}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div
        role="alert"
        className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-200"
      >
        {message}
      </div>

      <div className="space-y-2">
        <button
          type="button"
          onClick={() => onSubmit(patches)}
          className={PRIMARY_BUTTON_CLASS}
        >
          Rescan
        </button>
        <button
          type="button"
          onClick={onRetake}
          className="w-full rounded-xl border border-black/10 px-4 py-3 text-sm font-medium dark:border-white/15"
        >
          Retake photo
        </button>
      </div>
    </div>
  );
}
