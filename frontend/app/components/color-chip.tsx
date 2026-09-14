import type { ScanSwatch } from "@/lib/api";

const SIZE_CLASSES: Record<"sm" | "md" | "lg", string> = {
  sm: "w-12",
  md: "w-16",
  lg: "w-20",
};

export function ColorChip({
  swatch,
  size = "md",
  muted = false,
  ring = false,
}: {
  swatch: ScanSwatch;
  size?: "sm" | "md" | "lg";
  muted?: boolean;
  ring?: boolean;
}) {
  return (
    <div className={`${SIZE_CLASSES[size]} space-y-1.5 text-center ${muted ? "opacity-50" : ""}`}>
      <div
        className={`aspect-square w-full rounded-full border shadow-sm ${
          ring
            ? "border-black/20 ring-2 ring-black dark:border-white/25 dark:ring-white"
            : "border-black/10 dark:border-white/15"
        }`}
        style={{ backgroundColor: swatch.hex }}
        aria-hidden
      />
      <p
        className={`leading-tight text-black/60 dark:text-white/60 ${
          size === "sm" ? "text-[10px]" : "text-[11px]"
        }`}
      >
        {swatch.name}
      </p>
    </div>
  );
}
