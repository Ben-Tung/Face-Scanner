import type { FullReport, ScanSwatch } from "@/lib/api";
import { ColorChip } from "./color-chip";

const CARD_CLASS = "space-y-3 rounded-xl border border-black/10 p-4 dark:border-white/15";
const SUBLABEL_CLASS = "text-xs font-medium text-black/50 dark:text-white/50";

function ChipRow({ swatches, size = "md", muted = false }: { swatches: ScanSwatch[]; size?: "sm" | "md" | "lg"; muted?: boolean }) {
  return (
    <div className="flex flex-wrap gap-3">
      {swatches.map((swatch) => (
        <ColorChip key={swatch.hex} swatch={swatch} size={size} muted={muted} />
      ))}
    </div>
  );
}

export function FullReportView({ fullReport }: { fullReport: FullReport }) {
  const { palette, makeup, jewelry, shoppingGuidance, paragraph } = fullReport;

  return (
    <div className="space-y-5">
      {paragraph && (
        <p className="text-sm leading-relaxed text-black/70 dark:text-white/70">{paragraph}</p>
      )}

      <section className={CARD_CLASS}>
        <p className="text-sm font-medium">Your Full Palette</p>
        <div className="space-y-2">
          <p className={SUBLABEL_CLASS}>Best on you</p>
          <ChipRow swatches={palette.best} size="md" />
        </div>
        <div className="space-y-2">
          <p className={SUBLABEL_CLASS}>Also good</p>
          <ChipRow swatches={palette.good} size="sm" />
        </div>
        <div className="space-y-2">
          <p className="text-xs font-medium text-red-700/80 dark:text-red-300/70">Steer clear of</p>
          <ChipRow swatches={palette.avoid} size="sm" muted />
        </div>
      </section>

      <section className={CARD_CLASS}>
        <p className="text-sm font-medium">Makeup &amp; Foundation</p>
        <p className="text-sm text-black/70 dark:text-white/70">
          {makeup.foundationUndertone} — {makeup.foundationTip}
        </p>
        <div className="space-y-2">
          <p className={SUBLABEL_CLASS}>Lip shades</p>
          <ChipRow swatches={makeup.lipShades} />
        </div>
        <div className="space-y-2">
          <p className={SUBLABEL_CLASS}>Blush shades</p>
          <ChipRow swatches={makeup.blushShades} />
        </div>
      </section>

      <section className={CARD_CLASS}>
        <p className="text-sm font-medium">Jewelry</p>
        <div className="flex justify-center gap-6">
          <ColorChip
            swatch={jewelry.goldSwatch}
            size="lg"
            ring={jewelry.metal !== "Silver"}
            muted={jewelry.metal === "Silver"}
          />
          <ColorChip
            swatch={jewelry.silverSwatch}
            size="lg"
            ring={jewelry.metal !== "Gold"}
            muted={jewelry.metal === "Gold"}
          />
        </div>
        <p className="text-sm text-black/70 dark:text-white/70">{jewelry.tip}</p>
      </section>

      <section className={CARD_CLASS}>
        <p className="text-sm font-medium">Shopping Guide</p>
        <p className="text-sm leading-relaxed text-black/70 dark:text-white/70">{shoppingGuidance}</p>
      </section>
    </div>
  );
}
