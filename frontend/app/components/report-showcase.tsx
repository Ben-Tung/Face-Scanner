const REPORT_CONTENTS = [
  "Your full color palette (~20–30 shades), sorted into best, good, and avoid",
  "Makeup shade guidance — your foundation undertone, plus lip and blush shades",
  "Your recommended jewelry metal tone",
  "A shopping guide for building a wardrobe around your colors",
  "A short, personalized write-up about your season",
];

export function ReportShowcase() {
  return (
    <section className="space-y-3 border-t border-black/10 pt-8 dark:border-white/15">
      <div className="space-y-1">
        <p className="text-xs font-medium uppercase tracking-wide text-black/40 dark:text-white/40">
          Preview
        </p>
        <h2 className="text-xl font-semibold tracking-tight">What&apos;s in your full report</h2>
      </div>
      <ul className="space-y-2 text-sm text-black/70 dark:text-white/70">
        {REPORT_CONTENTS.map((item) => (
          <li key={item} className="flex gap-2">
            <span aria-hidden className="text-black/30 dark:text-white/30">
              •
            </span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
