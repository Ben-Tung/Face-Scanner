import { HealthCheck } from "@/app/components/health-check";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center gap-6 px-5 py-12">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Palette</h1>
        <p className="text-black/60 dark:text-white/60">
          Find the colors that suit you.
        </p>
      </div>

      <HealthCheck />
    </main>
  );
}
