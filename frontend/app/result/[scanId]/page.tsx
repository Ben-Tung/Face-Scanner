import { Suspense } from "react";

import { ScanResultView } from "@/app/components/scan-result-view";

export default function ResultPage() {
  return (
    <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center gap-6 px-5 py-12">
      <Suspense
        fallback={
          <div className="flex justify-center py-12">
            <span
              aria-hidden
              className="h-8 w-8 animate-spin rounded-full border-2 border-black/20 border-t-black dark:border-white/20 dark:border-t-white"
            />
          </div>
        }
      >
        <ScanResultView />
      </Suspense>
    </main>
  );
}
