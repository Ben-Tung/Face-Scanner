import { Suspense } from "react";

import { ScanResultView } from "@/app/components/scan-result-view";
import { SPINNER_CLASS } from "@/app/components/ui";

export default function ResultPage() {
  return (
    <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center gap-6 px-5 py-12">
      <Suspense
        fallback={
          <div className="flex justify-center py-12">
            <span aria-hidden className={SPINNER_CLASS} />
          </div>
        }
      >
        <ScanResultView />
      </Suspense>
    </main>
  );
}
