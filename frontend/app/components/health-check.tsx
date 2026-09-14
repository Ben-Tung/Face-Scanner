"use client";

import { useEffect, useState } from "react";

import { apiUrl } from "@/lib/api";

import { CARD_CLASS } from "./ui";

type Health = {
  status: string;
  service: string;
  version: string;
  environment: string;
};

type State =
  | { kind: "checking" }
  | { kind: "connected"; health: Health }
  | { kind: "failed"; error: string };

// Runs in the browser on purpose: it reaches the API on the host-published
// port, so the same code works under docker compose and when running natively.
export function HealthCheck() {
  const [state, setState] = useState<State>({ kind: "checking" });

  useEffect(() => {
    const controller = new AbortController();

    fetch(apiUrl("/api/health"), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`API responded ${response.status}`);
        }
        return (await response.json()) as Health;
      })
      .then((health) => setState({ kind: "connected", health }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setState({
          kind: "failed",
          error: error instanceof Error ? error.message : "Unknown error",
        });
      });

    return () => controller.abort();
  }, []);

  return (
    <div className={`${CARD_CLASS} text-sm`}>
      {state.kind === "checking" && (
        <p className="text-black/60 dark:text-white/60">Checking API…</p>
      )}

      {state.kind === "connected" && (
        <p>
          <span className="font-medium text-green-700 dark:text-green-400">
            Connected
          </span>
          <span className="text-black/60 dark:text-white/60">
            {" · "}
            {state.health.service} v{state.health.version} (
            {state.health.environment})
          </span>
        </p>
      )}

      {state.kind === "failed" && (
        <p>
          <span className="font-medium text-red-700 dark:text-red-400">
            API unreachable
          </span>
          <span className="text-black/60 dark:text-white/60">
            {" · "}
            {state.error}
          </span>
        </p>
      )}
    </div>
  );
}
