const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Build an absolute URL for a backend API path, e.g. apiUrl("/api/health"). */
export function apiUrl(path: string): string {
  return `${API_BASE_URL.replace(/\/$/, "")}${path.startsWith("/") ? path : `/${path}`}`;
}
