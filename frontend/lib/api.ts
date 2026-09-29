/**
 * Typed API client for the Research Gap Hunter FastAPI backend.
 * Response shapes mirror the Pydantic models in api/main.py exactly.
 */

// NEXT_PUBLIC_* values are inlined at BUILD time, not read at runtime. If the
// build environment has no NEXT_PUBLIC_API_URL, the deployed bundle ships
// "http://localhost:8000" baked in and every request fails in the browser with
// no server-side error to notice. Fail the production build instead, so the
// misconfiguration surfaces during deploy rather than in front of a user.
const CONFIGURED_API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!CONFIGURED_API_URL && process.env.NODE_ENV === "production") {
  throw new Error(
    "NEXT_PUBLIC_API_URL is not set. Set it in the build environment " +
      "(e.g. Vercel project settings) — it is inlined at build time and " +
      "cannot be supplied at runtime.",
  );
}

const API_BASE = CONFIGURED_API_URL ?? "http://localhost:8000";

/**
 * Thrown when the API refuses a feature that is switched off for this
 * deployment (the public demo disables ingestion and live explanations).
 *
 * Distinct from a generic error because it is an expected state, not a failure:
 * callers should render the server's explanation as information rather than as
 * something that went wrong.
 */
export class FeatureDisabledError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FeatureDisabledError";
  }
}

/** Thrown when the API rejects a request because of rate limiting. */
export class RateLimitError extends Error {
  readonly retryAfterSeconds: number | null;

  constructor(message: string, retryAfterSeconds: number | null) {
    super(message);
    this.name = "RateLimitError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

// ---------------------------------------------------------------------------
// Types — mirror api/main.py response models
// ---------------------------------------------------------------------------

export interface GapResult {
  gap_description: string;
  score: number;
  frequency_score: number;
  recency_score: number;
  solution_deficit_score: number;
  supporting_papers: string[];
  proposed_solutions: string[];
  /**
   * "corroborated" (>= 2 papers) or "single_source". Every corroborated gap ranks
   * above every single-source one, so **scores are non-monotonic across the tier
   * boundary** — a single-source gap lower down the list can carry a higher score.
   * Showing the tier is what stops that reading as a bug.
   *
   * Sent by the backend rather than derived from supporting_papers.length here, so
   * the ranking and the label can never disagree.
   */
  tier: "corroborated" | "single_source";
}

export interface LimitationResult {
  limitation_text: string;
  score: number;
  paper_ids: string[];
  domain: string;
}

export interface CrossDomainMatch {
  source_gap: string;
  target_solution: string;
  similarity_score: number;
  source_papers: string[];
  target_papers: string[];
  source_domain: string;
  target_domain: string;
}

export interface CorpusInfo {
  domain: string;
  /** Every paper in the domain — the corpus SIZE. */
  papers: number | null;
  /**
   * Papers with at least one extracted limitation. This is the value that
   * divides `frequency_score`, which is NOT the same as the corpus size: a paper
   * that extracted no limitations cannot corroborate any gap. Showing only
   * `papers` described a number the scores were not computed against.
   */
  papers_reporting_limitations: number | null;
  /** null when Qdrant is unreachable — distinct from a genuine 0. */
  limitations: number | null;
  future_directions: number | null;
  last_updated: string | null;
  graph_available: boolean;
  vectors_available: boolean;
}

export interface HealthResponse {
  /** "ok" or "degraded" — derived from `services`, never asserted. */
  status: string;
  /** null when the backing store is unreachable, as opposed to genuinely empty. */
  papers: number | null;
  limitations: number | null;
  future_directions: number | null;
  /** Per-dependency state: "ok" | "absent" | "unreachable". */
  services: Record<string, string>;
}

export interface ExplainResponse {
  explanation: string;
  /**
   * Recomputed from the stored vectors by the server, never a placeholder — the
   * endpoint used to hardcode 0.0 and never show the model any score.
   */
  similarity_score: number;
  /** The measured noise floor this pairing had to clear to be explained at all. */
  threshold: number;
  /** Why this was groundable. Currently only "corpus_match". */
  grounding: string;
  /**
   * Always true. An LLM's account of why two papers might connect is a suggestion
   * to evaluate, not a finding about the literature, however fluent it reads.
   * Render this; do not branch on it.
   */
  is_hypothesis: boolean;
  source_papers: string[];
  target_papers: string[];
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(path, API_BASE);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      url.searchParams.set(key, String(value));
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    if (res.status === 429) {
      // The API paces expensive endpoints (/ingest 3/min, /explain 10/min,
      // /gaps and /cross-domain 20/min). Surface this as its own error type so
      // callers can show a wait message instead of a raw "API 429: {...}".
      const header = res.headers.get("Retry-After");
      const parsed = header ? Number.parseInt(header, 10) : Number.NaN;
      const retryAfter = Number.isFinite(parsed) ? parsed : null;
      throw new RateLimitError(
        retryAfter
          ? `Rate limit reached. Try again in ${retryAfter}s.`
          : "Rate limit reached. Try again shortly.",
        retryAfter,
      );
    }
    const body = await res.text().catch(() => "");
    if (res.status === 403) {
      // The API sends a human-readable reason in `detail`; surface that rather
      // than a raw "API 403: {"detail":...}" blob.
      let detail = "";
      try {
        detail = (JSON.parse(body) as { detail?: string }).detail ?? "";
      } catch {
        detail = "";
      }
      throw new FeatureDisabledError(
        detail || "This feature is disabled in the public demo.",
      );
    }
    throw new Error(`API ${res.status}: ${body.slice(0, 200) || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Endpoint functions
// ---------------------------------------------------------------------------

export function fetchHealth(): Promise<HealthResponse> {
  return get<HealthResponse>("/health");
}

export function fetchGaps(domain: string, topN: number): Promise<GapResult[]> {
  return get<GapResult[]>("/gaps", { domain, top_n: topN });
}

export function fetchCorpus(domain: string): Promise<CorpusInfo> {
  return get<CorpusInfo>("/corpus", { domain });
}

export function searchLimitations(
  q: string,
  topK: number = 10,
  domain: string = "computer_vision",
): Promise<LimitationResult[]> {
  return get<LimitationResult[]>("/search", { q, top_k: topK, domain });
}

export function fetchCrossDomainMatches(
  source: string,
  target: string,
  topN: number = 10,
): Promise<CrossDomainMatch[]> {
  return get<CrossDomainMatch[]>("/cross-domain", {
    source,
    target,
    top_n: topN,
  });
}

export function fetchExplanation(
  sourceGap: string,
  targetSolution: string,
  source: string = "computer_vision",
  target: string = "medical_imaging",
): Promise<ExplainResponse> {
  return get<ExplainResponse>("/explain", {
    source_gap: sourceGap,
    target_solution: targetSolution,
    source,
    target,
  });
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

export const DOMAINS = [
  { value: "computer_vision", label: "Computer Vision" },
  { value: "medical_imaging", label: "Medical Imaging" },
] as const;

export function domainLabel(value: string): string {
  return DOMAINS.find((d) => d.value === value)?.label ?? value;
}
