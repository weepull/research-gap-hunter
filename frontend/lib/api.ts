/**
 * Typed API client for the Research Gap Hunter FastAPI backend.
 * Response shapes mirror the Pydantic models in api/main.py exactly.
 */

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

export interface HealthResponse {
  status: string;
  papers: number;
  limitations: number;
  future_directions: number;
}

export interface ExplainResponse {
  explanation: string;
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
    const body = await res.text().catch(() => "");
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
