"use client";

import { useEffect, useState } from "react";
import Container from "@/components/ui/Container";
import Eyebrow from "@/components/ui/Eyebrow";
import Skeleton from "@/components/ui/Skeleton";
import Chip from "@/components/ui/Chip";
import EmptyState from "@/components/ui/EmptyState";
import ErrorState from "@/components/ui/ErrorState";
import CorpusBanner from "@/components/CorpusBanner";
import DomainPicker from "@/components/DomainPicker";
import SearchField from "@/components/SearchField";
import { LimitationResult, searchLimitations } from "@/lib/api";

/** Paces the backend. Not input latency — the field responds instantly. */
const DEBOUNCE_MS = 500;

function ResultCard({ result }: { result: LimitationResult }) {
  return (
    <article className="rounded-lg border border-hairline bg-surface card-p shadow-[var(--shadow-2)]">
      <div className="flex items-start justify-between gap-4">
        <p className="text-body text-label">{result.limitation_text}</p>
        <span
          className="tabular shrink-0 rounded-full border border-hairline bg-surface-inset px-2.5 py-0.5 text-caption text-label-2"
          title="Cosine similarity between your query and this limitation statement."
        >
          {result.score.toFixed(3)}
          <span className="sr-only"> similarity to your query</span>
        </span>
      </div>
      {result.paper_ids.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {result.paper_ids.map((id) => (
            <Chip key={id} id={id} />
          ))}
        </div>
      )}
    </article>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("computer_vision");
  const [debounced, setDebounced] = useState("");

  // Debounce lives in its own effect and touches no other state, so the
  // request effect below stays a pure subscription.
  useEffect(() => {
    const trimmed = query.trim();
    const timer = setTimeout(() => setDebounced(trimmed), DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [query]);

  const requestKey = `${debounced}:${domain}`;
  const [result, setResult] = useState<{
    key: string;
    results: LimitationResult[] | null;
    error: string | null;
  } | null>(null);

  const hasQuery = query.trim().length > 0;
  // Busy the moment there is a query whose results are not in hand — which
  // includes the debounce window, so typing never looks like nothing happened.
  const busy = hasQuery && result?.key !== requestKey;
  const results = result?.key === requestKey ? result.results : null;
  const error = result?.key === requestKey ? result.error : null;

  useEffect(() => {
    if (!debounced) return;
    let cancelled = false;
    searchLimitations(debounced, 10, domain)
      .then((data) => {
        if (!cancelled) setResult({ key: requestKey, results: data, error: null });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setResult({
          key: requestKey,
          results: null,
          error: err instanceof Error ? err.message : "Search failed",
        });
      });
    return () => {
      cancelled = true;
    };
  }, [debounced, domain, requestKey]);

  return (
    <Container width="wide" className="section-y">
      <header>
        <Eyebrow>Retrieval</Eyebrow>
        <h1 className="mt-2 text-title-1 text-label">Semantic search</h1>
        <p className="mt-2 max-w-[62ch] text-body text-label-2">
          Vector search across every limitation statement extracted from the
          corpus, matched by meaning rather than keyword.
        </p>
      </header>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <SearchField value={query} onChange={setQuery} busy={busy} />
        <DomainPicker value={domain} onChange={setDomain} />
      </div>

      <div className="mt-5">
        <CorpusBanner domain={domain} />
      </div>

      <div className="mt-5">
        {!hasQuery && (
          <EmptyState eyebrow="Ready" title="Search by meaning, not keyword">
            Describe a problem in your own words — &ldquo;struggles with small
            objects&rdquo;, &ldquo;needs too much labelled data&rdquo;. Results
            are limitation statements whose meaning is closest to what you
            typed, not papers containing those words.
          </EmptyState>
        )}

        {hasQuery && busy && (
          <div className="space-y-3" aria-busy="true" aria-label="Searching">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-24 w-full" rounded="rounded-lg" />
            ))}
          </div>
        )}

        {hasQuery && !busy && error && (
          <ErrorState title="Search failed" message={error} />
        )}

        {hasQuery && !busy && !error && results && results.length === 0 && (
          <EmptyState eyebrow="No matches" title="Nothing close enough in this domain">
            No limitation statement in this corpus is semantically near that
            query. Try different wording, or switch domain.
          </EmptyState>
        )}

        {hasQuery && !busy && !error && results && results.length > 0 && (
          <>
            <p aria-live="polite" className="mb-3 text-caption text-label-3">
              {results.length} {results.length === 1 ? "result" : "results"}, closest first
            </p>
            <div className="space-y-3">
              {results.map((r, i) => (
                <ResultCard key={`${r.limitation_text}-${i}`} result={r} />
              ))}
            </div>
          </>
        )}
      </div>
    </Container>
  );
}
