"use client";

import { useEffect, useRef, useState } from "react";
import { DOMAINS, LimitationResult, searchLimitations } from "@/lib/api";

const DEBOUNCE_MS = 500;

function ResultCard({ result }: { result: LimitationResult }) {
  return (
    <div className="rounded-xl border border-card-border bg-card p-4 transition-colors hover:border-accent/40">
      <div className="flex items-start justify-between gap-4">
        <p className="text-base leading-snug text-foreground">{result.limitation_text}</p>
        <span className="shrink-0 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-0.5 text-sm font-semibold tabular-nums text-accent">
          {result.score.toFixed(3)}
        </span>
      </div>
      {result.paper_ids.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {result.paper_ids.map((pid) => (
            <a
              key={pid}
              href={`https://arxiv.org/abs/${pid}`}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded bg-card-border/40 px-2 py-0.5 font-mono text-xs text-muted hover:bg-accent/15 hover:text-accent"
            >
              {pid}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("computer_vision");
  const [results, setResults] = useState<LimitationResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced search — fires 500ms after the user stops typing.
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    const trimmed = query.trim();
    if (!trimmed) {
      setResults(null);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    debounceRef.current = setTimeout(async () => {
      try {
        setError(null);
        setResults(await searchLimitations(trimmed, 10, domain));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Search failed");
        setResults(null);
      } finally {
        setLoading(false);
      }
    }, DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, domain]);

  return (
    <div>
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Semantic Search</h1>
        <p className="mt-1 text-muted">
          Vector search over every extracted limitation statement
        </p>
      </header>

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Find papers about..."
          autoFocus
          className="min-w-64 flex-1 rounded-lg border border-card-border bg-card px-4 py-2.5 text-foreground placeholder:text-muted/60 focus:border-accent focus:outline-none"
        />
        <select
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
          className="rounded-lg border border-card-border bg-card px-3 py-2.5 text-sm text-foreground focus:border-accent focus:outline-none"
        >
          {DOMAINS.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="space-y-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-20 animate-pulse rounded-xl border border-card-border bg-card" />
          ))}
        </div>
      )}

      {!loading && error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-6 text-center">
          <p className="font-medium text-red-400">Search failed</p>
          <p className="mt-1 text-sm text-muted">{error}</p>
        </div>
      )}

      {!loading && !error && results && results.length === 0 && (
        <div className="rounded-xl border border-card-border bg-card p-10 text-center">
          <p className="text-lg font-medium text-foreground">No results found</p>
          <p className="mt-1 text-sm text-muted">Try a different query or domain.</p>
        </div>
      )}

      {!loading && !error && results && results.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs uppercase tracking-wide text-muted">
            {results.length} {results.length === 1 ? "result" : "results"}
          </p>
          {results.map((r, i) => (
            <ResultCard key={`${r.limitation_text}-${i}`} result={r} />
          ))}
        </div>
      )}

      {!loading && !error && results === null && (
        <div className="rounded-xl border border-dashed border-card-border p-10 text-center text-muted">
          Start typing to search limitations across the corpus.
        </div>
      )}
    </div>
  );
}
