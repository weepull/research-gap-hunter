"use client";

import { useState } from "react";
import {
  CrossDomainMatch,
  DOMAINS,
  domainLabel,
  fetchCrossDomainMatches,
  fetchExplanation,
} from "@/lib/api";

function PaperChips({ ids, tone }: { ids: string[]; tone: "blue" | "green" }) {
  const cls =
    tone === "blue"
      ? "hover:bg-sky-500/15 hover:text-sky-400"
      : "hover:bg-emerald-500/15 hover:text-emerald-400";
  return (
    <div className="flex flex-wrap gap-1.5">
      {ids.map((pid) => (
        <a
          key={pid}
          href={`https://arxiv.org/abs/${pid}`}
          target="_blank"
          rel="noopener noreferrer"
          className={`rounded bg-card-border/40 px-2 py-0.5 font-mono text-xs text-muted ${cls}`}
        >
          {pid}
        </a>
      ))}
    </div>
  );
}

function ConnectionCard({ match }: { match: CrossDomainMatch }) {
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);

  async function explain() {
    setExplaining(true);
    setExplainError(null);
    try {
      const res = await fetchExplanation(
        match.source_gap,
        match.target_solution,
        match.source_domain,
        match.target_domain,
      );
      setExplanation(res.explanation);
    } catch (err) {
      setExplainError(err instanceof Error ? err.message : "Explanation failed");
    } finally {
      setExplaining(false);
    }
  }

  return (
    <div className="overflow-hidden rounded-xl border border-card-border bg-card transition-colors hover:border-accent/40">
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr]">
        {/* Source gap — blue */}
        <div className="border-b border-sky-500/20 bg-sky-500/5 p-5 md:border-b-0 md:border-r">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-sky-400">
            {domainLabel(match.source_domain)} · unresolved gap
          </p>
          <p className="text-sm leading-relaxed text-foreground">{match.source_gap}</p>
          <div className="mt-3">
            <PaperChips ids={match.source_papers} tone="blue" />
          </div>
        </div>

        {/* Bridge */}
        <div className="flex flex-row items-center justify-center gap-2 px-4 py-3 md:flex-col md:py-5">
          <div className="hidden h-px w-8 bg-gradient-to-r from-sky-500/50 to-accent md:block" />
          <div className="flex flex-col items-center gap-1">
            <span className="rounded-full border border-accent/40 bg-accent/15 px-3 py-1 text-sm font-bold tabular-nums text-accent">
              {match.similarity_score.toFixed(3)}
            </span>
            <span className="text-lg text-accent">⇄</span>
          </div>
          <div className="hidden h-px w-8 bg-gradient-to-r from-accent to-emerald-500/50 md:block" />
        </div>

        {/* Target solution — green */}
        <div className="border-t border-emerald-500/20 bg-emerald-500/5 p-5 md:border-t-0 md:border-l">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-400">
            {domainLabel(match.target_domain)} · proposed solution
          </p>
          <p className="text-sm leading-relaxed text-foreground">{match.target_solution}</p>
          <div className="mt-3">
            <PaperChips ids={match.target_papers} tone="green" />
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="border-t border-card-border px-5 py-3">
        {explanation ? (
          <div>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-accent">
              Why this connection matters
            </p>
            <p className="text-sm leading-relaxed text-foreground/85">{explanation}</p>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <button
              onClick={explain}
              disabled={explaining}
              className="rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-sm font-medium text-accent transition-colors hover:bg-accent/20 disabled:cursor-wait disabled:opacity-60"
            >
              {explaining ? "Generating… (LLM, ~10s)" : "Generate Explanation"}
            </button>
            {explainError && <span className="text-sm text-red-400">{explainError}</span>}
          </div>
        )}
      </div>
    </div>
  );
}

export default function CrossDomainPage() {
  const [source, setSource] = useState("computer_vision");
  const [target, setTarget] = useState("medical_imaging");
  const [matches, setMatches] = useState<CrossDomainMatch[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function discover() {
    setLoading(true);
    setError(null);
    try {
      setMatches(await fetchCrossDomainMatches(source, target, 10));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Discovery failed");
      setMatches(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <header className="mb-8">
        <h1 className="bg-gradient-to-r from-sky-400 via-indigo-400 to-emerald-400 bg-clip-text text-3xl font-bold tracking-tight text-transparent">
          Cross-Domain Discovery
        </h1>
        <p className="mt-1 text-muted">
          Solutions proposed in one field, matched to open problems in another
        </p>
      </header>

      <div className="mb-8 flex flex-wrap items-end gap-4 rounded-xl border border-card-border bg-card p-5">
        <label className="flex flex-col gap-1.5 text-sm text-muted">
          Source domain (open problems)
          <select
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="rounded-md border border-card-border bg-background px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none"
          >
            {DOMAINS.map((d) => (
              <option key={d.value} value={d.value}>
                {d.label}
              </option>
            ))}
          </select>
        </label>

        <span className="pb-2 text-xl text-accent">→</span>

        <label className="flex flex-col gap-1.5 text-sm text-muted">
          Target domain (solutions)
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="rounded-md border border-card-border bg-background px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none"
          >
            {DOMAINS.map((d) => (
              <option key={d.value} value={d.value}>
                {d.label}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={discover}
          disabled={loading || source === target}
          className="rounded-lg bg-accent px-5 py-2 font-medium text-white shadow-lg shadow-accent/20 transition-colors hover:bg-accent/80 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Discovering…" : "Discover Connections"}
        </button>
        {source === target && (
          <span className="pb-2 text-xs text-amber-400">
            Pick two different domains
          </span>
        )}
      </div>

      {loading && (
        <div className="space-y-4">
          {[0, 1].map((i) => (
            <div key={i} className="h-44 animate-pulse rounded-xl border border-card-border bg-card" />
          ))}
        </div>
      )}

      {!loading && error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-6 text-center">
          <p className="font-medium text-red-400">Discovery failed</p>
          <p className="mt-1 text-sm text-muted">{error}</p>
        </div>
      )}

      {!loading && !error && matches && matches.length === 0 && (
        <div className="rounded-xl border border-card-border bg-card p-10 text-center">
          <p className="text-lg font-medium text-foreground">No connections found</p>
          <p className="mt-1 text-sm text-muted">
            No cross-domain matches above the similarity threshold for this domain pair.
          </p>
        </div>
      )}

      {!loading && !error && matches && matches.length > 0 && (
        <div className="space-y-4">
          <p className="text-xs uppercase tracking-wide text-muted">
            {matches.length} {matches.length === 1 ? "connection" : "connections"} discovered
          </p>
          {matches.map((m, i) => (
            <ConnectionCard key={`${m.source_gap}-${m.target_solution}-${i}`} match={m} />
          ))}
        </div>
      )}

      {!loading && !error && matches === null && (
        <div className="rounded-xl border border-dashed border-card-border p-14 text-center">
          <p className="text-2xl">🔭</p>
          <p className="mt-2 text-lg font-medium text-foreground">
            Ready to discover research hypotheses
          </p>
          <p className="mt-1 text-sm text-muted">
            Pick a source and target domain, then hit Discover Connections.
          </p>
        </div>
      )}
    </div>
  );
}
