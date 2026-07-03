"use client";

import { useCallback, useEffect, useState } from "react";
import { DOMAINS, GapResult, fetchGaps } from "@/lib/api";

const TOP_N_OPTIONS = [5, 10, 20];

function scoreColor(score: number): string {
  if (score > 0.6) return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
  if (score >= 0.4) return "bg-amber-500/15 text-amber-400 border-amber-500/30";
  return "bg-red-500/15 text-red-400 border-red-500/30";
}

function SubScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 shrink-0 text-xs text-muted">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-card-border/50">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="w-10 shrink-0 text-right text-xs tabular-nums text-muted">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function GapCard({ gap, rank }: { gap: GapResult; rank: number }) {
  return (
    <div className="rounded-xl border border-card-border bg-card p-5 transition-colors hover:border-accent/40">
      <div className="mb-3 flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent/15 text-sm font-bold text-accent">
            {rank}
          </span>
          <p className="text-lg font-medium leading-snug text-foreground">
            {gap.gap_description}
          </p>
        </div>
        <span
          className={`shrink-0 rounded-full border px-3 py-1 text-sm font-semibold tabular-nums ${scoreColor(gap.score)}`}
        >
          {gap.score.toFixed(3)}
        </span>
      </div>

      <div className="mb-4 flex gap-4 text-xs text-muted">
        <span>
          <span className="font-semibold text-foreground">{gap.supporting_papers.length}</span>{" "}
          supporting {gap.supporting_papers.length === 1 ? "paper" : "papers"}
        </span>
        <span>
          <span className="font-semibold text-foreground">{gap.proposed_solutions.length}</span>{" "}
          proposed {gap.proposed_solutions.length === 1 ? "solution" : "solutions"}
        </span>
      </div>

      <div className="mb-1 space-y-1.5">
        <SubScoreBar label="Frequency" value={gap.frequency_score} color="bg-accent" />
        <SubScoreBar label="Recency" value={gap.recency_score} color="bg-sky-500" />
        <SubScoreBar label="Solution deficit" value={gap.solution_deficit_score} color="bg-rose-500" />
      </div>

      {gap.proposed_solutions.length > 0 && (
        <div className="mt-4 border-t border-card-border pt-3">
          <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
            Proposed solutions
          </p>
          <ul className="space-y-1">
            {gap.proposed_solutions.map((sol, i) => (
              <li key={i} className="flex gap-2 text-sm text-foreground/80">
                <span className="text-accent">→</span>
                {sol}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      {[0, 1, 2].map((i) => (
        <div key={i} className="animate-pulse rounded-xl border border-card-border bg-card p-5">
          <div className="mb-4 flex justify-between">
            <div className="h-6 w-2/3 rounded bg-card-border/60" />
            <div className="h-6 w-16 rounded-full bg-card-border/60" />
          </div>
          <div className="space-y-2">
            <div className="h-2 w-full rounded bg-card-border/40" />
            <div className="h-2 w-5/6 rounded bg-card-border/40" />
            <div className="h-2 w-4/6 rounded bg-card-border/40" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function GapExplorerPage() {
  const [domain, setDomain] = useState<string>("computer_vision");
  const [topN, setTopN] = useState<number>(10);
  const [gaps, setGaps] = useState<GapResult[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setGaps(await fetchGaps(domain, topN));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch gaps");
      setGaps(null);
    } finally {
      setLoading(false);
    }
  }, [domain, topN]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          Research Gap Hunter
        </h1>
        <p className="mt-1 text-muted">AI-powered scientific discovery</p>
      </header>

      <div className="mb-6 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-muted">
          Domain
          <select
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="rounded-md border border-card-border bg-card px-3 py-1.5 text-sm text-foreground focus:border-accent focus:outline-none"
          >
            {DOMAINS.map((d) => (
              <option key={d.value} value={d.value}>
                {d.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-3 text-sm text-muted">
          Top gaps
          <input
            type="range"
            min={0}
            max={TOP_N_OPTIONS.length - 1}
            step={1}
            value={TOP_N_OPTIONS.indexOf(topN)}
            onChange={(e) => setTopN(TOP_N_OPTIONS[Number(e.target.value)])}
            className="w-32 accent-[#6366f1]"
          />
          <span className="w-6 font-semibold tabular-nums text-foreground">{topN}</span>
        </label>
      </div>

      {loading && <LoadingSkeleton />}

      {!loading && error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-6 text-center">
          <p className="font-medium text-red-400">Failed to load gaps</p>
          <p className="mt-1 text-sm text-muted">{error}</p>
          <button
            onClick={load}
            className="mt-4 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent/80"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && gaps && gaps.length === 0 && (
        <div className="rounded-xl border border-card-border bg-card p-10 text-center">
          <p className="text-lg font-medium text-foreground">No gaps found</p>
          <p className="mt-1 text-sm text-muted">
            No scored research gaps exist for this domain yet. Try ingesting more papers.
          </p>
        </div>
      )}

      {!loading && !error && gaps && gaps.length > 0 && (
        <div className="space-y-4">
          {gaps.map((gap, i) => (
            <GapCard key={gap.gap_description} gap={gap} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
