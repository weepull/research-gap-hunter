"use client";

import { useEffect, useMemo, useState } from "react";
import Container from "@/components/ui/Container";
import Eyebrow from "@/components/ui/Eyebrow";
import Skeleton from "@/components/ui/Skeleton";
import EmptyState from "@/components/ui/EmptyState";
import ErrorState from "@/components/ui/ErrorState";
import SegmentedControl from "@/components/ui/SegmentedControl";
import CorpusBanner from "@/components/CorpusBanner";
import DomainPicker from "@/components/DomainPicker";
import GapCard from "@/components/GapCard";
import GapDetailSheet from "@/components/GapDetailSheet";
import { GapResult, fetchGaps } from "@/lib/api";

const TOP_N_OPTIONS = [
  { value: 5, label: "5" },
  { value: 10, label: "10" },
  { value: 20, label: "20" },
] as const;

function LoadingList() {
  return (
    <div className="space-y-4" aria-busy="true" aria-label="Loading gaps">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="rounded-lg border border-hairline bg-surface card-p shadow-[var(--shadow-2)]"
        >
          <div className="flex items-start justify-between gap-4">
            <Skeleton className="h-6 w-2/3" />
            <Skeleton className="h-12 w-12" rounded="rounded-full" />
          </div>
          <div className="mt-5 space-y-3">
            <Skeleton className="h-1.5 w-full" rounded="rounded-full" />
            <Skeleton className="h-1.5 w-5/6" rounded="rounded-full" />
            <Skeleton className="h-1.5 w-4/6" rounded="rounded-full" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function GapsPage() {
  const [domain, setDomain] = useState("computer_vision");
  const [topN, setTopN] = useState<number>(10);
  const [selected, setSelected] = useState<GapResult | null>(null);
  // Bumped by Retry so a failed request can be re-issued without changing the
  // query itself.
  const [attempt, setAttempt] = useState(0);

  // The request this render is *asking* for. Loading is then a derived fact —
  // "what I have is not what I asked for" — rather than a flag some effect has
  // to remember to set. That removes the synchronous setState in the effect
  // body, and with it a whole class of cascading-render bugs.
  const requestKey = `${domain}:${topN}:${attempt}`;
  const [result, setResult] = useState<{
    key: string;
    gaps: GapResult[] | null;
    error: string | null;
  } | null>(null);

  const loading = result?.key !== requestKey;
  const gaps = result?.key === requestKey ? result.gaps : null;
  const error = result?.key === requestKey ? result.error : null;

  useEffect(() => {
    let cancelled = false;
    fetchGaps(domain, topN)
      .then((data) => {
        if (!cancelled) setResult({ key: requestKey, gaps: data, error: null });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setResult({
          key: requestKey,
          gaps: null,
          error: err instanceof Error ? err.message : "Failed to fetch gaps",
        });
      });
    return () => {
      cancelled = true;
    };
  }, [domain, topN, requestKey]);

  // Hoisted out of the render map: it was previously recomputed once per card,
  // making an O(n) reduction O(n²) for no reason.
  const maxFrequency = useMemo(
    () => (gaps && gaps.length > 0 ? Math.max(...gaps.map((g) => g.frequency_score)) : 0),
    [gaps],
  );

  return (
    <Container width="wide" className="section-y">
      <header>
        <Eyebrow>Discovery</Eyebrow>
        <h1 className="mt-2 text-title-1 text-label">Ranked research gaps</h1>
        <p className="mt-2 max-w-[62ch] text-body text-label-2">
          Open problems the literature keeps reporting, scored on how often they
          recur, how recent those reports are, and how little has been proposed
          to address them.
        </p>
      </header>

      <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2.5">
          <span className="text-caption text-label-3" id="domain-label">
            Domain
          </span>
          <DomainPicker value={domain} onChange={setDomain} />
        </div>
        <div className="flex items-center gap-2.5">
          <span className="text-caption text-label-3">Show</span>
          <SegmentedControl
            label="Number of gaps to show"
            value={topN}
            onChange={setTopN}
            options={TOP_N_OPTIONS}
            size="sm"
          />
        </div>
      </div>

      <div className="mt-5">
        <CorpusBanner domain={domain} />
      </div>

      <div className="mt-5">
        {loading && <LoadingList />}

        {!loading && error && (
          <ErrorState
            title="Could not load gaps"
            message={error}
            onRetry={() => setAttempt((n) => n + 1)}
          />
        )}

        {!loading && !error && gaps && gaps.length === 0 && (
          <EmptyState eyebrow="No results" title="No scored gaps for this domain">
            Nothing in this domain has produced a scored cluster yet. That
            usually means too few papers with explicit limitation statements
            have been ingested.
          </EmptyState>
        )}

        {!loading && !error && gaps && gaps.length > 0 && (
          <>
            <p className="sr-only" aria-live="polite">
              {gaps.length} gaps shown.
            </p>
            <div className="space-y-4">
              {gaps.map((gap, i) => (
                <GapCard
                  key={gap.gap_description}
                  gap={gap}
                  rank={i + 1}
                  index={i}
                  maxFrequency={maxFrequency}
                  onOpen={() => setSelected(gap)}
                />
              ))}
            </div>
          </>
        )}
      </div>

      <GapDetailSheet
        gap={selected}
        maxFrequency={maxFrequency}
        open={selected !== null}
        onClose={() => setSelected(null)}
      />
    </Container>
  );
}
