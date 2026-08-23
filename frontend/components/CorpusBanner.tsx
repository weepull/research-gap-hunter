"use client";

import { useEffect, useState } from "react";
import Skeleton from "@/components/ui/Skeleton";
import { CorpusInfo, domainLabel, fetchCorpus } from "@/lib/api";

/**
 * States what the numbers on this page were computed over.
 *
 * Shown on every results page. A ranked gap list reads very differently over
 * 46 papers than over 4,600, and without this a reader has no way to tell —
 * which is what made the output look more authoritative than it is.
 *
 * The figure comes from `/corpus`, which reports the Neo4j paper count that
 * divides `frequency_score`, so the number beside a score is the one the score
 * was actually computed against rather than a near-miss from another store.
 *
 * A failed fetch still renders nothing: a broken banner must never break the
 * results below it. The only change here is that loading now shows a
 * placeholder instead of nothing, so the banner no longer pops in after the
 * page has settled.
 */
export default function CorpusBanner({ domain }: { domain: string }) {
  const [info, setInfo] = useState<CorpusInfo | null>(null);
  const [settled, setSettled] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchCorpus(domain)
      .then((data) => {
        if (cancelled) return;
        setInfo(data);
        setSettled(true);
      })
      .catch(() => {
        if (cancelled) return;
        setInfo(null);
        setSettled(true);
      });
    return () => {
      cancelled = true;
    };
  }, [domain]);

  if (!settled) {
    return <Skeleton className="h-9 w-full" rounded="rounded-md" />;
  }

  if (!info) return null;

  const updated = info.last_updated
    ? new Date(info.last_updated).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null;

  return (
    <p
      aria-live="polite"
      className="rounded-md border border-hairline bg-surface-sunken px-4 py-2.5 text-footnote text-label-3"
    >
      <span className="text-label">Research prototype.</span> Computed over{" "}
      <span className="tabular text-label">{info.papers}</span>{" "}
      {domainLabel(info.domain)} {info.papers === 1 ? "paper" : "papers"}
      {updated && <> as of {updated}</>} &mdash;{" "}
      <span className="tabular">{info.limitations}</span> limitations,{" "}
      <span className="tabular">{info.future_directions}</span> future
      directions. This is a curated sample, not a survey of the field.
    </p>
  );
}
