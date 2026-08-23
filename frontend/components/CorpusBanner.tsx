"use client";

import { useEffect, useState } from "react";
import { CorpusInfo, domainLabel, fetchCorpus } from "@/lib/api";

/**
 * States what the numbers on this page were computed over.
 *
 * Shown on every results page. A ranked gap list reads very differently over
 * 46 papers than over 4,600, and without this a reader has no way to tell —
 * which is what made the output look more authoritative than it is.
 */
export default function CorpusBanner({ domain }: { domain: string }) {
  const [info, setInfo] = useState<CorpusInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchCorpus(domain)
      .then((data) => {
        if (!cancelled) setInfo(data);
      })
      // A failed banner must never break the results below it.
      .catch(() => {
        if (!cancelled) setInfo(null);
      });
    return () => {
      cancelled = true;
    };
  }, [domain]);

  if (!info) return null;

  const updated = info.last_updated
    ? new Date(info.last_updated).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null;

  return (
    <div className="mb-6 rounded-lg border border-card-border bg-card/60 px-4 py-2.5 text-xs text-muted">
      <span className="font-medium text-foreground">Research prototype.</span>{" "}
      Computed over{" "}
      <span className="font-semibold tabular-nums text-foreground">{info.papers}</span>{" "}
      {domainLabel(info.domain)} {info.papers === 1 ? "paper" : "papers"}
      {updated && <> as of {updated}</>} —{" "}
      <span className="tabular-nums">{info.limitations}</span> limitations,{" "}
      <span className="tabular-nums">{info.future_directions}</span> future directions.
      This is a curated sample, not a survey of the field.
    </div>
  );
}
