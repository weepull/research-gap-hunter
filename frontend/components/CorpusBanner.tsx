"use client";

import { useEffect, useState } from "react";
import Skeleton from "@/components/ui/Skeleton";
import { CorpusInfo, domainLabel, fetchCorpus } from "@/lib/api";

/**
 * States what the numbers on this page were computed over.
 *
 * Shown on every results page. A ranked gap list reads very differently over
 * 31 papers than over 3,100, and without this a reader has no way to tell —
 * which is what made the output look more authoritative than it is.
 *
 * Two counts, because they are not the same number and the difference matters:
 * the corpus size, and the papers that actually reported a limitation. The
 * second is what divides `frequency_score`; a paper that extracted nothing
 * cannot corroborate a gap. This banner used to show only the first while
 * claiming it was the divisor, which was true then and would be wrong now.
 *
 * Counts arrive as `null` when the store behind them is unreachable, which is
 * deliberately distinct from a genuine zero — so this renders "unavailable"
 * rather than asserting an empty corpus under a working results list.
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
      <span className="tabular text-label">{info.papers ?? "an unknown number of"}</span>{" "}
      {domainLabel(info.domain)} {info.papers === 1 ? "paper" : "papers"}
      {updated && <> as of {updated}</>}
      {info.papers_reporting_limitations !== null && (
        <>
          , of which{" "}
          <span className="tabular text-label">
            {info.papers_reporting_limitations}
          </span>{" "}
          reported a limitation and so count toward frequency
        </>
      )}
      {" "}&mdash;{" "}
      {info.vectors_available ? (
        <>
          <span className="tabular">{info.limitations}</span> limitations,{" "}
          <span className="tabular">{info.future_directions}</span> future
          directions.
        </>
      ) : (
        <>limitation and future-direction counts unavailable.</>
      )}{" "}
      This is a curated sample, not a survey of the field.
    </p>
  );
}
