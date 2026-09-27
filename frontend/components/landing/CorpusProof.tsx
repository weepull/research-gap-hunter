"use client";

import { useEffect, useState } from "react";
import Container from "@/components/ui/Container";
import Eyebrow from "@/components/ui/Eyebrow";
import Stat from "@/components/ui/Stat";
import Surface from "@/components/ui/Surface";
import { CorpusInfo, DOMAINS, fetchCorpus } from "@/lib/api";

/**
 * The live corpus figures, from the same `/corpus` endpoint the results pages
 * use — which reports the Neo4j paper count that divides `frequency_score`,
 * so the number shown here is the number the scores were computed against.
 *
 * This is the honesty note promoted to the headline. A ranked gap list means
 * something very different over 46 papers than over 4,600, and putting the
 * real figure up front is a better answer than a disclaimer further down.
 *
 * A failed fetch renders nothing rather than an error: the landing page must
 * not turn a background request into a visible failure.
 */
export default function CorpusProof() {
  const [corpora, setCorpora] = useState<CorpusInfo[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all(DOMAINS.map((d) => fetchCorpus(d.value)))
      .then((data) => {
        if (!cancelled) setCorpora(data);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (failed) return null;

  // Counts are null when the store behind them is unreachable, which must read
  // as "unknown" rather than as a confident zero (PLAN.md #6).
  const formatted = (n: number | null) => (n === null ? "—" : n.toLocaleString());
  const asOf = (iso: string | null) =>
    iso
      ? new Date(iso).toLocaleDateString(undefined, {
          year: "numeric",
          month: "short",
          day: "numeric",
        })
      : null;

  return (
    <Container width="wide" className="section-y">
      <Eyebrow>What the numbers are computed over</Eyebrow>
      <h2 className="mt-3 max-w-[24ch] text-title-1 text-label">
        A curated sample, stated plainly.
      </h2>
      <p className="mt-3 max-w-[62ch] text-body text-label-2">
        Every score on this site is derived from the corpus below and nothing
        else. It is a research prototype over hand-picked papers — not a survey
        of either field.
      </p>

      <div
        className="mt-6 grid gap-4 sm:grid-cols-2"
        aria-busy={corpora === null}
      >
        {(corpora ?? DOMAINS.map(() => null)).map((info, i) => {
          const label = DOMAINS[i].label;
          const updated = info ? asOf(info.last_updated) : null;
          return (
            <Surface key={label} elevation={2} className="card-p">
              <div className="flex items-baseline justify-between gap-3">
                <h3 className="text-title-3 text-label">{label}</h3>
                {updated && (
                  <span className="text-footnote text-label-3">
                    as of {updated}
                  </span>
                )}
              </div>
              <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat
                  value={info ? formatted(info.papers) : ""}
                  label="Papers"
                  loading={!info}
                />
                <Stat
                  value={info ? formatted(info.papers_reporting_limitations) : ""}
                  label="Reporting a limitation"
                  loading={!info}
                />
                <Stat
                  value={info ? formatted(info.limitations) : ""}
                  label="Limitations"
                  loading={!info}
                />
                <Stat
                  value={info ? formatted(info.future_directions) : ""}
                  label="Future directions"
                  loading={!info}
                />
              </div>
            </Surface>
          );
        })}
      </div>
    </Container>
  );
}
