"use client";

import { motion } from "motion/react";
import { CrossDomainMatch, domainLabel } from "@/lib/api";
import Chip from "@/components/ui/Chip";
import ExplanationPanel from "@/components/ExplanationPanel";
import { useSprings } from "@/lib/motion";

/**
 * One cross-domain pairing: an unresolved problem in one field beside a
 * solution proposed in another.
 *
 * The bridge between them is drawn rather than implied. The previous version
 * used two hairlines that were hidden below the medium breakpoint and a text
 * arrow, so on a phone the two panels read as unrelated stacked blocks — the
 * connection, which is the entire point of the card, disappeared exactly where
 * space was tightest.
 */
export default function ConnectionCard({
  match,
  index,
}: {
  match: CrossDomainMatch;
  index: number;
}) {
  const { spring, travel } = useSprings();

  return (
    <motion.article
      initial={{ y: travel(10) }}
      animate={{ y: 0 }}
      transition={{ ...spring("move"), delay: Math.min(index * 0.04, 0.2) }}
      className="overflow-hidden rounded-lg border border-hairline bg-surface shadow-[var(--shadow-2)]"
    >
      <h2 className="sr-only">
        Connection {index + 1}: a {domainLabel(match.source_domain)} gap paired
        with a {domainLabel(match.target_domain)} proposal, similarity{" "}
        {match.similarity_score.toFixed(3)}
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr]">
        <div className="border-b border-source-line bg-source-wash p-5 md:border-b-0 md:border-r">
          <p className="text-eyebrow uppercase text-source">
            {/* Real spaces, not margins: the separator is decorative and hidden
                from assistive tech, so without them a screen reader runs the
                domain name straight into the role. */}
            {domainLabel(match.source_domain)}{" "}
            <span aria-hidden="true" className="text-source-line">
              &middot;
            </span>{" "}
            unresolved gap
          </p>
          <p className="mt-2 text-callout text-label">{match.source_gap}</p>
          {match.source_papers.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {match.source_papers.map((id) => (
                <Chip key={id} id={id} tone="source" />
              ))}
            </div>
          )}
        </div>

        {/* The bridge. Present at every breakpoint — it rotates rather than
            disappearing, so the relationship survives a narrow screen. */}
        <div className="flex items-center justify-center gap-3 bg-surface px-4 py-3 md:flex-col md:px-5">
          <svg
            aria-hidden="true"
            className="h-6 w-12 rotate-90 md:h-14 md:w-6 md:rotate-0"
            viewBox="0 0 48 24"
            fill="none"
          >
            <path
              d="M2 12 H46"
              stroke="url(#bridge)"
              strokeWidth="2.25"
              strokeLinecap="round"
              strokeDasharray="1 5"
            />
            <circle cx="2" cy="12" r="2.5" fill="var(--source-graph)" />
            <circle cx="46" cy="12" r="2.5" fill="var(--target-graph)" />
            <defs>
              <linearGradient id="bridge" x1="0" y1="0" x2="48" y2="0" gradientUnits="userSpaceOnUse">
                <stop stopColor="var(--source-graph)" />
                <stop offset="1" stopColor="var(--target-graph)" />
              </linearGradient>
            </defs>
          </svg>
          <span
            className="tabular rounded-full border border-hairline bg-surface-inset px-2.5 py-0.5 text-caption text-label-2"
            title="Cosine similarity between the gap and the proposed solution, above the corpus's measured noise floor."
          >
            {match.similarity_score.toFixed(3)}
            <span className="sr-only"> cosine similarity</span>
          </span>
        </div>

        <div className="border-t border-target-line bg-target-wash p-5 md:border-t-0 md:border-l">
          <p className="text-eyebrow uppercase text-target">
            {/* Real spaces, not margins: the separator is decorative and hidden
                from assistive tech, so without them a screen reader runs the
                domain name straight into the role. */}
            {domainLabel(match.target_domain)}{" "}
            <span aria-hidden="true" className="text-target-line">
              &middot;
            </span>{" "}
            proposed solution
          </p>
          <p className="mt-2 text-callout text-label">{match.target_solution}</p>
          {match.target_papers.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {match.target_papers.map((id) => (
                <Chip key={id} id={id} tone="target" />
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-hairline px-5 py-4">
        <ExplanationPanel match={match} />
      </div>
    </motion.article>
  );
}
