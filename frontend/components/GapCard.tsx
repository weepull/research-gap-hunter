"use client";

import { motion } from "motion/react";
import { GapResult } from "@/lib/api";
import Meter from "@/components/ui/Meter";
import ScoreDial from "@/components/ui/ScoreDial";
import SupportBadge from "@/components/SupportBadge";
import Pressable from "@/components/ui/Pressable";
import { useSprings } from "@/lib/motion";

/**
 * One ranked gap.
 *
 * Everything the previous card displayed is still displayed here: the rank,
 * the description, the support count, the proposed-solution count, all three
 * sub-scores with their true values, and the full list of proposed solutions.
 * The detail sheet *adds* the supporting papers as links and explains what each
 * sub-score means; it does not take anything off the card. Moving data behind a
 * tap would change what a reader sees at a glance, which is exactly the kind of
 * change this redesign is not allowed to make.
 */
export default function GapCard({
  gap,
  rank,
  maxFrequency,
  onOpen,
  index,
}: {
  gap: GapResult;
  rank: number;
  maxFrequency: number;
  onOpen: () => void;
  index: number;
}) {
  const { spring, travel } = useSprings();

  return (
    <motion.article
      // Transform-only entrance: content is never made invisible by chrome, so
      // a frame the animation does not get leaves the card low, not missing.
      initial={{ y: travel(10) }}
      animate={{ y: 0 }}
      // A short stagger, capped so a list of twenty does not take two seconds
      // to finish arriving.
      transition={{ ...spring("move"), delay: Math.min(index * 0.03, 0.24) }}
      className="rounded-lg border border-hairline bg-surface shadow-[var(--shadow-2)]"
    >
      <div className="card-p">
        <div className="flex items-start gap-4">
          <span
            aria-hidden="true"
            className="tabular mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-xs bg-surface-inset text-caption text-label-2"
          >
            {rank}
          </span>

          <div className="min-w-0 flex-1">
            <h2 className="text-title-3 text-label">{gap.gap_description}</h2>

            {/* Directly under the title: a reader cannot see the gap without
                also seeing how much evidence stands behind it. */}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <SupportBadge count={gap.supporting_papers.length} />
              <span className="text-caption text-label-3">
                {gap.proposed_solutions.length} proposed{" "}
                {gap.proposed_solutions.length === 1 ? "solution" : "solutions"}
              </span>
            </div>
          </div>

          <div className="flex shrink-0 flex-col items-center gap-1">
            <ScoreDial value={gap.score} />
            <span className="text-eyebrow uppercase text-label-3">Score</span>
          </div>
        </div>

        <div className="mt-5 space-y-2">
          <Meter
            label="Frequency (relative)"
            value={gap.frequency_score}
            colorVar="--meter-frequency"
            scaleMax={maxFrequency}
            relative
            hint="Share of the domain's papers reporting this limitation. The bar is scaled to the highest frequency on screen so gaps can be compared; the number is the true score."
          />
          <Meter
            label="Recency"
            value={gap.recency_score}
            colorVar="--meter-recency"
          />
          <Meter
            label="Solution deficit"
            value={gap.solution_deficit_score}
            colorVar="--meter-deficit"
          />
        </div>

        {gap.proposed_solutions.length > 0 && (
          <div className="mt-5 border-t border-hairline pt-4">
            <h3 className="text-eyebrow uppercase text-label-3">
              Proposed solutions
            </h3>
            <ul className="mt-2 space-y-1.5">
              {gap.proposed_solutions.map((solution, i) => (
                <li key={i} className="flex gap-2 text-callout text-label-2">
                  <span aria-hidden="true" className="text-label-4">
                    &rarr;
                  </span>
                  {solution}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="border-t border-hairline px-5 py-2.5">
        <Pressable
          onClick={onOpen}
          feedback="opacity"
          className="rounded-sm text-subhead text-accent-strong"
        >
          Evidence and method
          <span className="sr-only"> for: {gap.gap_description}</span>
          <span aria-hidden="true"> &rarr;</span>
        </Pressable>
      </div>
    </motion.article>
  );
}
