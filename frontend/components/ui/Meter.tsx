"use client";

import { motion } from "motion/react";
import { useSprings } from "@/lib/motion";

/**
 * One sub-score bar.
 *
 * `scaleMax` exists for frequency, and the reasoning behind it is a deliberate
 * honesty fix that must survive any restyle. `frequency_score` is
 * weighted_papers / total_papers_in_domain, so on a corpus this size real
 * values land around 0.01–0.06. Drawn on an absolute 0–1 scale the bar is
 * always visually empty, which reads as "no signal" or "broken" beside recency
 * and deficit sitting near 1.0. Drawing it relative to the largest value on
 * screen restores the comparison between gaps, which is the only comparison
 * that is actually meaningful here.
 *
 * Three things therefore hold, and none of them are styling choices:
 *   - the numeric readout always shows the TRUE score, at three decimals;
 *   - the label says the bar is relative, so a full bar is never read as 100%;
 *   - a 2% floor keeps a zero-value bar visible as a stub rather than absent.
 *
 * `role="meter"` with an explicit `aria-valuetext` carries the same caveat to
 * assistive tech, which previously got only a bare percentage.
 */
export default function Meter({
  label,
  value,
  colorVar,
  scaleMax,
  hint,
  relative = false,
}: {
  label: string;
  value: number;
  /** CSS custom property name for the fill, e.g. "--meter-recency". */
  colorVar: string;
  scaleMax?: number;
  hint?: string;
  relative?: boolean;
}) {
  const { spring } = useSprings();

  const denominator = scaleMax && scaleMax > 0 ? scaleMax : 1;
  const width = Math.max(2, Math.min(100, Math.round((value / denominator) * 100)));

  const valueText = relative
    ? `${value.toFixed(3)} — bar shown relative to the highest value on screen`
    : value.toFixed(3);

  return (
    <div className="flex items-center gap-3">
      <span className="w-40 shrink-0 text-caption text-label-3" title={hint}>
        {label}
      </span>
      <div
        role="meter"
        aria-label={label}
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuetext={valueText}
        className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-inset"
      >
        <motion.div
          className="h-full rounded-full"
          style={{ background: `var(${colorVar})` }}
          initial={false}
          animate={{ width: `${width}%` }}
          transition={spring("move")}
        />
      </div>
      <span className="tabular w-12 shrink-0 text-right text-caption text-label-2">
        {value.toFixed(3)}
      </span>
      {hint && <span className="sr-only">{hint}</span>}
    </div>
  );
}
