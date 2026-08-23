"use client";

import { motion } from "motion/react";
import { useSprings } from "@/lib/motion";
import type { ReactNode } from "react";

/**
 * A large numeric figure with its label.
 *
 * Hierarchy here is built from weight + size + leading as a set: the figure is
 * large, tight-leading and negatively tracked; the label is small, positively
 * tracked and set in a weaker label colour. Size alone would leave the pair
 * reading as two unrelated lines.
 *
 * Tabular figures are non-negotiable — these numbers update when the domain
 * changes, and proportional digits make the whole row twitch as they do.
 */
export default function Stat({
  value,
  label,
  detail,
  loading = false,
}: {
  value: ReactNode;
  label: string;
  detail?: string;
  loading?: boolean;
}) {
  const { spring, travel } = useSprings();

  return (
    <div className="flex flex-col gap-1">
      {loading ? (
        <div
          className="h-[2.75rem] w-24 rounded-sm bg-surface-inset"
          aria-hidden="true"
        />
      ) : (
        <motion.span
          // Transform only — opacity deliberately stays at 1.
          //
          // These figures are the payload of the section, and they mount after
          // an async fetch resolves. Starting them at opacity 0 means that any
          // frame the animation never gets (a starved rAF, a throttled
          // background tab, a headless render) leaves the numbers invisible
          // while their labels sit there looking broken — which is exactly
          // what happened. Animating position alone makes the worst case "the
          // number is 8px low", not "the number is gone".
          initial={{ y: travel(8) }}
          animate={{ y: 0 }}
          transition={spring("move")}
          className="tabular text-title-1 text-label"
        >
          {value}
        </motion.span>
      )}
      <span className="text-subhead text-label-2">{label}</span>
      {detail && <span className="text-footnote text-label-3">{detail}</span>}
    </div>
  );
}
