"use client";

import { useSprings } from "@/lib/motion";

/**
 * Loading placeholder.
 *
 * The shimmer is suppressed under `prefers-reduced-motion` — the previous
 * implementation ran Tailwind's `animate-pulse` unconditionally, which is a
 * slow looping oscillation of exactly the kind reduced motion exists to stop.
 * The static state is a plain inset tint, which still reads as "content is
 * coming" without the loop.
 *
 * Marked `aria-hidden` and paired with `aria-busy` on the region that owns it:
 * a screen reader should hear "loading", not a description of grey boxes.
 */
export default function Skeleton({
  className = "",
  rounded = "rounded-sm",
}: {
  className?: string;
  rounded?: string;
}) {
  const { reduced } = useSprings();

  return (
    <div
      aria-hidden="true"
      className={[
        "bg-surface-inset",
        rounded,
        reduced ? "" : "shimmer",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    />
  );
}
