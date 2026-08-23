import type { ReactNode } from "react";

/**
 * A card or panel.
 *
 * Elevation pairs radius, shadow and hairline as a set rather than leaving
 * each call site to assemble depth by hand. Depth scales with surface size:
 * a sheet casts further than a chip because it is further off the page, and
 * mixing the two reads as inconsistent lighting.
 *
 * Radii are concentric — a nested surface should use `outer − padding`, which
 * is why `elevation` and the padding utility are chosen together at the call
 * site rather than defaulted here.
 */

const ELEVATION = {
  1: "rounded-md shadow-[var(--shadow-1)]",
  2: "rounded-lg shadow-[var(--shadow-2)]",
  3: "rounded-lg shadow-[var(--shadow-3)]",
  4: "rounded-xl shadow-[var(--shadow-4)]",
} as const;

export default function Surface({
  children,
  elevation = 2,
  className = "",
  tone = "surface",
  bordered = true,
}: {
  children: ReactNode;
  elevation?: keyof typeof ELEVATION;
  className?: string;
  tone?: "surface" | "sunken" | "inset";
  bordered?: boolean;
}) {
  const background =
    tone === "sunken"
      ? "bg-surface-sunken"
      : tone === "inset"
        ? "bg-surface-inset"
        : "bg-surface";

  return (
    <div
      className={[
        background,
        ELEVATION[elevation],
        // The hairline is what keeps a white card legible against the tinted
        // canvas at the exact point where the shadow has faded to nothing.
        bordered ? "border border-hairline" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </div>
  );
}
