import type { ReactNode } from "react";

/**
 * The small uppercase label above a section title.
 *
 * Uppercase at this size needs markedly positive tracking (0.06em, set on the
 * `eyebrow` type step) — capitals set at body tracking read as a solid block
 * rather than as words.
 */
export default function Eyebrow({
  children,
  className = "",
  tone = "muted",
}: {
  children: ReactNode;
  className?: string;
  tone?: "muted" | "accent" | "source" | "target";
}) {
  const color =
    tone === "accent"
      ? "text-accent-strong"
      : tone === "source"
        ? "text-source"
        : tone === "target"
          ? "text-target"
          : "text-label-3";

  return (
    <p className={["text-eyebrow uppercase", color, className].join(" ")}>
      {children}
    </p>
  );
}
