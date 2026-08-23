import type { ReactNode } from "react";

/**
 * Page width.
 *
 * This lives per-page rather than in the root layout so a hero can go
 * full-bleed while the content below it stays measured. The previous layout
 * hard-coded `max-w-6xl px-6 py-8` around every route, which made a
 * full-bleed section impossible.
 *
 * `prose` is capped near 68ch because a measure much beyond that costs the
 * reader the start of the next line.
 */
const WIDTH = {
  prose: "max-w-[68ch]",
  content: "max-w-5xl",
  wide: "max-w-6xl",
  full: "max-w-none",
} as const;

export default function Container({
  children,
  width = "wide",
  className = "",
  as: Tag = "div",
}: {
  children: ReactNode;
  width?: keyof typeof WIDTH;
  className?: string;
  as?: "div" | "section" | "main" | "header" | "footer";
}) {
  return (
    <Tag
      className={[
        "mx-auto w-full",
        WIDTH[width],
        // Gutters in rem so they scale with the user's text size along with
        // everything else, rather than pinning content to a fixed inset.
        "px-5 sm:px-6",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </Tag>
  );
}
