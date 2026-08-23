import type { ReactNode } from "react";

/**
 * A translucent layer — chrome that content scrolls *under*, rather than an
 * opaque bar that permanently consumes a strip of the viewport.
 *
 * Weight encodes hierarchy: `thin` for navigation, `thick` for sheets. A
 * bigger surface has to read as thicker glass (stronger blur, deeper shadow),
 * not as the same glass stretched over more area.
 *
 * The `prefers-reduced-transparency` and `prefers-contrast` fallbacks live in
 * `globals.css` on `.material-thin` / `.material-thick`, deliberately in one
 * place: a caller cannot forget them because there is no way to opt out.
 *
 * Never nest one Material inside another — light on light destroys legibility,
 * which is the one hard rule of translucent surfaces.
 */
export default function Material({
  children,
  weight = "thin",
  className = "",
  as: Tag = "div",
}: {
  children: ReactNode;
  weight?: "thin" | "thick";
  className?: string;
  as?: "div" | "header" | "nav" | "aside" | "section";
}) {
  return (
    <Tag
      className={[
        weight === "thin" ? "material-thin" : "material-thick",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </Tag>
  );
}
