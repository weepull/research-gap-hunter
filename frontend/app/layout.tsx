import type { Metadata, Viewport } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "Research Gap Hunter",
  description:
    "Ranks what the research literature says is still unsolved — a discovery tool, not a search engine.",
};

export const viewport: Viewport = {
  // Matches --canvas so the browser chrome does not flash a different ground
  // colour before the page paints.
  themeColor: "#f1f0ee",
  colorScheme: "light",
};

/**
 * Root layout.
 *
 * Deliberately does *not* impose a width or padding on its children. The
 * previous version wrapped every route in `max-w-6xl px-6 py-8`, which made a
 * full-bleed hero impossible; page width is now each page's own decision via
 * the `Container` primitive.
 *
 * No webfont is loaded. The system stack already ships optical sizing,
 * tracking tables and legibility tuning that a downloaded face would discard,
 * and it costs nothing to fetch.
 */
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="flex min-h-full flex-col bg-canvas text-label">
        <Nav />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
