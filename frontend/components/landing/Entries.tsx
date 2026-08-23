"use client";

import Container from "@/components/ui/Container";
import Eyebrow from "@/components/ui/Eyebrow";
import Pressable from "@/components/ui/Pressable";

/**
 * The three ways in.
 *
 * Each card says what is behind it in the terms the tool actually uses, so a
 * reader can predict what they will get before they click.
 *
 * `pressScale` is shallower than the 0.97 default: these are large surfaces,
 * and the same ratio moves far more pixels at the edge of a big card than on a
 * button, which reads as the card flexing rather than depressing.
 */
const ENTRIES = [
  {
    href: "/gaps",
    title: "Gaps",
    body: "Ranked open problems for a domain, each with its score breakdown, the papers behind it, and what has been proposed so far.",
  },
  {
    href: "/search",
    title: "Search",
    body: "Vector search across every extracted limitation statement in the corpus, by meaning rather than keyword.",
  },
  {
    href: "/cross-domain",
    title: "Connections",
    body: "Solutions proposed in one field matched to unresolved problems in another — the hypotheses nobody has written down yet.",
  },
];

export default function Entries() {
  return (
    <Container width="wide" className="section-y">
      <Eyebrow>Start here</Eyebrow>
      <h2 className="mt-3 max-w-[22ch] text-title-1 text-label">
        Three ways in.
      </h2>

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {ENTRIES.map((entry) => (
          <Pressable
            key={entry.href}
            as="link"
            href={entry.href}
            pressScale={0.985}
            className="group flex h-full flex-col rounded-lg border border-hairline bg-surface card-p text-left shadow-[var(--shadow-2)] transition-shadow hover:shadow-[var(--shadow-3)]"
          >
            <span className="text-title-3 text-label">{entry.title}</span>
            <span className="mt-2 flex-1 text-callout text-label-2">
              {entry.body}
            </span>
            <span
              aria-hidden="true"
              className="mt-4 text-subhead text-accent-strong"
            >
              Open &rarr;
            </span>
          </Pressable>
        ))}
      </div>
    </Container>
  );
}
