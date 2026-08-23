"use client";

import { motion } from "motion/react";
import Pressable from "@/components/ui/Pressable";
import Container from "@/components/ui/Container";
import { useSprings } from "@/lib/motion";

/**
 * The landing hero.
 *
 * The background is a static gradient, not an animated one: a full-viewport
 * moving background is one of the specific things to avoid, and a slow loop
 * behind text is worse than no motion at all.
 *
 * The honest sentence sits in the hero rather than in a footnote. What this
 * tool is — and what its output is worth — is the first thing a reader needs,
 * not a caveat they find later.
 */
export default function Hero() {
  const { spring, travel } = useSprings();

  return (
    <section className="relative overflow-hidden">
      {/* Two very low-opacity radial washes give the canvas some depth without
          ever becoming a "designed background" competing with the type. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10"
        style={{ background: "var(--hero-bg)" }}
      />

      <Container width="wide" className="hero-y">
        <motion.div
          // Transform-only entrance, for the same reason as Stat: content is
          // never made invisible by chrome. A slide that does not run leaves
          // the headline 14px low; a fade that does not run leaves a blank page.
          initial={{ y: travel(14) }}
          animate={{ y: 0 }}
          transition={spring("move")}
        >
          {/* The measure lives on the h1, not on this wrapper: `ch` resolves
              against the element's own font-size, so on a body-sized wrapper
              20ch is ~190px and the headline is forced into five stacked
              words. On the h1 it is 20 display characters, which is what was
              meant. */}
          <h1 className="hero-measure text-display text-label">Find what hasn&rsquo;t been done.</h1>
        </motion.div>

        <motion.p
          initial={{ y: travel(14) }}
          animate={{ y: 0 }}
          // A short stagger so the sentence reads as following the headline
          // rather than racing it. Delay, not a slower spring — the motion
          // itself should stay as quick as everything else.
          transition={{ ...spring("move"), delay: 0.06 }}
          className="mt-5 max-w-[58ch] text-body text-label-2"
        >
          Research Gap Hunter reads the limitations that papers state about
          themselves, clusters them, and ranks what the literature keeps
          flagging as unsolved. It answers{" "}
          <span className="text-label">what should be done next</span> — not
          what has already been published.
        </motion.p>

        <motion.div
          initial={{ y: travel(14) }}
          animate={{ y: 0 }}
          transition={{ ...spring("move"), delay: 0.12 }}
          className="mt-7 flex flex-wrap items-center gap-3"
        >
          <Pressable
            as="link"
            href="/gaps"
            className="rounded-sm bg-accent px-5 py-2.5 text-headline text-white shadow-[var(--shadow-2)] hover:bg-accent-hover"
          >
            Explore ranked gaps
          </Pressable>
          <Pressable
            as="link"
            href="/cross-domain"
            className="rounded-sm border border-hairline bg-surface px-5 py-2.5 text-headline text-label shadow-[var(--shadow-1)] hover:border-hairline-strong"
          >
            Cross-domain connections
          </Pressable>
        </motion.div>
      </Container>
    </section>
  );
}
