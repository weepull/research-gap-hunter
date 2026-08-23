"use client";

import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import Mark from "@/components/nav/Mark";
import Pressable from "@/components/ui/Pressable";
import { useSprings } from "@/lib/motion";

/**
 * Primary navigation.
 *
 * The bar is an opaque white plane rather than a translucent material. That is
 * a deliberate departure from the translucent-chrome default: on this paper
 * canvas a frosted bar reads as a smudge rather than as glass, and the page's
 * whole visual thesis is that depth comes from edges and type, not from tint.
 * A solid plane with a real bottom rule and a shadow is the honest version of
 * that argument. Because the bar is opaque there is no translucency to reduce,
 * so `prefers-reduced-transparency` has nothing to answer here.
 *
 * Link names describe contents rather than umbrellas: "Gaps" and "Connections"
 * say what is behind them where "Home" or "Explore" would not.
 */
const LINKS = [
  { href: "/gaps", label: "Gaps" },
  { href: "/search", label: "Search" },
  { href: "/cross-domain", label: "Connections" },
];

export default function Nav() {
  const pathname = usePathname();
  const { spring } = useSprings();

  return (
    <header className="nav-bar sticky top-0 z-50">
      <nav
        aria-label="Primary"
        className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-5 py-2.5 sm:px-6"
      >
        <Pressable
          as="link"
          href="/"
          feedback="opacity"
          className="flex items-center gap-3 rounded-sm"
        >
          <span className="text-label">
            <Mark />
          </span>
          {/* Wordmark and tagline are one locked unit, not a title with
              something trailing after it. The tagline earns its place by
              saying what the product is for in four words; it is set small,
              tracked open and stacked directly beneath so the pair reads as a
              single designed mark. */}
          {/* The lockup degrades in two steps. Below `sm` the wordmark and the
              three nav links cannot share a row — measured at 430px they need
              511px — so the wordmark becomes screen-reader-only and the
              logomark carries the brand alone. `sr-only` rather than `hidden`
              keeps it in the accessibility tree, so the link is never an
              unlabelled icon. The tagline, being the most expendable line,
              goes first. */}
          <span className="flex flex-col leading-none">
            <span className="nav-wordmark sr-only whitespace-nowrap text-label sm:not-sr-only">
              Research Gap Hunter
            </span>
            <span className="nav-tagline hidden text-label-3 sm:block">
              What to work on next
            </span>
          </span>
        </Pressable>

        <ul className="flex items-center gap-1">
          {LINKS.map(({ href, label }) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <li key={href} className="relative">
                <Pressable
                  as="link"
                  href={href}
                  feedback="opacity"
                  aria-current={active ? "page" : undefined}
                  className={`relative block rounded-sm px-3 py-1.5 text-subhead transition-colors ${
                    active ? "text-label" : "text-label-3 hover:text-label-2"
                  }`}
                >
                  {/* One indicator that travels between items, rather than one
                      per item fading in and out — where a thing came from is
                      part of what it means. */}
                  {active && (
                    <motion.span
                      layoutId="nav-indicator"
                      transition={spring("snap")}
                      className="absolute inset-0 -z-10 rounded-sm bg-accent-wash ring-1 ring-accent-line"
                      aria-hidden="true"
                    />
                  )}
                  <span className="relative">{label}</span>
                </Pressable>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
