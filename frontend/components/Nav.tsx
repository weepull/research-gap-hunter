"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Gap Explorer" },
  { href: "/search", label: "Semantic Search" },
  { href: "/cross-domain", label: "Cross-Domain Discovery" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 border-b border-card-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        <Link href="/" className="flex items-baseline gap-2">
          <span className="text-lg font-bold tracking-tight text-foreground">
            Research Gap Hunter
          </span>
          <span className="hidden text-xs text-muted sm:inline">
            AI-powered scientific discovery
          </span>
        </Link>
        <div className="flex gap-1">
          {LINKS.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                  active
                    ? "bg-accent/15 text-accent font-medium"
                    : "text-muted hover:bg-card hover:text-foreground"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
