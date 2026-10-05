import type { Finding } from "../lib/crossDomainFinding";

/**
 * A cross-domain result with no matches, stated as a finding (T2).
 *
 * Deliberately not EmptyState: that dashed, sunken panel reads as "nothing here".
 * This is a solid card with the statement and the evidence behind it, announced as a
 * status. No `@/` imports, so the render test can load it under node:test.
 */
export default function CrossDomainFinding({ finding }: { finding: Finding }) {
  return (
    <section
      role="status"
      className="rounded-lg border border-hairline bg-surface card-p shadow-[var(--shadow-2)]"
    >
      <p className="text-eyebrow uppercase text-label-3">{finding.eyebrow}</p>
      <h2 className="mt-2 text-title-3 text-label">{finding.title}</h2>
      <p className="mt-2 max-w-[68ch] text-callout text-label-2">{finding.statement}</p>
      <dl className="mt-4 grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
        {finding.evidence.map((row) => (
          <div key={row.label} className="flex flex-col">
            <dt className="text-caption text-label-3">{row.label}</dt>
            <dd className="text-callout text-label">{row.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
