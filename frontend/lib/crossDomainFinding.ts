/**
 * What a cross-domain result with no matches actually says (T2, 2026-10-05).
 *
 * The API used to return a bare list, and an empty list rendered as "nothing here",
 * which a reader takes to mean "no such connection exists". The API now returns a
 * status, and each non-match status is a different finding:
 *
 * - none_above_threshold — corroborated gaps were matched; no pair clears the noise
 *   floor at the current corpus size.
 * - no_corroborated_gaps — no source gap met the evidence gate; nothing was matched.
 * - no_data — one side has nothing to match.
 *
 * Kept free of imports (domain labels are passed in) so it runs under node:test
 * without the app's path aliases. Mirrors CrossDomainReport in pipeline/cross_domain.py.
 */

export type CrossDomainStatus =
  | "matches_found"
  | "none_above_threshold"
  | "no_corroborated_gaps"
  | "no_data";

export interface CrossDomainReportSummary {
  source_domain: string;
  target_domain: string;
  status: CrossDomainStatus;
  message: string;
  total_matches: number;
  similarity_threshold: number;
  min_seed_papers: number;
  source_gaps: number;
  seed_gaps: number;
  target_future_directions: number;
}

export interface Finding {
  eyebrow: string;
  title: string;
  statement: string;
  evidence: { label: string; value: string }[];
}

/** The finding to state for a report, or null when the matches are the result. */
export function describeCrossDomainFinding(
  report: CrossDomainReportSummary,
  label: (domain: string) => string,
): Finding | null {
  const src = label(report.source_domain);
  const tgt = label(report.target_domain);
  const threshold = report.similarity_threshold.toFixed(4);
  const evidence = [
    { label: "Gaps that met the evidence gate", value: `${report.seed_gaps} of ${report.source_gaps}` },
    { label: "Future directions searched", value: String(report.target_future_directions) },
    { label: "Similarity threshold", value: threshold },
    { label: "Evidence gate", value: `at least ${report.min_seed_papers} supporting papers, unresolved` },
  ];

  switch (report.status) {
    case "matches_found":
      return null;
    case "none_above_threshold":
      return {
        eyebrow: "Finding",
        title: `No corroborated ${src} gap has a ${tgt} match above the noise floor`,
        statement:
          `${report.seed_gaps} ${src} gap${report.seed_gaps === 1 ? "" : "s"} met the evidence gate ` +
          `and ${report.seed_gaps === 1 ? "was" : "were"} compared against ` +
          `${report.target_future_directions} ${tgt} future directions. ` +
          `None reached the similarity threshold of ${threshold}, the 95th percentile of ` +
          `similarity between unrelated pairs in this corpus. This is a result at the current ` +
          `corpus size: no pair in these papers clears the threshold.`,
        evidence,
      };
    case "no_corroborated_gaps":
      return {
        eyebrow: "Finding",
        title: `No ${src} gap meets the evidence gate`,
        statement:
          `None of the ${report.source_gaps} ${src} gaps is both supported by at least ` +
          `${report.min_seed_papers} papers and unresolved, so no matching was attempted. ` +
          `A gap reported by a single paper does not seed a cross-domain hypothesis.`,
        evidence,
      };
    case "no_data":
      return {
        eyebrow: "No data",
        title: "Nothing to match",
        statement: report.message,
        evidence,
      };
    default:
      throw new Error(`Unknown cross-domain status: ${String((report as { status: unknown }).status)}`);
  }
}
