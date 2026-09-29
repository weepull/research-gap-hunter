/**
 * How many papers back this gap, and which ranking tier it therefore sits in.
 *
 * The wording is deliberately "report similar limitations", not "report this
 * limitation". A gap is a *cluster* of semantically similar limitation statements, and
 * the label shown is one representative member — the one nearest the cluster centroid.
 * The papers did not write the same sentence, and claiming they did overstated the
 * evidence. For a multi-member cluster the old text was simply false.
 *
 * Since A9 Option F, corroborated gaps (>= 2 papers) rank above every single-source
 * gap, so **scores are non-monotonic across the tier boundary**: a single-source gap
 * further down the list may show a higher score. The badge carries the tier so that
 * reads as intended rather than as a broken sort.
 *
 * Three things are load-bearing and must survive any restyle: the badge sits directly
 * under the gap title, single-source is visually distinct and carries a warning glyph,
 * and the count is stated in words rather than implied.
 */
export default function SupportBadge({
  count,
  tier,
}: {
  count: number;
  tier?: "corroborated" | "single_source";
}) {
  // Fall back to the count if the tier is absent, so an older cached response still
  // renders sensibly rather than mislabelling.
  const single = tier ? tier === "single_source" : count === 1;

  const explanation = single
    ? "Only one paper reports a limitation like this — treat it as a single-source signal, not a corroborated trend. Single-source gaps are ranked below every corroborated gap regardless of score."
    : `${count} papers report similar limitations, which were grouped into one gap. The wording shown is the cluster member closest to the centroid, not a sentence all ${count} papers wrote.`;

  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-caption ${
        single
          ? "border-caution-line bg-caution-wash text-caution"
          : "border-hairline bg-surface-inset text-label-2"
      }`}
      title={explanation}
    >
      {single && <span aria-hidden="true">⚠</span>}
      {single
        ? "Single source — 1 paper"
        : `Corroborated — ${count} papers report similar limitations`}
      <span className="sr-only">. {explanation}</span>
    </span>
  );
}
