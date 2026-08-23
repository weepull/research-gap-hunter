/**
 * How many papers back this gap — the single most important qualifier on it.
 *
 * A one-paper gap can out-rank a well-attested one because recency and
 * solution-deficit both saturate at 1.0 for it, so support count has to be
 * impossible to miss. Single-source gaps are called out in amber rather than
 * hidden: the fix for over-stated confidence is showing the evidence, not
 * filtering the list. Nothing here filters anything.
 *
 * Three things are load-bearing and must survive any restyle: the badge sits
 * directly under the gap title, single-source is visually distinct and carries
 * a warning glyph, and the count is stated in words rather than implied.
 *
 * The explanatory sentence used to live only in a `title` attribute, which is
 * hover-only — invisible to keyboard and touch users entirely. It is now also
 * rendered as visually-hidden text so assistive tech reads it, and stated in
 * full, visibly, in the gap's detail sheet. The `title` is kept for pointer
 * users who have learned to hover it.
 */
export default function SupportBadge({ count }: { count: number }) {
  const single = count === 1;

  const explanation = single
    ? "Only one paper reports this limitation — treat as a single-source signal, not a corroborated trend."
    : `${count} papers independently report this limitation.`;

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
      Supported by {count} {count === 1 ? "paper" : "papers"}
      <span className="sr-only">. {explanation}</span>
    </span>
  );
}
