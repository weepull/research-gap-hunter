/**
 * The composite score, as a ring plus its number.
 *
 * The ring is *redundant* encoding, never a replacement: the number is always
 * present and always exact. Colour bands the score, but colour is never the
 * only carrier — the figure says the same thing to someone who cannot
 * distinguish the bands.
 */
export default function ScoreDial({ value }: { value: number }) {
  // Static class strings, not interpolated ones: Tailwind scans source text,
  // so a composed `text-${band}` class is never generated and silently falls
  // back to inherited colour.
  const BANDS = {
    positive: { stroke: "var(--positive-graph)", text: "text-positive" },
    caution: { stroke: "var(--caution-graph)", text: "text-caution" },
    negative: { stroke: "var(--negative-graph)", text: "text-negative" },
  } as const;

  const band =
    value > 0.6 ? "positive" : value >= 0.4 ? "caution" : "negative";
  const { stroke, text } = BANDS[band];

  const r = 19;
  const circumference = 2 * Math.PI * r;
  const filled = Math.max(0, Math.min(1, value)) * circumference;

  return (
    <div className="relative flex h-12 w-12 shrink-0 items-center justify-center">
      <svg
        className="absolute inset-0 -rotate-90"
        viewBox="0 0 44 44"
        aria-hidden="true"
        focusable="false"
      >
        <circle cx="22" cy="22" r={r} fill="none" stroke="var(--surface-inset)" strokeWidth="3" />
        <circle
          cx="22"
          cy="22"
          r={r}
          fill="none"
          stroke={stroke}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
        />
      </svg>
      <span className={`tabular relative text-caption ${text}`}>
        {value.toFixed(2)}
      </span>
    </div>
  );
}
