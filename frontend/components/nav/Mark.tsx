/**
 * The logomark: two measures with a deliberate space between them.
 *
 * The idea is the gap itself — the interval the tool exists to find. Drawn in
 * `currentColor` so it inherits text colour and needs no second asset.
 *
 * The bars are deliberately thin and widely separated. An earlier draft used
 * thicker bars set closer together and read unmistakably as a pause icon;
 * narrowing them and opening the gap to ~44% of the mark's width is what makes
 * the space, rather than the strokes, the thing you notice.
 *
 * Unequal heights matter for the same reason: two bars of identical height are
 * a transport control, two of different heights are a measurement.
 */
export default function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 22 22"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <rect x="2.6" y="2.4" width="3.1" height="17.2" rx="1.55" fill="currentColor" />
      <rect x="16.3" y="6.6" width="3.1" height="13" rx="1.55" fill="currentColor" />
    </svg>
  );
}
