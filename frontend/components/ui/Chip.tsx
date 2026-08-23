import Pressable from "@/components/ui/Pressable";

/**
 * An arXiv identifier, linked to the paper.
 *
 * Set in mono because these are identifiers, not words — the fixed advance
 * makes a row of them scannable and stops digits from jittering between chips.
 *
 * The "opens on arXiv" note is visually hidden rather than absent: sighted
 * users get the affordance from the mono treatment and the surrounding
 * context, but a screen-reader user following a list of bare numbers has no
 * way to know where any of them go.
 */
export default function Chip({
  id,
  tone = "neutral",
}: {
  id: string;
  tone?: "neutral" | "source" | "target";
}) {
  const toneClass =
    tone === "source"
      ? "hover:border-source-line hover:bg-source-wash hover:text-source"
      : tone === "target"
        ? "hover:border-target-line hover:bg-target-wash hover:text-target"
        : "hover:border-accent-line hover:bg-accent-wash hover:text-accent-strong";

  return (
    <Pressable
      as="a"
      href={`https://arxiv.org/abs/${id}`}
      target="_blank"
      rel="noopener noreferrer"
      pressScale={0.95}
      className={`inline-flex items-center rounded-xs border border-hairline bg-surface-sunken px-2 py-0.5 font-mono text-caption text-label-2 transition-colors ${toneClass}`}
    >
      {id}
      <span className="sr-only"> — opens on arXiv in a new tab</span>
    </Pressable>
  );
}
