import Container from "@/components/ui/Container";
import Eyebrow from "@/components/ui/Eyebrow";

/**
 * What the tool does not do.
 *
 * Stated as its own section, at the same weight as the capabilities. Anticipating
 * how output can be misread is part of building the thing, and the single-paper
 * caveat in particular is the one a reader most needs before looking at a
 * ranked list — it is why every gap carries a visible support count.
 */
const LIMITS = [
  {
    title: "It is not a search engine.",
    body: "There is no retrieval over the literature here. It will not find you papers on a topic; it reports what a fixed, curated corpus says is unfinished.",
  },
  {
    title: "It does not verify claims.",
    body: "A limitation is counted because the authors wrote it down, not because it was checked. Extraction is an LLM pass and it can be wrong.",
  },
  {
    title: "A single paper can top the list.",
    body: "Recency and solution-deficit both saturate for a gap only one paper reports, so it can out-rank a better-attested one. Every gap shows its support count, and single-source gaps are flagged — nothing is filtered out to make the list look stronger.",
  },
  {
    title: "An empty result is a real answer.",
    body: "Cross-domain matching returns nothing when no pair clears the measured noise floor. That is the honest output, not a failure.",
  },
];

export default function Limits() {
  return (
    <Container width="wide" className="section-y">
      <Eyebrow>Limits</Eyebrow>
      <h2 className="mt-3 max-w-[24ch] text-title-1 text-label">
        What it doesn&rsquo;t do.
      </h2>

      <dl className="mt-6 grid gap-x-8 gap-y-6 sm:grid-cols-2">
        {LIMITS.map((limit) => (
          <div key={limit.title} className="border-t border-hairline pt-4">
            <dt className="text-headline text-label">{limit.title}</dt>
            <dd className="mt-1.5 text-callout text-label-2">{limit.body}</dd>
          </div>
        ))}
      </dl>
    </Container>
  );
}
