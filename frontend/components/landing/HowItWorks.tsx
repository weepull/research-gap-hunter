import Container from "@/components/ui/Container";
import Eyebrow from "@/components/ui/Eyebrow";
import Surface from "@/components/ui/Surface";

/**
 * The pipeline, stated concretely.
 *
 * Each step names the actual mechanism rather than a vague verb — including
 * the real scoring weights. Specificity is what makes the output judgeable;
 * "AI-powered analysis" would tell a reader nothing they could check.
 */
const STEPS = [
  {
    n: "01",
    title: "Extract",
    body: "A local llama3.1 pass pulls structured fields out of each paper — objectives, methods, datasets, and above all the limitations the authors state about their own work.",
  },
  {
    n: "02",
    title: "Cluster",
    body: "Limitation statements are embedded with Specter2 and grouped by seed-anchored union-find, so the same problem described three different ways counts once.",
  },
  {
    n: "03",
    title: "Score",
    body: "Each cluster is ranked on how often it recurs, how recent those papers are, and how little of the literature proposes anything that addresses it — weighted 0.40 / 0.35 / 0.25.",
  },
  {
    n: "04",
    title: "Connect",
    body: "Unresolved problems in one field are matched against future directions proposed in another, above a similarity threshold derived from the corpus's own measured noise floor.",
  },
];

export default function HowItWorks() {
  return (
    <Container width="wide" className="section-y">
      <Eyebrow>How it works</Eyebrow>
      <h2 className="mt-3 max-w-[22ch] text-title-1 text-label">
        Four steps, no magic.
      </h2>

      <ol className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step) => (
          <li key={step.n}>
            <Surface elevation={1} className="h-full card-p">
              <span className="tabular text-caption text-accent-strong">
                {step.n}
              </span>
              <h3 className="mt-2 text-title-3 text-label">{step.title}</h3>
              <p className="mt-2 text-callout text-label-2">{step.body}</p>
            </Surface>
          </li>
        ))}
      </ol>
    </Container>
  );
}
