"use client";

import { GapResult } from "@/lib/api";
import Sheet from "@/components/ui/Sheet";
import Chip from "@/components/ui/Chip";
import Meter from "@/components/ui/Meter";
import Eyebrow from "@/components/ui/Eyebrow";

/**
 * The evidence behind one gap.
 *
 * Adds two things the list cannot show without becoming unreadable: the
 * supporting papers as links you can actually follow, and a plain-language
 * account of what each sub-score means and how to read it. No score is
 * recomputed here and no figure differs from the card — this is the same data
 * with room to explain itself.
 */
export default function GapDetailSheet({
  gap,
  maxFrequency,
  open,
  onClose,
}: {
  gap: GapResult | null;
  maxFrequency: number;
  open: boolean;
  onClose: () => void;
}) {
  return (
    <Sheet open={open && gap !== null} onClose={onClose} title="Gap detail" labelledBy="gap-detail-title">
      {gap && (
        <div>
          <Eyebrow>Gap</Eyebrow>
          <h2 id="gap-detail-title" className="mt-2 text-title-2 text-label">
            {gap.gap_description}
          </h2>

          <section className="mt-6">
            <h3 className="text-eyebrow uppercase text-label-3">
              Supporting papers ({gap.supporting_papers.length})
            </h3>
            {gap.supporting_papers.length === 1 && (
              <p className="mt-2 text-callout text-caution">
                Only one paper reports this limitation. Recency and solution
                deficit both saturate for a single-source gap, which is why one
                can out-rank a better-attested one. Read the rank accordingly.
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-1.5">
              {gap.supporting_papers.map((id) => (
                <Chip key={id} id={id} />
              ))}
            </div>
          </section>

          <section className="mt-7">
            <h3 className="text-eyebrow uppercase text-label-3">
              How this was scored
            </h3>
            <div className="mt-3 space-y-2">
              <Meter
                label="Frequency (corroboration)"
                value={gap.frequency_score}
                colorVar="--meter-frequency"
                scaleMax={maxFrequency}
                relative
              />
              <Meter label="Recency" value={gap.recency_score} colorVar="--meter-recency" />
              <Meter
                label="Solution deficit"
                value={gap.solution_deficit_score}
                colorVar="--meter-deficit"
              />
            </div>

            <dl className="mt-4 space-y-3 text-callout">
              <div>
                <dt className="text-headline text-label">
                  Frequency &mdash; weighted 40%, but it moves the ranking far less
                </dt>
                <dd className="text-label-2">
                  Of the papers in this domain that contributed at least one
                  limitation &mdash; not all papers in the domain &mdash; the
                  share reporting a limitation like this one, weighted by how
                  explicitly each stated it. Papers that extracted nothing are
                  excluded from the divisor, since they cannot corroborate
                  anything.
                  <br />
                  <span className="text-label-3">
                    Its coefficient is 0.40, but on a corpus this size the term
                    only varies across a narrow range, so it accounts for
                    roughly a tenth of what separates these gaps &mdash; read it
                    as corroboration, not as the main driver. Recency and
                    solution deficit do most of the ordering. The bar is drawn
                    relative to the largest value on screen; the number beside
                    it is always the true score.
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-headline text-label">Recency &mdash; weighted 35%</dt>
                <dd className="text-label-2">
                  The proportion of reporting papers published in the two years
                  up to the newest paper in the corpus, not the current calendar
                  year.
                  <br />
                  <span className="text-label-3">
                    This term spans its full range, so in practice it is the
                    strongest influence on the order.
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-headline text-label">
                  Solution deficit &mdash; weighted 25%
                </dt>
                <dd className="text-label-2">
                  How far the nearest proposed solution in the same domain is
                  from addressing it. Future directions from the paper that
                  raised the limitation are excluded, since restating your own
                  open problem is not a solution to it.
                  <br />
                  <span className="text-label-3">
                    Measured continuously: the closest eligible future direction
                    is rescaled between the similarity random pairs reach by
                    chance and the similarity at which a match is near-certain.
                    1.0 means nothing in the corpus comes closer than chance.
                  </span>
                </dd>
              </div>
            </dl>
          </section>

          {gap.proposed_solutions.length > 0 && (
            <section className="mt-7">
              <h3 className="text-eyebrow uppercase text-label-3">
                Proposed solutions ({gap.proposed_solutions.length})
              </h3>
              <ul className="mt-3 space-y-2">
                {gap.proposed_solutions.map((solution, i) => (
                  <li
                    key={i}
                    className="rounded-md border border-hairline bg-surface-sunken px-3 py-2 text-callout text-label-2"
                  >
                    {solution}
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-footnote text-label-3">
                These are exactly the future directions counted against the
                solution-deficit score &mdash; never a looser set.
              </p>
            </section>
          )}
        </div>
      )}
    </Sheet>
  );
}
