"use client";

import { useState } from "react";
import Pressable from "@/components/ui/Pressable";
import Notice from "@/components/ui/Notice";
import Skeleton from "@/components/ui/Skeleton";
import {
  CrossDomainMatch,
  FeatureDisabledError,
  fetchExplanation,
} from "@/lib/api";

/**
 * The generated account of why a cross-domain pairing might matter.
 *
 * Three outcomes are kept visually distinct on purpose:
 *
 *  - a generated explanation;
 *  - the feature being switched off for this deployment, which is an expected
 *    state and is rendered as information rather than as failure — this
 *    deployment disables live explanations to avoid unbounded LLM cost;
 *  - a genuine error.
 *
 * Collapsing the middle case into the third would make a working, deliberate
 * configuration look broken.
 */
export default function ExplanationPanel({ match }: { match: CrossDomainMatch }) {
  const [explanation, setExplanation] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [disabled, setDisabled] = useState<string | null>(null);

  async function explain() {
    setPending(true);
    setError(null);
    try {
      const res = await fetchExplanation(
        match.source_gap,
        match.target_solution,
        match.source_domain,
        match.target_domain,
      );
      setExplanation(res.explanation);
    } catch (err) {
      if (err instanceof FeatureDisabledError) setDisabled(err.message);
      else setError(err instanceof Error ? err.message : "Explanation failed");
    } finally {
      setPending(false);
    }
  }

  if (explanation) {
    return (
      <div>
        <h4 className="text-eyebrow uppercase text-label-3">
          Why this connection might matter
        </h4>
        <p className="mt-2 text-callout text-label-2">{explanation}</p>
      </div>
    );
  }

  if (disabled) {
    return (
      <Notice title="Explanations are off in this demo">
        {disabled}
      </Notice>
    );
  }

  if (pending) {
    return (
      <div aria-busy="true" aria-label="Generating explanation" className="space-y-2">
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-11/12" />
        <Skeleton className="h-3 w-3/5" />
        <p className="text-caption text-label-3">
          Generating with a local model — this usually takes about ten seconds.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Pressable
        onClick={explain}
        className="rounded-sm border border-hairline-strong bg-surface px-3.5 py-2 text-subhead text-label shadow-[var(--shadow-1)]"
      >
        Explain this connection
      </Pressable>
      {error && (
        <span role="alert" className="text-caption text-negative">
          {error}
        </span>
      )}
    </div>
  );
}
