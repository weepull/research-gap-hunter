"use client";

import { useState } from "react";
import Container from "@/components/ui/Container";
import Eyebrow from "@/components/ui/Eyebrow";
import Skeleton from "@/components/ui/Skeleton";
import EmptyState from "@/components/ui/EmptyState";
import ErrorState from "@/components/ui/ErrorState";
import Pressable from "@/components/ui/Pressable";
import CorpusBanner from "@/components/CorpusBanner";
import DomainPicker from "@/components/DomainPicker";
import ConnectionCard from "@/components/ConnectionCard";
import CrossDomainFinding from "@/components/CrossDomainFinding";
import { CrossDomainReport, domainLabel, fetchCrossDomainReport } from "@/lib/api";
import { describeCrossDomainFinding } from "@/lib/crossDomainFinding";

export default function CrossDomainPage() {
  const [source, setSource] = useState("computer_vision");
  const [target, setTarget] = useState("medical_imaging");
  const [report, setReport] = useState<CrossDomainReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sameDomain = source === target;
  const matches = report?.matches ?? null;
  const finding = report ? describeCrossDomainFinding(report, domainLabel) : null;

  // Explicitly user-triggered, so this is an event handler rather than an
  // effect — nothing here runs on mount.
  async function discover() {
    setLoading(true);
    setError(null);
    try {
      setReport(await fetchCrossDomainReport(source, target, 10));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Discovery failed");
      setReport(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Container width="wide" className="section-y">
      <header>
        <Eyebrow>Hypotheses</Eyebrow>
        <h1 className="mt-2 text-title-1 text-label">Cross-domain connections</h1>
        <p className="mt-2 max-w-[62ch] text-body text-label-2">
          Solutions proposed in one field, matched against unresolved problems
          in another. These pairings are candidate hypotheses, not findings —
          nobody has written them down.
        </p>
      </header>

      <div className="mt-6 flex flex-wrap items-end gap-x-6 gap-y-4 rounded-lg border border-hairline bg-surface card-p shadow-[var(--shadow-2)]">
        <div className="flex flex-col gap-2">
          <span className="text-caption text-label-3">Open problems from</span>
          <DomainPicker value={source} onChange={setSource} label="Source domain" />
        </div>
        <span aria-hidden="true" className="pb-2 text-label-4">
          &rarr;
        </span>
        <div className="flex flex-col gap-2">
          <span className="text-caption text-label-3">Solutions from</span>
          <DomainPicker value={target} onChange={setTarget} label="Target domain" />
        </div>
        <div className="flex flex-col gap-2">
          <span className="sr-only" aria-live="polite">
            {sameDomain ? "Source and target must be different domains." : ""}
          </span>
          <Pressable
            onClick={discover}
            disabled={loading || sameDomain}
            className="rounded-sm bg-accent px-5 py-2.5 text-headline text-white shadow-[var(--shadow-2)] hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-45"
          >
            {loading ? "Searching…" : "Find connections"}
          </Pressable>
        </div>
        {sameDomain && (
          <p className="text-caption text-caution">
            Pick two different domains — a field is not cross-referenced with itself.
          </p>
        )}
      </div>

      <div className="mt-5">
        <CorpusBanner domain={source} />
      </div>

      <div className="mt-5">
        {loading && (
          <div className="space-y-4" aria-busy="true" aria-label="Searching for connections">
            {[0, 1].map((i) => (
              <Skeleton key={i} className="h-48 w-full" rounded="rounded-lg" />
            ))}
          </div>
        )}

        {!loading && error && <ErrorState title="Discovery failed" message={error} onRetry={discover} />}

        {!loading && !error && report === null && (
          <EmptyState eyebrow="Start" title="Pick two fields and look for overlap">
            The matcher compares unresolved limitations in the source field
            against future directions proposed in the target field, and reports
            only pairs that clear the similarity threshold derived from this
            corpus&rsquo;s own measured noise floor.
          </EmptyState>
        )}

        {/* No matches is a finding, stated with its reason and evidence (T2): the API
            says whether no pair cleared the noise floor at this corpus size, no gap met
            the evidence gate, or there was no data. Never an empty panel. */}
        {!loading && !error && finding && <CrossDomainFinding finding={finding} />}

        {!loading && !error && report && !finding && matches && matches.length > 0 && (
          <>
            <p aria-live="polite" className="mb-3 text-caption text-label-3">
              {report.total_matches} {report.total_matches === 1 ? "connection" : "connections"} above
              threshold
              {report.total_matches > matches.length ? `, showing the top ${matches.length}` : ""}, from
              gaps supported by at least {report.min_seed_papers} papers
            </p>
            <div className="space-y-4">
              {matches.map((m, i) => (
                <ConnectionCard
                  key={`${m.source_gap}-${m.target_solution}-${i}`}
                  match={m}
                  index={i}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </Container>
  );
}
