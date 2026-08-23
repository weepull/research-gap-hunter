import Hero from "@/components/landing/Hero";
import CorpusProof from "@/components/landing/CorpusProof";
import HowItWorks from "@/components/landing/HowItWorks";
import Limits from "@/components/landing/Limits";
import Entries from "@/components/landing/Entries";

/**
 * Landing page.
 *
 * The order is deliberate: what it is, what it was computed over, how it
 * works, what it can't do, then the ways in. The honesty material sits in the
 * first half of the page rather than at the bottom, because a reader forms
 * their view of what the output is worth before they reach the tool, not
 * after.
 */
export default function LandingPage() {
  return (
    <>
      <Hero />
      <CorpusProof />
      <HowItWorks />
      <Limits />
      <Entries />
    </>
  );
}
