"""Integration proof for PLAN.md items #4 (frequency denominator) and #2 (ordering).

Fails against pre-fix code because frequency_score divides by every Paper node in
the domain — including papers that extracted zero limitations and therefore can
never contribute a numerator (43% of CV, 38% of MI in the live corpus) — and
because equal scores are ordered by the stable sort's carry-through of
seed_order, which is descending description length.
"""

import pytest

pytestmark = pytest.mark.integration


# Function-scoped, deliberately. A module- or session-scoped fixture is created
# BEFORE function-scoped autouse fixtures run, so it would execute outside the
# store isolation in conftest._isolate_collections and score the *real* corpus
# instead of the fixture one. That is not hypothetical: it is what happened, and it
# only showed up when both tiers ran in one process. Recomputing per test costs
# little — the embedding model is session-cached and the fixture corpus is tiny.
@pytest.fixture
def cv_gaps(loaded_corpus):
    from pipeline.gap_scorer import score_gaps

    return score_gaps(domain="computer_vision", top_n=100)


@pytest.fixture
def mi_gaps(loaded_corpus):
    from pipeline.gap_scorer import score_gaps

    return score_gaps(domain="medical_imaging", top_n=100)


# ---------------------------------------------------------------------------
# #4 — frequency denominator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["computer_vision", "medical_imaging"])
def test_frequency_denominator_counts_only_contributing_papers(loaded_corpus, domain):
    """The divisor must be papers that extracted >= 1 limitation, not all papers.

    The fixture includes one zero-limitation paper per domain (9901.00004 survey,
    9902.00005 review) precisely so this is observable.
    """
    from pipeline.gap_scorer import _count_contributing_papers, _count_papers_in_domain

    all_papers = _count_papers_in_domain(domain)
    contributing = _count_contributing_papers(domain)
    expected = sum(
        1 for p in loaded_corpus["loaded"] if p["domain"] == domain and p["limitations"]
    )

    assert contributing == expected, f"{domain}: contributing {contributing} != expected {expected}"
    assert contributing < all_papers, (
        f"{domain}: fixture must contain a zero-limitation paper for this test to mean anything "
        f"(all={all_papers}, contributing={contributing})"
    )


def test_frequency_score_uses_the_contributing_denominator(cv_gaps, loaded_corpus):
    """A single-paper explicit-tier gap must score exactly 1.0 / contributing_papers."""
    from pipeline.gap_scorer import _count_contributing_papers

    contributing = _count_contributing_papers("computer_vision")
    singletons = [g for g in cv_gaps if len(g.supporting_papers) == 1]
    assert singletons, "fixture should produce at least one single-paper gap"

    # Tier weights: explicit 1.0, conclusion 0.75, inferred 0.5.
    permitted = {round(w / contributing, 4) for w in (1.0, 0.75, 0.5)}
    for gap in singletons:
        assert gap.frequency_score in permitted, (
            f"frequency {gap.frequency_score} not a tier weight over {contributing}: "
            f"{sorted(permitted)} — gap: {gap.gap_description[:60]}"
        )


# ---------------------------------------------------------------------------
# #2 — ordering
# ---------------------------------------------------------------------------


def _sort_key(gap):
    """The documented ranking policy: score, corroboration, recency, then text."""
    return (-gap.score, -len(gap.supporting_papers), gap.gap_description)


@pytest.mark.parametrize("which", ["cv_gaps", "mi_gaps"])
def test_ranking_follows_the_documented_tiebreak_policy(which, request):
    """Equal scores must be ordered by supporting-paper count, not description length.

    Pre-fix, the live CV list had 18 of 27 gaps in exact-score ties, and within
    each tie block rank tracked description length exactly: ranks 1-3 at 0.6065
    had lengths 72, 70, 64; ranks 4-7 at 0.6043 had 50, 44, 41, 34.
    """
    gaps = request.getfixturevalue(which)
    if len(gaps) < 2:
        pytest.skip("need at least two gaps to check ordering")

    scores = [g.score for g in gaps]
    assert scores == sorted(scores, reverse=True), "gaps must be sorted by score descending"

    for earlier, later in zip(gaps, gaps[1:]):
        if earlier.score != later.score:
            continue
        assert len(earlier.supporting_papers) >= len(later.supporting_papers), (
            "a tied gap with fewer supporting papers outranked a better-corroborated one: "
            f"{earlier.gap_description[:50]!r} ({len(earlier.supporting_papers)} papers) "
            f"before {later.gap_description[:50]!r} ({len(later.supporting_papers)} papers)"
        )


@pytest.mark.parametrize("which", ["cv_gaps", "mi_gaps"])
def test_tied_gaps_are_not_ordered_by_description_length(which, request):
    """Within a tie block, ordering must not track description length."""
    gaps = request.getfixturevalue(which)
    blocks: dict[float, list] = {}
    for gap in gaps:
        blocks.setdefault(gap.score, []).append(gap)

    for score, block in blocks.items():
        if len(block) < 3:
            continue
        lengths = [len(g.gap_description) for g in block]
        counts = [len(g.supporting_papers) for g in block]
        if len(set(counts)) == 1:
            continue  # equally corroborated: length order is coincidental, not causal
        assert lengths != sorted(lengths, reverse=True), (
            f"tie block at {score} is ordered by descending description length {lengths} "
            f"while supporting-paper counts differ {counts}"
        )


# ---------------------------------------------------------------------------
# #2 — cluster representative text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["computer_vision", "medical_imaging"])
def test_gap_description_is_the_centroid_nearest_member(loaded_corpus, domain):
    """gap_description must be the member nearest the cluster's vector centroid.

    Pre-fix it was Counter(...).most_common(1), which — because Limitation is
    UNIQUE on text, so every count is 1 — always resolved to insertion order,
    i.e. the seed, i.e. the longest string. Measured on the live corpus:
    description == longest member in 27 of 27 CV clusters.
    """
    import numpy as np

    from pipeline.gap_scorer import cluster_limitations, get_all_limitations
    from pipeline.gap_scorer import _cluster_representative_text
    from vectors.embed import _embed_texts, load_embedding_model

    lims = get_all_limitations(domain)
    if not lims:
        pytest.skip(f"{domain} has no limitations in the fixture")

    model = load_embedding_model()
    multi = [c for c in cluster_limitations(lims) if len(c) > 1]
    if not multi:
        pytest.skip(f"{domain} produced no multi-member clusters in the fixture")

    longest_wins = 0
    discriminating = 0
    for cluster in multi:
        texts = [m["text"] for m in cluster]
        vectors = np.asarray(_embed_texts(model, texts), dtype=float)
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        centroid = unit.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        similarities = unit @ centroid

        # Mirror the production tie-break rather than np.argmax's array order.
        # This is not a detail: in a TWO-member cluster both members are exactly
        # equidistant from their own centroid by symmetry, so the similarity is
        # always a tie and the label is always decided lexicographically. Using
        # argmax here would assert whatever order the cluster happened to be in.
        best = float(similarities.max())
        nearest = min(
            text for text, sim in zip(texts, similarities) if sim >= best - 1e-12
        )

        assert _cluster_representative_text(cluster) == nearest, (
            f"{domain}: representative text is not the centroid-nearest member.\n"
            f"  got:      {_cluster_representative_text(cluster)[:90]}\n"
            f"  expected: {nearest[:90]}"
        )
        if len(texts) > 2:
            discriminating += 1
            if nearest == max(texts, key=len):
                longest_wins += 1

    # Sanity: only clusters with 3+ members can discriminate the fix from the bug,
    # since a 2-member cluster's label is decided by the tie-break either way. If
    # every such cluster's centroid-nearest member is also its longest, this test
    # proves nothing and the fixture needs adjusting.
    if discriminating:
        assert longest_wins < discriminating, (
            f"{domain}: in all {discriminating} clusters of 3+ members the "
            "centroid-nearest member is also the longest, so this assertion cannot "
            "tell the fix from the bug — adjust the fixture"
        )


# ---------------------------------------------------------------------------
# determinism — must hold before and after every phase
# ---------------------------------------------------------------------------


def test_scoring_is_deterministic(loaded_corpus):
    """The same corpus must rank identically across calls, per gap_scorer's contract."""
    from pipeline.gap_scorer import score_gaps

    first = score_gaps(domain="medical_imaging", top_n=50)
    second = score_gaps(domain="medical_imaging", top_n=50)

    assert [g.gap_description for g in first] == [g.gap_description for g in second]
    assert [g.score for g in first] == [g.score for g in second]
