"""Integration proof for PLAN.md item #3 — cluster threshold and domain filter.

Fails against pre-fix code because _CLUSTER_THRESHOLD = 0.86 sits below the
limitation x limitation null p95 in both domains (CV 0.8736, MI 0.8957), so a
long generic seed absorbs a large share of its domain, and because
_query_neighbours issues an unfiltered query whose window is consumed by
other-domain points.
"""

import math

import pytest

pytestmark = pytest.mark.integration


def _limitations(domain: str) -> list[dict]:
    from pipeline.gap_scorer import get_all_limitations

    return get_all_limitations(domain)


def _expected_cap(n: int) -> int:
    """Mirror of gap_scorer's cap so the test states the rule independently."""
    from pipeline.gap_scorer import _MAX_CLUSTER_SHARE, _MIN_CLUSTER_CAP

    return max(_MIN_CLUSTER_CAP, math.ceil(_MAX_CLUSTER_SHARE * n))


@pytest.mark.parametrize("domain", ["computer_vision", "medical_imaging"])
def test_no_cluster_exceeds_the_domain_share_cap(loaded_corpus, domain):
    """No single cluster may hold more than the capped share of a domain.

    The live corpus puts 52 of 104 medical-imaging limitations (50%) in one
    cluster, labelled by an arbitrary member about convolution kernel sizes and
    containing unrelated statements about stain normalisation and FC-layer
    matrix sizes. That cluster is reported as a single gap 'supported by' 33
    papers, which is the most misleading output the system produces.
    """
    from pipeline.gap_scorer import cluster_limitations

    lims = _limitations(domain)
    if len(lims) < 2:
        pytest.skip(f"{domain} has too few limitations in the fixture to cluster")

    clusters = cluster_limitations(lims)
    cap = _expected_cap(len(lims))
    oversized = [len(c) for c in clusters if len(c) > cap]

    assert not oversized, (
        f"{domain}: {len(lims)} limitations, cap {cap}, "
        f"but found clusters of size {oversized}"
    )


@pytest.mark.parametrize("domain", ["computer_vision", "medical_imaging"])
def test_clusters_partition_the_limitations_exactly(loaded_corpus, domain):
    """Splitting a capped cluster must not drop or duplicate any limitation."""
    from pipeline.gap_scorer import cluster_limitations

    lims = _limitations(domain)
    if not lims:
        pytest.skip(f"{domain} has no limitations in the fixture")

    clusters = cluster_limitations(lims)
    assigned = [m["text"] for c in clusters for m in c]

    assert len(assigned) == len(lims), (
        f"{domain}: clustered {len(assigned)} members from {len(lims)} limitations — "
        "a split must re-seed the remainder, never truncate it"
    )
    assert sorted(assigned) == sorted(l["text"] for l in lims)


@pytest.mark.parametrize("domain", ["computer_vision", "medical_imaging"])
def test_clusters_never_mix_domains(loaded_corpus, domain):
    """A cluster's papers must all be in the scored domain."""
    from pipeline.gap_scorer import cluster_limitations

    in_domain = {p["arxiv_id"] for p in loaded_corpus["loaded"] if p["domain"] == domain}
    lims = _limitations(domain)
    if not lims:
        pytest.skip(f"{domain} has no limitations in the fixture")

    for cluster in cluster_limitations(lims):
        for member in cluster:
            foreign = set(member["paper_ids"]) - in_domain
            assert not foreign, f"{domain} cluster contains foreign papers: {foreign}"


def test_neighbour_query_is_domain_filtered(loaded_corpus):
    """The neighbour query must narrow by domain, like every other filtered query.

    Pre-fix, _query_neighbours passed no filter and limit=len(limitations), so a
    mean 42 of every 64 CV hits were other-domain points that were fetched,
    discarded, and — critically — crowded genuine same-domain neighbours out of
    the window. Three above-threshold CV pairs were lost that way.
    """
    from vectors.embed import _embed_texts, get_qdrant_client, load_embedding_model
    from pipeline.gap_scorer import _query_neighbours

    model = load_embedding_model()
    client = get_qdrant_client()
    vectors = _embed_texts(model, ["segmentation accuracy degrades on unseen anatomy"])

    hits = _query_neighbours(client, vectors, limit=50, domain="medical_imaging")[0]
    assert hits, "expected at least one neighbour in the fixture corpus"
    domains = {h.payload.get("domain") for h in hits}
    assert domains == {"medical_imaging"}, f"unfiltered neighbour query returned {domains}"
