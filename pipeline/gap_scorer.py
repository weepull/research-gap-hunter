"""Gap scoring engine: clusters limitation statements and ranks research gaps."""

from pydantic import BaseModel


class GapResult(BaseModel):
    gap_description: str
    score: float
    frequency_score: float
    recency_score: float
    solution_deficit_score: float
    supporting_papers: list[str]
    proposed_solutions: list[str]


def cluster_limitations(limitation_vectors: list, texts: list[str]) -> dict[int, list[int]]:
    """Run HDBSCAN over limitation embeddings and return cluster_id → [indices] map."""
    pass


def compute_frequency_score(cluster_paper_ids: list[str], total_papers: int) -> float:
    """Return papers_reporting / total_papers_in_domain."""
    pass


def compute_recency_score(cluster_paper_ids: list[str], papers_meta: dict) -> float:
    """Return ratio of papers from last 2 years vs all-time within the cluster."""
    pass


def compute_solution_deficit_score(
    cluster_paper_ids: list[str], future_directions_map: dict
) -> float:
    """Return 1 - (future_directions_addressing / papers_reporting)."""
    pass


def score_gaps(domain: str = "computer_vision") -> list[GapResult]:
    """Load limitations from SQLite, cluster, score, and return ranked GapResult list.

    Scoring formula: 0.40*frequency + 0.35*recency + 0.25*solution_deficit
    """
    pass
