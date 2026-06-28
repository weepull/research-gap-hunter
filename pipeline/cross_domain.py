"""Cross-domain hypothesis matching: CV limitations ↔ Medical Imaging future directions."""

from pydantic import BaseModel


class CrossDomainMatch(BaseModel):
    source_gap: str
    target_solution: str
    similarity_score: float
    source_papers: list[str]
    target_papers: list[str]
    source_domain: str
    target_domain: str


def find_cross_domain_matches(
    source_domain: str = "computer_vision",
    target_domain: str = "medical_imaging",
    similarity_threshold: float = 0.78,
) -> list[CrossDomainMatch]:
    """Match unresolved limitations in source domain to future directions in target domain.

    Uses cosine similarity over Specter2 embeddings. Only returns pairs above threshold.
    """
    pass
