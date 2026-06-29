"""Populates Neo4j (rgh-mvp instance) from SQLite paper records."""

import logging
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from pipeline.batch import _get_db, _LIST_FIELDS, get_paper

load_dotenv()

logger = logging.getLogger(__name__)

_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Limitation) REQUIRE l.text IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:FutureDirection) REQUIRE f.text IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
]


def get_neo4j_driver():
    """Return an authenticated Neo4j Driver, verifying connectivity on creation."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def create_constraints(driver) -> None:
    """Run all uniqueness constraint Cypher statements against Neo4j."""
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        for cypher in _CONSTRAINTS:
            session.run(cypher)


def upsert_paper(tx, paper: dict) -> None:
    """MERGE Paper node on arxiv_id; set title, year, domain properties."""
    tx.run(
        """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title = $title,
            p.year  = $year,
            p.domain = $domain
        """,
        arxiv_id=paper["arxiv_id"],
        title=paper.get("title", ""),
        year=paper.get("year", 0),
        domain=paper.get("domain", "computer_vision"),
    )


def upsert_limitation(tx, paper_arxiv_id: str, limitation_text: str) -> None:
    """MERGE Limitation node on text; MERGE REPORTS_LIMITATION relationship to Paper."""
    tx.run(
        """
        MERGE (l:Limitation {text: $text})
        WITH l
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (p)-[:REPORTS_LIMITATION]->(l)
        """,
        text=limitation_text,
        arxiv_id=paper_arxiv_id,
    )


def upsert_future_direction(tx, paper_arxiv_id: str, text: str) -> None:
    """MERGE FutureDirection node on text; MERGE SUGGESTS_FUTURE relationship to Paper."""
    tx.run(
        """
        MERGE (f:FutureDirection {text: $text})
        WITH f
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (p)-[:SUGGESTS_FUTURE]->(f)
        """,
        text=text,
        arxiv_id=paper_arxiv_id,
    )


def upsert_method(tx, paper_arxiv_id: str, method_name: str) -> None:
    """MERGE Method node on name; MERGE USES_METHOD relationship to Paper."""
    tx.run(
        """
        MERGE (m:Method {name: $name})
        WITH m
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (p)-[:USES_METHOD]->(m)
        """,
        name=method_name,
        arxiv_id=paper_arxiv_id,
    )


def upsert_dataset(tx, paper_arxiv_id: str, dataset_name: str) -> None:
    """MERGE Dataset node on name; MERGE USES_DATASET relationship to Paper."""
    tx.run(
        """
        MERGE (d:Dataset {name: $name})
        WITH d
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        MERGE (p)-[:USES_DATASET]->(d)
        """,
        name=dataset_name,
        arxiv_id=paper_arxiv_id,
    )


def populate_graph(batch_size: int = 50) -> dict:
    """Read all papers from SQLite, upsert every node and relationship into Neo4j.

    Prints "[{i}/{total}] {arxiv_id} → graph" for each paper.
    Returns {"nodes_created": n, "relationships_created": n}.
    """
    db = _get_db()
    if "papers" not in db.table_names():
        return {"nodes_created": 0, "relationships_created": 0}

    arxiv_ids = [row[0] for row in db.execute("SELECT arxiv_id FROM papers").fetchall()]
    total = len(arxiv_ids)

    driver = get_neo4j_driver()
    create_constraints(driver)

    nodes_created = 0
    relationships_created = 0

    for i, arxiv_id in enumerate(arxiv_ids, start=1):
        print(f"[{i}/{total}] {arxiv_id} → graph")
        paper = get_paper(arxiv_id)
        if paper is None:
            continue

        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            # Paper node
            result = session.execute_write(
                lambda tx, p=paper: _upsert_paper_counting(tx, p)
            )
            nodes_created += result["nodes"]
            relationships_created += result["rels"]

    driver.close()
    return {"nodes_created": nodes_created, "relationships_created": relationships_created}


def _upsert_paper_counting(tx, paper: dict) -> dict:
    """Run all upserts for one paper inside a single transaction; return counts."""
    arxiv_id = paper["arxiv_id"]
    nodes = 0
    rels = 0

    # Paper node (1 node, 0 rels from this call)
    res = tx.run(
        """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        ON CREATE SET p.title = $title, p.year = $year, p.domain = $domain
        ON MATCH  SET p.title = $title, p.year = $year, p.domain = $domain
        RETURN p, (CASE WHEN p.arxiv_id = $arxiv_id THEN 1 ELSE 0 END) AS created
        """,
        arxiv_id=arxiv_id,
        title=paper.get("title", ""),
        year=paper.get("year", 0),
        domain=paper.get("domain", "computer_vision"),
    )
    summary = res.consume()
    nodes += summary.counters.nodes_created
    rels += summary.counters.relationships_created

    for limitation in paper.get("limitations") or []:
        if not limitation:
            continue
        res = tx.run(
            """
            MERGE (l:Limitation {text: $text})
            WITH l
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:REPORTS_LIMITATION]->(l)
            """,
            text=limitation,
            arxiv_id=arxiv_id,
        )
        s = res.consume()
        nodes += s.counters.nodes_created
        rels += s.counters.relationships_created

    for fd in paper.get("future_directions") or []:
        if not fd:
            continue
        res = tx.run(
            """
            MERGE (f:FutureDirection {text: $text})
            WITH f
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:SUGGESTS_FUTURE]->(f)
            """,
            text=fd,
            arxiv_id=arxiv_id,
        )
        s = res.consume()
        nodes += s.counters.nodes_created
        rels += s.counters.relationships_created

    for method in paper.get("methods") or []:
        if not method:
            continue
        res = tx.run(
            """
            MERGE (m:Method {name: $name})
            WITH m
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:USES_METHOD]->(m)
            """,
            name=method,
            arxiv_id=arxiv_id,
        )
        s = res.consume()
        nodes += s.counters.nodes_created
        rels += s.counters.relationships_created

    for dataset in paper.get("datasets") or []:
        if not dataset:
            continue
        res = tx.run(
            """
            MERGE (d:Dataset {name: $name})
            WITH d
            MATCH (p:Paper {arxiv_id: $arxiv_id})
            MERGE (p)-[:USES_DATASET]->(d)
            """,
            name=dataset,
            arxiv_id=arxiv_id,
        )
        s = res.consume()
        nodes += s.counters.nodes_created
        rels += s.counters.relationships_created

    return {"nodes": nodes, "rels": rels}
