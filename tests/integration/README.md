# Integration test tier

Closes **PLAN.md item #7**.

```bash
pytest                      # 327 unit tests, ~0.8s, no services needed (default)
pytest -m integration       # this tier: real Neo4j + Qdrant + Ollama
pytest -m ''                # both tiers
```

## Why this exists

The unit suite is 327 tests in 0.78s and every service is mocked. That is a good
unit suite, and it is structurally incapable of catching any of the seven defects
in PLAN.md — all of which live in the interaction between real components and
real data. A cluster swallowing half a domain, a threshold below its own noise
floor, and a `/health` endpoint that reports `ok` with Neo4j down are all
invisible to a mock.

## Isolation

Nothing here touches the real corpus:

- **Neo4j** — a separate database, `rghintegration` by default
  (`NEO4J_TEST_DATABASE` to override). If it ever resolves to the same name as
  `NEO4J_DATABASE`, the fixture fails loudly rather than proceeding.
- **Qdrant** — separate collections, `limitations_integration` and
  `future_directions_integration`, dropped and rebuilt per session.
- **SQLite** — a `tmp_path` database; `pipeline.batch._DB_PATH` is redirected.

Services that are down cause a **skip**, not a failure, so a contributor without
Docker sees `skipped` rather than red that says nothing about their change.

## The fixture corpus

`fixtures/corpus.json` — 13 hand-labelled synthetic papers. arXiv ids use a
deliberately impossible `99xx` year prefix so nothing here can be mistaken for a
real paper (CLAUDE.md's rule about never trusting a recalled arXiv id applies to
fixtures too).

Each entry carries `true_domain` (ground truth) and `declared_domain` (what the
pre-fix pipeline stores). Three entries disagree on purpose, reproducing the live
defects in miniature:

| id | declared | true | mirrors |
|---|---|---|---|
| `9902.00006` | computer_vision | medical_imaging | `2305.17456` — clinical content stamped CV |
| `9903.00001` | computer_vision | other | `2405.00658` — number theory ingested as CV |
| `9903.00002` | computer_vision | other | `2501.03894` — control theory ingested as CV |

It also contains one zero-limitation paper per domain (so the frequency
denominator is observable) and one paper whose only limitations are the
extraction prompt's own example phrases, `"remains challenging"` and
`"future work includes"` (mirroring `2405.14458`).

## Fail-first baseline

Per this project's rule, every test here was confirmed to fail against pre-fix
code before being trusted. Baseline at commit `0eb58bc`:

```
18 failed, 12 passed
```

The 18 failures are the proof harness for items #1–#6 and are expected to be red
until the corresponding phase lands. The 12 that pass are either regression
guards (determinism, graph/SQLite agreement) or invariants the pre-fix code
already happens to satisfy.

Two ordering tests (`test_ranking_follows_the_documented_tiebreak_policy`,
`test_tied_gaps_are_not_ordered_by_description_length`) pass against pre-fix code
because this fixture does not happen to produce a misordered tie block, although
the live corpus has 18 of 27 CV gaps in exact-score ties ordered by description
length. Item #2 therefore also carries a unit test that pins the sort key
directly, and the live before/after counts are reported in the run summary.
