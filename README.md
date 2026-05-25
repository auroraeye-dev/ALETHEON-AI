# Aletheon

**AI co-pilot for drug intelligence — evidence-backed drug research in minutes.**

Enter a drug name and Aletheon runs a multi-agent pipeline that fetches evidence
from multiple authoritative sources, retrieves the most relevant material per
topic, and generates a **structured, fully-cited intelligence report** — with
confidence scoring, contradiction detection, and preprints clearly separated
from peer-reviewed evidence.

Backend prototype. Runs locally. A few dollars of API cost. No Docker, no AWS.

---

## What it does (the pipeline)

```
   drug name
       │
   ┌───┴────────────────────────────────────┐
   │  PARALLEL FETCH (LangGraph fan-out)     │
   │  FDA · ClinicalTrials.gov · Europe PMC  │
   └───┬────────────────────────────────────┘
       │  each returns Evidence (tiered)
   [combine]      dedup, tally by tier
       │
   [index]        boundary-aware chunk → embed → Qdrant
       │
   [retrieve]     section-targeted, drug-filtered, tier-biased
       │
   [report]       cited report: Summary · Key Findings (+confidence) ·
       │          Safety · Contradictions · Preprints (quarantined)
   cited report (Markdown, saved to data/reports/)
```

### Evidence tiers (the credibility backbone)
Every piece of evidence is tagged by authority:
`regulatory` (FDA) → `peer_reviewed` (trials, papers) → `real_world` →
`preprint` (NOT peer-reviewed) → `patent`. Confidence scoring weights by tier,
and preprints are confined to their own clearly-labeled section.

---

## Setup

Requires Python 3.11–3.14 and an OpenAI API key.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add your OPENAI_API_KEY
```

---

## Usage

```bash
# Full orchestrated pipeline (recommended): fetch -> ... -> report, one command
python main.py flow aspirin --reset

# Or the step-by-step path:
python main.py ingest aspirin --reset   # fetch + store evidence
python main.py "aspirin"                # generate report from stored evidence

# Inspect retrieval directly:
python main.py search "cardiovascular risks of aspirin"
```

Reports are printed and saved to `data/reports/{drug}/`.

Multiple drugs coexist in one store (evidence is drug-tagged), so after
ingesting several drugs you can generate any of their reports without re-ingesting.

---

## Architecture

```
aletheon/
├── main.py                 # entry point + CLI commands
├── core/
│   ├── models.py           # the Evidence contract (+ tiers)
│   ├── config.py           # settings/secrets from .env
│   ├── logging_setup.py    # shared logger
│   ├── chunk.py            # boundary-aware chunking (drug-tagged)
│   ├── embed.py            # OpenAI embeddings (batched)
│   ├── retrieve.py         # section-targeted, drug-filtered retrieval
│   ├── combine.py          # source registry + dedup
│   └── graph.py            # LangGraph orchestration (parallel fetch)
├── sources/                # one fetch() per source, all return Evidence
│   ├── fda.py · clinicaltrials.py · europepmc.py · pubmed.py
├── storage/
│   └── vectorstore.py      # Qdrant (embedded, local)
├── report/
│   └── generate.py         # cited report + confidence + contradictions
└── data/                   # raw + reports + qdrant (git-ignored)
```

### Local ↔ production mapping (for later scaling)
| Local (now) | Production (later) |
|---|---|
| Qdrant embedded | Qdrant Cloud / Kendra |
| OpenAI API | Bedrock |
| local folders | S3 |
| LangGraph (local) | LangGraph + Step Functions |
| (none) | React/Amplify UI |

---

## Status

**Build 1 (working prototype): complete.** Multi-source ingestion, orchestrated
parallel agents, drug-isolated retrieval, cited multi-tier reports with
confidence and contradiction detection.

Roadmap: more sources (FAERS, medRxiv, PubMed), comparator + critic agents,
evaluation harness, then web UI and cloud deployment.
