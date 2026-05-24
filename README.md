# Aletheon

AI co-pilot for drug intelligence — evidence-backed drug research in minutes.

Enter a drug name → Aletheon fetches from many authentic sources (regulatory,
peer-reviewed, real-world, preprints, patents), retrieves the most relevant
evidence, and generates a **structured, cited report** — with preprints clearly
separated from peer-reviewed evidence.

> Backend-only prototype. Runs locally. See `Aletheon_14Day_Sprint_Plan.md` for the build plan.

---

## Project structure

```
aletheon/
├── main.py              # entry point
├── core/
│   ├── models.py        # ⭐ Evidence contract (the heart of the system)
│   ├── config.py        # settings + secrets (from .env)
│   ├── logging_setup.py # one shared logger
│   ├── chunk.py         # (Day 7) boundary-aware chunking
│   ├── embed.py         # (Day 2) embeddings
│   └── retrieve.py      # (Day 3) query → top-k chunks
├── sources/             # one fetch() per data source
│   └── pubmed.py        # (Day 2) first source
├── storage/
│   └── vectorstore.py   # (Day 2) Qdrant wrapper
├── report/
│   └── generate.py      # (Day 3) cited report
├── data/                # raw fetches + saved reports (git-ignored)
├── .env.example         # copy to .env and add your keys
└── requirements.txt
```

---

## Setup (Day 1)

1. **Make a virtual environment** (use Python 3.11 or 3.12 — NOT 3.14):
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your keys:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your OpenAI key
   ```

4. **Run the healthcheck:**
   ```bash
   python main.py
   ```
   You should see "Aletheon is alive ✅" and a sample Evidence line.

5. **Commit:**
   ```bash
   git init
   git add .
   git commit -m "Day 1: foundation — structure, Evidence contract, config, logging"
   ```

That's Day 1 done. Next: `python main.py "aspirin"` will run the real pipeline once Days 2–3 are built.

---

## The one rule

> No new layer until the current one runs end-to-end and is committed to git.
