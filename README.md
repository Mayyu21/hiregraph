# HireGraph — Intelligent Candidate Discovery Engine

> **INDIA.RUNS Hackathon | Redrob AI × Hack2Skill | Track 1: Data & AI Challenge**
> Team: **TalentRadar** | Developer: **Mahesh** | Solo Submission

---

## What Is HireGraph?

HireGraph is a multi-signal AI candidate ranking engine that goes beyond keyword matching to intelligently rank candidates the way a great recruiter would — by understanding **who genuinely fits the role**, not just who has the right words in their profile.

Traditional hiring filters miss hidden gems. HireGraph finds them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1 — PREPROCESSING                  │
│                  (run once, API calls allowed)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  job_description.txt                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │  Layer 1        │  Gemini API → structured JSON          │
│  │  JD Parser      │  role, seniority, 24 skills,           │
│  │  jd_parser.py   │  must-haves, enriched 300w text        │
│  └────────┬────────┘                                        │
│           │  parsed_jd.json → cache/                        │
│           ▼                                                  │
│  candidates.jsonl (475MB, ~100k candidates)                  │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │  Layer 2D       │  all-MiniLM-L6-v2               │
│  │  Semantic       │  384-dim embeddings                   │
│  │  Embedder       │  L2 normalized → FAISS IndexFlatIP     │
│  └────────┬────────┘                                        │
│           │  candidate_embeddings.npy → cache/              │
│           │  candidate_ids.json → cache/                    │
└───────────┼─────────────────────────────────────────────────┘
            │
┌───────────┼─────────────────────────────────────────────────┐
│           ▼         PHASE 2 — RANKING                       │
│                  (zero API calls, CPU only, <5 min)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Layer 2 — Signal Extraction             │   │
│  │                                                      │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │   │
│  │  │  Signal A   │ │  Signal B   │ │   Signal C    │  │   │
│  │  │  Profile    │ │  Career     │ │  Behavioral   │  │   │
│  │  │             │ │             │ │               │  │   │
│  │  │ • Skills    │ │ • Trajectory│ │ • 23 Redrob   │  │   │
│  │  │   match     │ │ • Velocity  │ │   signals     │  │   │
│  │  │ • Exp fit   │ │ • Domain    │ │ • Availability│  │   │
│  │  │ • Edu tier  │ │   align     │ │ • Response    │  │   │
│  │  │ • Certs     │ │ • Company   │ │   rate        │  │   │
│  │  │ • Assess    │ │   growth    │ │ • Reliability │  │   │
│  │  │   scores    │ │ • Gap det.  │ │ • Market val  │  │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬────────┘  │   │
│  └─────────┼───────────────┼───────────────┼────────────┘   │
│            │               │               │                 │
│            ▼               ▼               ▼                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Layer 3 — Fusion Engine                 │   │
│  │                                                      │   │
│  │   Seniority-adaptive weights (senior role):          │   │
│  │   semantic(0.35) + profile(0.30) + career(0.25)     │   │
│  │   × behavioral multiplier (0.7 + 0.3 × beh_score)  │   │
│  │                                                      │   │
│  │   Hidden Gem Detection: high career_z, low profile_z │   │
│  │   Honeypot Detection: impossible profile signals     │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Layer 4 — Ranked Output                 │   │
│  │                                                      │   │
│  │   output/submission.csv                              │   │
│  │   candidate_id | rank | score | reasoning            │   │
│  │   Exactly 100 rows | Strictly decreasing scores      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## The 4-Signal Design

### Signal A — Semantic Score (35% weight)
Uses `all-MiniLM-L6-v2` embeddings (384 dimensions) to compute cosine similarity between the enriched JD text and each candidate's full profile text. This captures semantic meaning beyond keyword overlap — a candidate who "built a recommendation system" matches "ranking systems" even without using those exact words.

### Signal B — Profile Score (30% weight)
Structured skill matching with proficiency weighting: expert=1.0, advanced=0.8, intermediate=0.5, beginner=0.2. Also scores experience fit (penalizes underqualified AND overqualified), education tier (tier_1 through tier_4), certifications, and verified Redrob skill assessment scores as ground truth.

### Signal C — Career Trajectory Score (25% weight)
Detects upward progression across roles using seniority keyword mapping. Scores promotion velocity (sweet spot: 12-24 months per level), domain alignment (how many roles are in the target domain), company size growth, and penalizes unexplained gaps. Current role weighted 2x. This is how we find hidden gems — strong trajectory signals even when profile keywords don't match.

### Signal D — Behavioral Multiplier (applied on top)
Uses all 23 Redrob platform signals as a hiring availability multiplier: `final = base × (0.7 + 0.3 × behavioral_score)`. A perfect-on-paper candidate who hasn't logged in for 6 months and has 5% recruiter response rate gets down-weighted. An actively engaged candidate gets boosted. Key sub-signals: availability (open_to_work + recency), responsiveness (recruiter_response_rate + avg_response_time), reliability (interview_completion_rate + offer_acceptance_rate), market validation (saved_by_recruiters + views), and profile quality (completeness + verified signals + GitHub activity).

---

## Honeypot Detection

The dataset contains ~80 honeypot candidates with impossible profiles. HireGraph detects them using:

- Expert skills with 0 months duration (3+ = honeypot)
- Future start dates in career history
- Claimed years_of_experience > 2.5× sum of all role durations
- Negative role durations

Detected honeypots receive score = 0.001, pushing them to the bottom of rankings.

---

## Hidden Gem Detection

HireGraph flags underrated candidates whose true potential is hidden behind a weak-looking profile. A hidden gem has:
- Career trajectory z-score > 0.8 (strong upward progression)
- Profile score z-score < 0.3 (low keyword visibility)

These candidates built real systems but never optimized their profiles. The JD explicitly asks for this: *"A Tier 5 candidate may not use the words RAG or Pinecone but if their career history shows they built a recommendation system at a product company, they're a fit."*

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Phase 1 — Preprocess (run once)
```bash
# Set your Gemini API key
set GEMINI_API_KEY=your_key_here   # Windows
export GEMINI_API_KEY=your_key_here  # Linux/Mac

python src/pipeline.py --preprocess
```
This parses the JD with Gemini API and computes embeddings for all candidates. Takes 15-30 minutes for the full dataset. Results cached to `cache/`.

### Phase 2 — Rank (fast, no API)
```bash
python src/pipeline.py --rank
```
Loads cached embeddings, scores all candidates, outputs `output/submission.csv`. Runs in under 5 minutes on CPU.

### Full pipeline (preprocess + rank)
```bash
python src/pipeline.py
```

### Test on sample data (50 candidates)
```bash
python src/pipeline.py --sample
```

### Custom paths
```bash
python src/pipeline.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

### Run API server
```bash
uvicorn src.api:app --reload --port 8000
```

---

## Sample Output

```
candidate_id,rank,score,reasoning
CAND_0000031,1,0.661,"Recommendation Systems Engineer with 6.0 yrs exp; top skills: MLflow, FAISS; response rate 0.91; composite score 0.661"
CAND_0000045,2,0.596,"Project Manager with 12.2 yrs exp; top skills: GCP, Sales; response rate 0.62; composite score 0.596"
CAND_0000043,29,0.545,"Cloud Engineer with 8.3 yrs exp; top skills: Elasticsearch, OpenSearch; response rate 0.04; composite score 0.545"
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Vector search | FAISS IndexFlatIP (cosine on L2-normalized vectors) |
| JD parsing | Gemini 2.0 Flash (preprocessing only) |
| Data validation | Pydantic v2 |
| API framework | FastAPI + Uvicorn |
| Data processing | Pandas, NumPy |
| Testing | pytest |
| Python | 3.11+ |

---

## Why This Beats Keyword Matching

| Approach | What it misses |
|---|---|
| Keyword filter | Semantic matches, hidden gems, behavioral signals |
| TF-IDF | Context, proficiency levels, career trajectory |
| Simple embeddings | Behavioral availability, honeypots, seniority fit |
| **HireGraph** | Nothing — uses all available signals |

The JD itself says: *"The right answer involves reasoning about the gap between what the JD says and what the JD means."* HireGraph does exactly this through semantic embeddings + career trajectory analysis + behavioral multipliers.

---

## Submission Compliance

- ✅ Exactly 100 candidates in output
- ✅ Columns: candidate_id, rank, score, reasoning
- ✅ Scores strictly decreasing
- ✅ Tie-breaking by candidate_id ascending
- ✅ Zero API calls during ranking phase
- ✅ Runs in under 5 minutes on CPU
- ✅ Under 16GB RAM
- ✅ No GPU required
- ✅ Honeypot detection active
- ✅ UTF-8 encoding

---

## Team

**TalentRadar**

Developer: Mahesh

Track: Data & AI Challenge — Intelligent Candidate Discovery

Hackathon: INDIA.RUNS by Redrob AI × Hack2Skill
