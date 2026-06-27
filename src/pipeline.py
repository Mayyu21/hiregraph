"""
HireGraph — Main Ranking Pipeline
Orchestrates all layers to rank candidates for a given job description.

Two modes:
  python pipeline.py --preprocess   (run once — computes embeddings, uses API)
  python pipeline.py --rank         (fast — no API, runs in <5 min on CPU)

Output: output/submission.csv with exactly 100 ranked candidates.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
SAMPLE_FILE = DATA_DIR / "sample_candidates.json"
JD_FILE = DATA_DIR / "job_description.txt"
PARSED_JD_FILE = CACHE_DIR / "parsed_jd.json"
EMBEDDINGS_FILE = CACHE_DIR / "candidate_embeddings.npy"
CANDIDATE_IDS_FILE = CACHE_DIR / "candidate_ids.json"
OUTPUT_FILE = OUTPUT_DIR / "submission.csv"


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_candidates(use_sample: bool = False) -> list[dict]:
    """Load candidates from JSONL or JSON file."""
    if use_sample and SAMPLE_FILE.exists():
        logger.info(f"Loading sample candidates from {SAMPLE_FILE}")
        with open(SAMPLE_FILE) as f:
            return json.load(f)

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"candidates.jsonl not found at {CANDIDATES_FILE}. "
            f"Copy it from the dataset into data/ folder."
        )
    logger.info(f"Loading full candidate dataset from {CANDIDATES_FILE}...")
    candidates = []
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    logger.info(f"Loaded {len(candidates)} candidates")
    return candidates


def load_jd() -> str:
    """Load job description text."""
    if not JD_FILE.exists():
        raise FileNotFoundError(
            f"job_description.txt not found at {JD_FILE}. "
            f"Copy the JD from the dataset into data/ folder."
        )
    with open(JD_FILE, "r", encoding="utf-8") as f:
        return f.read()


# ── Phase 1: Preprocessing ───────────────────────────────────────────────────

def run_preprocessing(use_sample: bool = False) -> None:
    """
    Phase 1 — Preprocessing (run once).
    - Parse JD with Gemini API
    - Generate candidate embeddings
    - Save everything to cache/
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 — PREPROCESSING")
    logger.info("=" * 60)

    # Step 1: Parse JD
    from src.jd_parser import parse_jd, save_parsed_jd, load_jd as read_jd
    logger.info("Step 1/3: Parsing job description with Gemini API...")
    jd_text = load_jd()
    parsed_jd = parse_jd(jd_text)
    save_parsed_jd(parsed_jd, PARSED_JD_FILE)
    logger.info(f"JD parsed: {parsed_jd.role_title} | {parsed_jd.seniority} | {len(parsed_jd.hard_skills)} skills")

    # Step 2: Load candidates
    logger.info("Step 2/3: Loading candidates...")
    candidates = load_candidates(use_sample=use_sample)

    # Step 3: Generate embeddings
    logger.info("Step 3/3: Computing candidate embeddings (this takes time)...")
    from src.semantic_embedder import precompute_candidate_embeddings
    embeddings = precompute_candidate_embeddings(
        candidates,
        output_path=EMBEDDINGS_FILE,
        batch_size=32
    )

    # Save candidate IDs in same order as embeddings
    ids = [c["candidate_id"] for c in candidates]
    with open(CANDIDATE_IDS_FILE, "w") as f:
        json.dump(ids, f)
    logger.info(f"Saved {len(ids)} candidate IDs to cache")

    logger.info("Preprocessing complete. Run with --rank to generate submission.")


# ── Phase 2: Ranking ──────────────────────────────────────────────────────────

def run_ranking(use_sample: bool = False) -> None:
    """
    Phase 2 — Ranking (no API, must finish in <5 min on CPU).
    Loads precomputed data and produces submission.csv.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 2 — RANKING (no API calls)")
    logger.info("=" * 60)

    # Load parsed JD
    if not PARSED_JD_FILE.exists():
        raise RuntimeError("Parsed JD not found. Run --preprocess first.")
    from src.jd_parser import load_parsed_jd
    parsed_jd = load_parsed_jd(PARSED_JD_FILE)
    logger.info(f"Loaded parsed JD: {parsed_jd.role_title}")

    # Load candidates
    candidates = load_candidates(use_sample=use_sample)
    cand_map = {c["candidate_id"]: c for c in candidates}

    # Load precomputed embeddings
    if not EMBEDDINGS_FILE.exists():
        raise RuntimeError("Embeddings not found. Run --preprocess first.")
    candidate_embeddings = np.load(str(EMBEDDINGS_FILE))
    with open(CANDIDATE_IDS_FILE) as f:
        embedding_ids = json.load(f)
    logger.info(f"Loaded embeddings: {candidate_embeddings.shape}")

    # Embed JD (fast — single text, no API)
    from src.semantic_embedder import embed_texts, compute_semantic_scores
    jd_embedding = embed_texts([parsed_jd.enriched_text])
    semantic_scores = compute_semantic_scores(jd_embedding, candidate_embeddings)
    semantic_score_map = {cid: float(score) for cid, score in zip(embedding_ids, semantic_scores)}

    # Initialize signal scorers
    from src.signal_extractor import ProfileSignal, CareerSignal, BehavioralSignal, detect_honeypot
    profile_scorer = ProfileSignal(
        jd_hard_skills=parsed_jd.hard_skills,
        jd_must_haves=parsed_jd.must_haves,
        jd_years_min=parsed_jd.years_experience_min,
        jd_years_max=parsed_jd.years_experience_max,
        jd_domain=parsed_jd.domain
    )
    career_scorer = CareerSignal(
        jd_domain=parsed_jd.domain,
        jd_industry=parsed_jd.industry_context
    )
    behavioral_scorer = BehavioralSignal()

    from src.fusion_engine import compute_composite, detect_hidden_gems, rank_candidates, CandidateScores

    logger.info("Scoring all candidates...")
    all_scores = []
    honeypot_count = 0

    for cid in embedding_ids:
        candidate = cand_map.get(cid)
        if not candidate:
            continue

        is_hp = detect_honeypot(candidate)
        if is_hp:
            honeypot_count += 1

        sem_score = semantic_score_map.get(cid, 0.0)
        prof_score = profile_scorer.score(candidate)
        car_score = career_scorer.score(candidate)
        beh_score = behavioral_scorer.score(candidate)

        scores = compute_composite(
            candidate_id=cid,
            semantic_score=sem_score,
            profile_score=prof_score,
            career_score=car_score,
            behavioral_score=beh_score,
            seniority=parsed_jd.seniority,
            is_honeypot=is_hp
        )
        all_scores.append(scores)

    logger.info(f"Scored {len(all_scores)} candidates | Honeypots detected: {honeypot_count}")

    # Detect hidden gems
    all_scores = detect_hidden_gems(all_scores)
    hidden_gem_count = sum(1 for s in all_scores if s.hidden_gem_flag)
    logger.info(f"Hidden gems identified: {hidden_gem_count}")

    # Rank candidates
    ranked = rank_candidates(all_scores)

    # Take top 100 (required by submission spec)
    top_100 = ranked[:100]

    # Generate reasoning for each candidate
    rows = []
    for rank_idx, scores in enumerate(top_100, start=1):
        candidate = cand_map.get(scores.candidate_id, {})
        profile = candidate.get("profile", {})
        rs = candidate.get("redrob_signals", {})
        skills = candidate.get("skills", [])

        # Build concise reasoning string
        top_skills = [s["name"] for s in skills[:3]]
        skills_str = ", ".join(top_skills) if top_skills else "various skills"
        yoe = profile.get("years_of_experience", 0)
        resp_rate = rs.get("recruiter_response_rate", 0)
        title = profile.get("current_title", "Professional")
        gem_note = " [Hidden Gem]" if scores.hidden_gem_flag else ""

        reasoning = (
            f"{title} with {yoe:.1f} yrs exp; "
            f"top skills: {skills_str}; "
            f"response rate {resp_rate:.2f}; "
            f"composite score {scores.composite_score:.3f}{gem_note}"
        )

        rows.append({
            "candidate_id": scores.candidate_id,
            "rank": rank_idx,
            "score": round(scores.composite_score, 6),
            "reasoning": reasoning
        })

    # Save submission CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    elapsed = time.time() - start_time
    logger.info(f"Submission saved to {OUTPUT_FILE}")
    logger.info(f"Total ranking time: {elapsed:.1f}s")
    logger.info(f"Top 5 candidates: {[r['candidate_id'] for r in rows[:5]]}")

    # Validate scores are strictly decreasing
    scores_list = [r["score"] for r in rows]
    if scores_list != sorted(scores_list, reverse=True):
        logger.warning("WARNING: Scores are not strictly decreasing — check fusion logic")
    else:
        logger.info("Score order validated: strictly decreasing")

    return df


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HireGraph — Intelligent Candidate Ranking")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocessing: parse JD and compute embeddings")
    parser.add_argument("--rank", action="store_true",
                        help="Run ranking: score and rank candidates (no API)")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample_candidates.json instead of full dataset")
    parser.add_argument("--candidates", type=str,
                        help="Path to candidates.jsonl file")
    parser.add_argument("--out", type=str,
                        help="Output path for submission CSV")
    args = parser.parse_args()

    # Override paths if provided
    global CANDIDATES_FILE, OUTPUT_FILE
    if args.candidates:
        CANDIDATES_FILE = Path(args.candidates)
    if args.out:
        OUTPUT_FILE = Path(args.out)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if args.preprocess:
        run_preprocessing(use_sample=args.sample)
    elif args.rank:
        run_ranking(use_sample=args.sample)
    else:
        # Default: run both
        logger.info("Running full pipeline (preprocess + rank)...")
        run_preprocessing(use_sample=args.sample)
        run_ranking(use_sample=args.sample)


if __name__ == "__main__":
    main()
