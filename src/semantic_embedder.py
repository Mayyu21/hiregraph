"""
Semantic Embedder — Layer 2D
Generates and stores embeddings for candidates and JD.
Uses all-MiniLM-L6-v2 via sentence-transformers.
CPU-only, no API calls.
"""

import json
import logging
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    """Load embedding model once and cache it."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading all-MiniLM-L6-v2 embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded.")
    return model


def candidate_to_text(candidate: dict) -> str:
    """
    Convert a candidate dict to a rich text representation for embedding.
    Includes profile, career, skills, education — maximizes semantic signal.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    edu = candidate.get("education", [])
    certs = candidate.get("certifications", [])

    parts = []

    # Profile section
    parts.append(f"Title: {profile.get('current_title', '')}")
    parts.append(f"Headline: {profile.get('headline', '')}")
    parts.append(f"Summary: {profile.get('summary', '')}")
    parts.append(f"Industry: {profile.get('current_industry', '')}")
    parts.append(f"Experience: {profile.get('years_of_experience', 0)} years")

    # Skills section
    skill_parts = []
    for s in skills:
        skill_parts.append(f"{s['name']} ({s.get('proficiency', 'intermediate')})")
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))

    # Career section
    for role in career[:5]:  # Top 5 roles
        role_text = f"{role.get('title')} at {role.get('company')} ({role.get('industry', '')}): {role.get('description', '')[:200]}"
        parts.append(role_text)

    # Education section
    for e in edu:
        parts.append(f"Education: {e.get('degree')} in {e.get('field_of_study')} from {e.get('institution')}")

    # Certifications
    for c in certs:
        parts.append(f"Certification: {c.get('name')} by {c.get('issuer')}")

    return " | ".join(parts)


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """
    Embed a list of texts using the BGE model.
    Returns normalized numpy array of shape (n, dim).
    """
    import faiss
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # BGE works best with normalization
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS inner product index (cosine on normalized vectors)."""
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def compute_semantic_scores(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between JD and all candidates.
    Returns array of scores in [0, 1].
    Both inputs must be normalized (done by embed_texts).
    """
    # Inner product of normalized vectors = cosine similarity
    scores = candidate_embeddings @ jd_embedding.T
    scores = scores.flatten()
    # Shift from [-1,1] to [0,1]
    scores = (scores + 1) / 2
    return scores.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    """Save embeddings to disk as numpy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embeddings)
    logger.info(f"Saved {embeddings.shape} embeddings to {path}")


def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from disk."""
    return np.load(str(path))


def precompute_candidate_embeddings(
    candidates: list[dict],
    output_path: Path,
    batch_size: int = 256
) -> np.ndarray:
    """
    Convert all candidates to text, embed them, save to disk.
    Run this during preprocessing — not during ranking.
    """
    logger.info(f"Converting {len(candidates)} candidates to text...")
    texts = [candidate_to_text(c) for c in candidates]

    logger.info("Computing embeddings...")
    embeddings = embed_texts(texts, batch_size=batch_size)

    save_embeddings(embeddings, output_path)
    return embeddings
