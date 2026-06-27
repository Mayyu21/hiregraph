"""
Layer 3 — Weighted Fusion Engine
Combines profile, career, behavioral, and semantic scores into one
composite score. Weights adapt based on seniority level.
No API calls — pure math.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Weight profiles by seniority
# Behavioral weight is higher for junior (they need to prove intent)
# Profile weight is higher for senior (track record matters more)
WEIGHT_PROFILES = {
    "junior":    {"semantic": 0.35, "profile": 0.25, "career": 0.15, "behavioral": 0.25},
    "mid":       {"semantic": 0.35, "profile": 0.30, "career": 0.20, "behavioral": 0.15},
    "senior":    {"semantic": 0.35, "profile": 0.30, "career": 0.25, "behavioral": 0.10},
    "lead":      {"semantic": 0.30, "profile": 0.30, "career": 0.30, "behavioral": 0.10},
    "staff":     {"semantic": 0.25, "profile": 0.30, "career": 0.35, "behavioral": 0.10},
    "principal": {"semantic": 0.25, "profile": 0.25, "career": 0.40, "behavioral": 0.10},
    "default":   {"semantic": 0.35, "profile": 0.28, "career": 0.22, "behavioral": 0.15},
}

# Honeypot penalty — knocked down to near zero
HONEYPOT_SCORE = 0.001


@dataclass
class CandidateScores:
    """All individual scores for a candidate."""
    candidate_id: str
    semantic_score: float = 0.0
    profile_score: float = 0.0
    career_score: float = 0.0
    behavioral_score: float = 0.0
    composite_score: float = 0.0
    is_honeypot: bool = False
    hidden_gem_flag: bool = False


def compute_composite(
    candidate_id: str,
    semantic_score: float,
    profile_score: float,
    career_score: float,
    behavioral_score: float,
    seniority: str = "default",
    is_honeypot: bool = False,
) -> CandidateScores:
    """
    Combine all signals into one composite score.

    Args:
        candidate_id: Candidate identifier
        semantic_score: Embedding similarity score 0-1
        profile_score: Static profile match score 0-1
        career_score: Career trajectory score 0-1
        behavioral_score: Redrob behavioral signals score 0-1
        seniority: JD seniority level for weight selection
        is_honeypot: If True, score is penalized to near zero

    Returns:
        CandidateScores dataclass with composite score
    """
    scores = CandidateScores(candidate_id=candidate_id)
    scores.semantic_score = semantic_score
    scores.profile_score = profile_score
    scores.career_score = career_score
    scores.behavioral_score = behavioral_score
    scores.is_honeypot = is_honeypot

    if is_honeypot:
        scores.composite_score = HONEYPOT_SCORE
        return scores

    weights = WEIGHT_PROFILES.get(seniority.lower(), WEIGHT_PROFILES["default"])

    composite = (
        weights["semantic"]   * semantic_score +
        weights["profile"]    * profile_score +
        weights["career"]     * career_score +
        weights["behavioral"] * behavioral_score
    )
    scores.composite_score = max(0.0, min(1.0, composite))
    return scores


def detect_hidden_gems(all_scores: list[CandidateScores]) -> list[CandidateScores]:
    """
    Flag hidden gems: candidates with strong career trajectory
    but lower profile/semantic visibility.

    A hidden gem = high career_score relative to their profile_score.
    These are the "rough diamonds" the problem statement mentions.
    """
    if not all_scores:
        return all_scores

    # Compute z-scores for career and profile signals
    careers = [s.career_score for s in all_scores]
    profiles = [s.profile_score for s in all_scores]

    mean_career = sum(careers) / len(careers)
    mean_profile = sum(profiles) / len(profiles)

    std_career = (sum((x - mean_career) ** 2 for x in careers) / len(careers)) ** 0.5 or 1.0
    std_profile = (sum((x - mean_profile) ** 2 for x in profiles) / len(profiles)) ** 0.5 or 1.0

    for s in all_scores:
        career_z = (s.career_score - mean_career) / std_career
        profile_z = (s.profile_score - mean_profile) / std_profile
        # Hidden gem: career trajectory is significantly above average
        # but profile visibility is not — they are underrated
        if career_z > 0.8 and profile_z < 0.3 and not s.is_honeypot:
            s.hidden_gem_flag = True

    return all_scores


def rank_candidates(all_scores: list[CandidateScores]) -> list[CandidateScores]:
    """
    Sort candidates by composite score descending.
    Honeypots always go to the bottom.
    """
    return sorted(all_scores, key=lambda s: s.composite_score, reverse=True)
