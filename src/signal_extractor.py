"""
Layer 2 — Candidate Signal Extractor
Three signal classes: ProfileSignal, CareerSignal, BehavioralSignal.
All scores normalized to 0.0-1.0.
No API calls — pure computation.
"""

import math
import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

REFERENCE_DATE = date(2026, 6, 25)  # Competition reference date


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, val))


def _parse_date(d: Optional[str]) -> Optional[date]:
    """Parse ISO date string to date object."""
    if not d:
        return None
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _days_since(d: Optional[str]) -> float:
    """Days since a date string. Returns large number if None."""
    parsed = _parse_date(d)
    if not parsed:
        return 9999.0
    return max(0.0, (REFERENCE_DATE - parsed).days)


def _minmax(val: float, lo: float, hi: float) -> float:
    """Min-max normalize val in [lo, hi] to [0, 1]."""
    if hi <= lo:
        return 0.5
    return _clamp((val - lo) / (hi - lo))


# ─────────────────────────────────────────────
# Honeypot Detection
# ─────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with impossible profiles (honeypots).
    Returns True if candidate is likely a honeypot.
    
    Checks:
    - Experience at company longer than company has existed
    - Expert skills with 0 months duration
    - Negative duration roles
    - Future start dates
    """
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check 1: expert skill with 0 months duration
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero_duration >= 3:
        return True

    # Check 2: role duration impossibly long vs company age
    for role in career:
        start = _parse_date(role.get("start_date"))
        if start and start > REFERENCE_DATE:
            return True  # Future start date
        duration = role.get("duration_months", 0)
        if duration < 0:
            return True  # Negative duration

    # Check 3: years of experience inconsistent with career history
    profile_yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    total_career_years = total_career_months / 12.0
    if total_career_years > 0 and profile_yoe > total_career_years * 2.5:
        return True  # Claimed YOE way more than sum of roles

    return False


# ─────────────────────────────────────────────
# Signal A — Profile Signal
# ─────────────────────────────────────────────

class ProfileSignal:
    """
    Scores a candidate's static profile against a parsed JD.
    Returns float 0.0-1.0.
    """

    def __init__(self, jd_hard_skills: list[str], jd_must_haves: list[str],
                 jd_years_min: float, jd_years_max: float, jd_domain: str):
        self.jd_skills = [s.lower() for s in jd_hard_skills]
        self.jd_must_haves = [m.lower() for m in jd_must_haves]
        self.jd_years_min = jd_years_min
        self.jd_years_max = jd_years_max
        self.jd_domain = jd_domain.lower()

    def score(self, candidate: dict) -> float:
        """Compute profile score for a candidate."""
        try:
            profile = candidate.get("profile", {})
            skills = candidate.get("skills", [])
            certs = candidate.get("certifications", [])
            edu = candidate.get("education", [])

            # 1. Skill match score (40%)
            skill_names = [s["name"].lower() for s in skills]
            skill_prof = {s["name"].lower(): s.get("proficiency", "beginner") for s in skills}
            prof_weights = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

            if self.jd_skills:
                matched = 0.0
                for jd_skill in self.jd_skills:
                    for cand_skill in skill_names:
                        if jd_skill in cand_skill or cand_skill in jd_skill:
                            prof = skill_prof.get(cand_skill, "beginner")
                            matched += prof_weights.get(prof, 0.2)
                            break
                skill_score = _clamp(matched / len(self.jd_skills))
            else:
                skill_score = 0.5

            # 2. Experience fit score (25%)
            yoe = profile.get("years_of_experience", 0)
            if yoe < self.jd_years_min:
                exp_score = _clamp(yoe / max(self.jd_years_min, 1))
            elif yoe > self.jd_years_max * 1.5:
                exp_score = 0.6  # Overqualified penalty
            else:
                exp_score = 1.0

            # 3. Education tier score (15%)
            tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}
            best_tier = max(
                (tier_map.get(e.get("tier", "unknown"), 0.4) for e in edu),
                default=0.4
            )
            edu_score = best_tier

            # 4. Certification bonus (10%)
            cert_score = _clamp(len(certs) / 5.0)

            # 5. Assessment score from Redrob platform (10%)
            rs = candidate.get("redrob_signals", {})
            assessment_scores = rs.get("skill_assessment_scores", {})
            if assessment_scores:
                relevant = []
                for jd_skill in self.jd_skills:
                    for k, v in assessment_scores.items():
                        if jd_skill in k.lower() or k.lower() in jd_skill:
                            relevant.append(v / 100.0)
                assess_score = sum(relevant) / len(relevant) if relevant else (
                    sum(assessment_scores.values()) / len(assessment_scores) / 100.0
                )
            else:
                assess_score = 0.4

            final = (
                0.40 * skill_score +
                0.25 * exp_score +
                0.15 * edu_score +
                0.10 * cert_score +
                0.10 * assess_score
            )
            return _clamp(final)

        except Exception as e:
            logger.warning(f"ProfileSignal error for {candidate.get('candidate_id')}: {e}")
            return 0.0


# ─────────────────────────────────────────────
# Signal B — Career Signal
# ─────────────────────────────────────────────

class CareerSignal:
    """
    Scores a candidate's career trajectory — progression, velocity, domain alignment.
    Returns float 0.0-1.0.
    """

    def __init__(self, jd_domain: str, jd_industry: str):
        self.jd_domain = jd_domain.lower()
        self.jd_industry = jd_industry.lower()

    def score(self, candidate: dict) -> float:
        """Compute career trajectory score."""
        try:
            career = candidate.get("career_history", [])
            if not career:
                return 0.2

            # 1. Upward trajectory — are titles getting more senior? (35%)
            seniority_keywords = {
                "intern": 1, "junior": 2, "associate": 3, "engineer": 4,
                "analyst": 4, "senior": 5, "lead": 6, "staff": 7,
                "principal": 8, "manager": 6, "director": 8, "vp": 9, "head": 8, "cto": 10
            }
            sorted_roles = sorted(career, key=lambda r: r.get("start_date", ""), reverse=False)
            levels = []
            for role in sorted_roles:
                title = role.get("title", "").lower()
                level = max(
                    (v for k, v in seniority_keywords.items() if k in title),
                    default=3
                )
                levels.append(level)
            if len(levels) >= 2:
                trajectory = (levels[-1] - levels[0]) / max(levels[0], 1)
                trajectory_score = _clamp(0.5 + trajectory * 0.25)
            else:
                trajectory_score = 0.5

            # 2. Velocity — how fast are they progressing? (25%)
            total_months = sum(r.get("duration_months", 0) for r in career)
            if total_months > 0 and len(levels) >= 2:
                months_per_level = total_months / max(len(levels) - 1, 1)
                # Sweet spot: 12-24 months per level = fast but stable
                if months_per_level < 6:
                    velocity_score = 0.5   # Too fast = job hopper
                elif months_per_level <= 24:
                    velocity_score = 1.0   # Ideal
                elif months_per_level <= 48:
                    velocity_score = 0.7   # Moderate
                else:
                    velocity_score = 0.4   # Slow progression
            else:
                velocity_score = 0.5

            # 3. Domain alignment (25%)
            domain_matches = 0
            for role in career:
                industry = role.get("industry", "").lower()
                desc = role.get("description", "").lower()
                if self.jd_domain in industry or self.jd_domain in desc:
                    domain_matches += 1
                elif self.jd_industry in industry:
                    domain_matches += 0.5
            domain_score = _clamp(domain_matches / max(len(career), 1))

            # 4. Company size progression — startup to scale or consistent (15%)
            size_map = {
                "1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
                "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8
            }
            sizes = [size_map.get(r.get("company_size", "1-10"), 1) for r in sorted_roles]
            if len(sizes) >= 2 and sizes[-1] >= sizes[0]:
                size_score = 0.8  # Growing into bigger companies
            else:
                size_score = 0.5

            final = (
                0.35 * trajectory_score +
                0.25 * velocity_score +
                0.25 * domain_score +
                0.15 * size_score
            )
            return _clamp(final)

        except Exception as e:
            logger.warning(f"CareerSignal error for {candidate.get('candidate_id')}: {e}")
            return 0.0


# ─────────────────────────────────────────────
# Signal C — Behavioral Signal (Redrob-specific)
# ─────────────────────────────────────────────

class BehavioralSignal:
    """
    Scores a candidate's behavioral and platform engagement signals.
    These signals predict hirability, not just skill fit.
    Returns float 0.0-1.0.
    """

    def score(self, candidate: dict) -> float:
        """Compute behavioral engagement score."""
        try:
            rs = candidate.get("redrob_signals", {})
            if not rs:
                return 0.3

            scores = []

            # 1. Availability signal (most important — 30%)
            avail = 0.0
            if rs.get("open_to_work_flag", False):
                avail += 0.5
            days_inactive = _days_since(rs.get("last_active_date"))
            if days_inactive <= 7:
                avail += 0.5
            elif days_inactive <= 30:
                avail += 0.35
            elif days_inactive <= 90:
                avail += 0.15
            # If inactive > 180 days, candidate is likely not looking
            avail_score = _clamp(avail)
            scores.append(("availability", avail_score, 0.30))

            # 2. Responsiveness signal (25%)
            resp_rate = rs.get("recruiter_response_rate", 0.0)
            resp_time = rs.get("avg_response_time_hours", 999)
            resp_time_score = _clamp(1.0 - (resp_time / 72.0))  # <72h is good
            resp_score = _clamp(0.7 * resp_rate + 0.3 * resp_time_score)
            scores.append(("responsiveness", resp_score, 0.25))

            # 3. Reliability signal (20%)
            interview_rate = rs.get("interview_completion_rate", 0.5)
            offer_rate = rs.get("offer_acceptance_rate", -1)
            if offer_rate == -1:
                offer_score = 0.5  # No history — neutral
            else:
                offer_score = _clamp(offer_rate)
            reliability = _clamp(0.6 * interview_rate + 0.4 * offer_score)
            scores.append(("reliability", reliability, 0.20))

            # 4. Market validation signal (15%)
            saved_30d = rs.get("saved_by_recruiters_30d", 0)
            views_30d = rs.get("profile_views_received_30d", 0)
            search_30d = rs.get("search_appearance_30d", 0)
            market_score = _clamp(
                _minmax(saved_30d, 0, 20) * 0.5 +
                _minmax(views_30d, 0, 100) * 0.3 +
                _minmax(search_30d, 0, 200) * 0.2
            )
            scores.append(("market_validation", market_score, 0.15))

            # 5. Profile quality signal (10%)
            completeness = rs.get("profile_completeness_score", 50) / 100.0
            verified = sum([
                rs.get("verified_email", False),
                rs.get("verified_phone", False),
                rs.get("linkedin_connected", False)
            ]) / 3.0
            github = rs.get("github_activity_score", -1)
            github_score = (github / 100.0) if github >= 0 else 0.3
            quality = _clamp(0.4 * completeness + 0.3 * verified + 0.3 * github_score)
            scores.append(("profile_quality", quality, 0.10))

            final = sum(s * w for _, s, w in scores)
            return _clamp(final)

        except Exception as e:
            logger.warning(f"BehavioralSignal error for {candidate.get('candidate_id')}: {e}")
            return 0.0
