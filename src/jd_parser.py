"""
Layer 1 — JD Intelligence Module
Parses a raw job description into structured signals using Gemini API.
Run during preprocessing only — not during ranking.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParsedJD(BaseModel):
    """Structured output from JD parsing."""
    role_title: str = Field(description="The exact job title")
    seniority: str = Field(description="junior/mid/senior/lead/staff/principal")
    domain: str = Field(description="Primary domain e.g. fintech/edtech/AI/ML/SaaS")
    hard_skills: list[str] = Field(description="Must-have technical skills")
    soft_signals: list[str] = Field(description="Implicit cultural/behavioral expectations")
    must_haves: list[str] = Field(description="Non-negotiable requirements")
    nice_to_haves: list[str] = Field(description="Bonus qualifications")
    years_experience_min: float = Field(description="Minimum years of experience required", default=0)
    years_experience_max: float = Field(description="Maximum years preferred", default=50)
    work_mode: str = Field(description="remote/hybrid/onsite/flexible/any", default="any")
    industry_context: str = Field(description="Industry the company operates in")
    key_phrases: list[str] = Field(description="Important phrases from JD for semantic matching")
    enriched_text: str = Field(description="A rewritten, enriched version of the JD for embedding")


def parse_jd(jd_text: str, api_key: Optional[str] = None) -> ParsedJD:
    """
    Parse a raw job description into structured intelligence using Gemini API.
    
    Args:
        jd_text: Raw job description text
        api_key: Optional Gemini API key (falls back to env var)
    
    Returns:
        ParsedJD structured object
    """

    # Use pre-parsed JD from cache if available and valid
    cache_path = Path(__file__).parent.parent / "cache" / "parsed_jd.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            if data.get("role_title", "Unknown Role") != "Unknown Role":
                logger.info(f"Using cached parsed JD: {data['role_title']}")
                return ParsedJD(**data)
        except Exception as e:
            logger.warning(f"Could not load cached JD: {e}")

    try:
        import google.generativeai as genai  # will migrate to google.genai soon

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""You are an expert technical recruiter. Analyze this job description deeply.

JOB DESCRIPTION:
{jd_text}

Extract structured intelligence. Return ONLY valid JSON, no markdown, no explanation.

{{
  "role_title": "exact job title",
  "seniority": "junior|mid|senior|lead|staff|principal",
  "domain": "primary domain e.g. ML/AI, Backend, Fintech",
  "hard_skills": ["skill1", "skill2", ...],
  "soft_signals": ["fast-paced startup tolerance", "ownership mindset", ...],
  "must_haves": ["non-negotiable requirement 1", ...],
  "nice_to_haves": ["bonus qualification 1", ...],
  "years_experience_min": 3,
  "years_experience_max": 10,
  "work_mode": "remote|hybrid|onsite|flexible|any",
  "industry_context": "e.g. B2B SaaS, Fintech, Edtech",
  "key_phrases": ["important phrase 1", "phrase 2", ...],
  "enriched_text": "A rich, detailed rewrite of this JD expanding all implied requirements for semantic embedding. At least 200 words."
}}"""

        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Clean markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        parsed = ParsedJD(**data)
        logger.info(f"JD parsed: role={parsed.role_title}, seniority={parsed.seniority}, skills={len(parsed.hard_skills)}")
        return parsed

    except Exception as e:
        logger.error(f"JD parsing failed: {e}")
        # Return a minimal fallback so pipeline doesn't crash
        return ParsedJD(
            role_title="Unknown Role",
            seniority="mid",
            domain="Technology",
            hard_skills=[],
            soft_signals=[],
            must_haves=[],
            nice_to_haves=[],
            years_experience_min=0,
            years_experience_max=50,
            work_mode="any",
            industry_context="Technology",
            key_phrases=[],
            enriched_text=jd_text
        )


def load_jd(jd_path: Path) -> str:
    """Load JD text from file."""
    with open(jd_path, "r", encoding="utf-8") as f:
        return f.read()


def save_parsed_jd(parsed: ParsedJD, output_path: Path) -> None:
    """Save parsed JD to JSON for reuse during ranking."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed.model_dump(), f, indent=2)
    logger.info(f"Parsed JD saved to {output_path}")


def load_parsed_jd(path: Path) -> ParsedJD:
    """Load previously parsed JD from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ParsedJD(**data)
