"""
MoSSGA — Semantic Skill Matcher
================================
Replaces keyword-only skill matching with SBERT-powered semantic matching.
Resolves abbreviations/synonyms (e.g., "ML" ↔ "Machine Learning") and
computes cosine similarity between candidate skills and JD requirements.
"""
from __future__ import annotations


import logging
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill Alias / Synonym Taxonomy
# ---------------------------------------------------------------------------
# Maps alternative names → canonical skill name. The canonical name is always
# the most commonly written, full-length form of the skill.

SKILL_ALIASES = {
    # Programming languages
    "js": "javascript", "es6": "javascript", "ecmascript": "javascript",
    "ts": "typescript",
    "py": "python", "python3": "python", "python2": "python",
    "cpp": "c++", "c plus plus": "c++",
    "csharp": "c#", "c sharp": "c#",
    "golang": "go",
    "rb": "ruby",

    # ML / AI
    "ml": "machine learning", "machine-learning": "machine learning",
    "dl": "deep learning", "deep-learning": "deep learning",
    "ai": "artificial intelligence",
    "natural language processing": "nlp",
    "natural-language-processing": "nlp",
    "cv": "computer vision", "computer-vision": "computer vision",
    "gen ai": "generative ai", "genai": "generative ai",
    "llm": "large language models", "llms": "large language models",
    "transformers": "transformers",

    # Frameworks
    "tf": "tensorflow", "tensor flow": "tensorflow",
    "torch": "pytorch",
    "sklearn": "scikit-learn", "sk-learn": "scikit-learn",
    "keras": "keras",
    "reactjs": "react", "react.js": "react",
    "nextjs": "next.js", "next js": "next.js",
    "vuejs": "vue", "vue.js": "vue",
    "angularjs": "angular", "angular.js": "angular",
    "expressjs": "express", "express.js": "express",
    "nodejs": "node.js", "node js": "node.js", "node": "node.js",

    # Data / DB
    "postgres": "postgresql", "pg": "postgresql",
    "mysql": "mysql",
    "mongo": "mongodb", "mongo db": "mongodb",
    "nosql": "nosql",
    "structured query language": "sql",
    "pandas": "pandas", "numpy": "numpy",

    # DevOps / Cloud
    "k8s": "kubernetes", "kube": "kubernetes",
    "ci cd": "ci/cd", "cicd": "ci/cd", "continuous integration": "ci/cd",
    "amazon web services": "aws", "amazon aws": "aws",
    "google cloud": "gcp", "google cloud platform": "gcp",
    "microsoft azure": "azure",
    "infrastructure as code": "terraform",
    "containerization": "docker",

    # Tools & practices
    "github": "git", "gitlab": "git", "version control": "git",
    "rest": "rest apis", "restful": "rest apis", "rest api": "rest apis",
    "api development": "rest apis",
    "mlops": "mlops", "ml ops": "mlops",
    "data viz": "data visualization", "data visualisation": "data visualization",
    "powerbi": "power bi", "power-bi": "power bi",

    # Soft skills
    "team work": "teamwork",
    "project mgmt": "project management",
    "pm": "project management",
    "communication skills": "communication",
}

# Build reverse map: canonical → all aliases (including itself)
CANONICAL_SKILLS = set(SKILL_ALIASES.values())


# ---------------------------------------------------------------------------
# Semantic Matching with SBERT
# ---------------------------------------------------------------------------

def _get_sbert_model():
    """Returns cached SBERT model from matcher module."""
    from src.matcher import get_sbert_model
    return get_sbert_model()


def resolve_skill_aliases(skills: list) -> list:
    """Resolves aliases/abbreviations to canonical skill names."""
    resolved = set()
    for skill in skills:
        skill_lower = skill.strip().lower()
        canonical = SKILL_ALIASES.get(skill_lower, skill_lower)
        resolved.add(canonical)
    return list(resolved)


def compute_skill_embeddings(skills: list) -> np.ndarray | None:
    """Encodes a list of skill names into SBERT vectors."""
    model = _get_sbert_model()
    if model is None:
        return None

    # Prefix each skill with context to improve embedding quality
    contextualised = [f"technical skill: {s}" for s in skills]
    embeddings = model.encode(contextualised, show_progress_bar=False)
    return embeddings


def semantic_skill_match(
    candidate_skills: list,
    jd_skills: list,
    threshold: float = 0.55,
) -> dict:
    """
    Performs semantic skill matching between candidate and JD skills.

    Instead of exact keyword overlap, uses SBERT cosine similarity to
    recognize that "ML" ≈ "Machine Learning", "React" ≈ "ReactJS", etc.

    Args:
        candidate_skills: Skills extracted from candidate's profile.
        jd_skills:        Skills required by the job description.
        threshold:        Minimum cosine similarity to consider a match.

    Returns:
        {
            "matched_skills":     [{jd_skill, candidate_skill, similarity}, ...],
            "missing_skills":     [skills in JD with no semantic match],
            "match_percentage":   float (0-100),
            "similarity_matrix":  2D list of similarities (candidate × JD),
            "candidate_resolved": canonical candidate skills,
            "jd_resolved":        canonical JD skills,
        }
    """
    # Step 1: Resolve aliases
    cand_resolved = resolve_skill_aliases(candidate_skills)
    jd_resolved = resolve_skill_aliases(jd_skills)

    if not cand_resolved or not jd_resolved:
        return {
            "matched_skills": [],
            "missing_skills": jd_resolved,
            "match_percentage": 0.0,
            "similarity_matrix": [],
            "candidate_resolved": cand_resolved,
            "jd_resolved": jd_resolved,
        }

    # Step 2: Quick exact match first (case-insensitive)
    cand_lower_set = {s.lower() for s in cand_resolved}
    exact_matches = []
    remaining_jd = []

    for jd_skill in jd_resolved:
        if jd_skill.lower() in cand_lower_set:
            exact_matches.append({
                "jd_skill": jd_skill,
                "candidate_skill": jd_skill,
                "similarity": 1.0,
                "match_type": "exact",
            })
        else:
            remaining_jd.append(jd_skill)

    # Step 3: Semantic matching for remaining skills
    semantic_matches = []
    still_missing = []

    if remaining_jd and cand_resolved:
        cand_embeddings = compute_skill_embeddings(cand_resolved)
        jd_embeddings = compute_skill_embeddings(remaining_jd)

        if cand_embeddings is not None and jd_embeddings is not None:
            sim_matrix = cosine_similarity(jd_embeddings, cand_embeddings)

            for i, jd_skill in enumerate(remaining_jd):
                best_idx = np.argmax(sim_matrix[i])
                best_sim = float(sim_matrix[i][best_idx])

                if best_sim >= threshold:
                    semantic_matches.append({
                        "jd_skill": jd_skill,
                        "candidate_skill": cand_resolved[best_idx],
                        "similarity": round(best_sim, 3),
                        "match_type": "semantic",
                    })
                else:
                    still_missing.append(jd_skill)
        else:
            # SBERT unavailable — fall back to pure exact matching
            still_missing = remaining_jd
    else:
        still_missing = remaining_jd

    # Step 4: Build full similarity matrix for visualisation
    full_sim_matrix = []
    all_cand_embeddings = compute_skill_embeddings(cand_resolved)
    all_jd_embeddings = compute_skill_embeddings(jd_resolved)

    if all_cand_embeddings is not None and all_jd_embeddings is not None:
        full_sim = cosine_similarity(all_jd_embeddings, all_cand_embeddings)
        full_sim_matrix = [[round(float(v), 3) for v in row] for row in full_sim]

    # Combine matches
    all_matches = exact_matches + semantic_matches
    match_pct = len(all_matches) / len(jd_resolved) * 100 if jd_resolved else 0

    return {
        "matched_skills": all_matches,
        "missing_skills": still_missing,
        "match_percentage": round(match_pct, 1),
        "similarity_matrix": full_sim_matrix,
        "candidate_resolved": cand_resolved,
        "jd_resolved": jd_resolved,
    }


def calculate_semantic_gap_score(match_result: dict) -> dict:
    """
    Computes a skill gap score with severity levels for each missing skill.

    Returns:
        {
            "gap_score": float (0-100, 0 = no gap),
            "gap_severity": "Low" | "Medium" | "High" | "Critical",
            "missing_with_severity": [
                {"skill": str, "importance": float, "severity": str},
            ],
        }
    """
    total_jd = len(match_result["jd_resolved"])
    missing = match_result["missing_skills"]

    if total_jd == 0:
        return {
            "gap_score": 0,
            "gap_severity": "Low",
            "missing_with_severity": [],
        }

    gap_ratio = len(missing) / total_jd
    gap_score = round(gap_ratio * 100, 1)

    # Classify severity
    if gap_ratio <= 0.15:
        severity = "Low"
    elif gap_ratio <= 0.35:
        severity = "Medium"
    elif gap_ratio <= 0.60:
        severity = "High"
    else:
        severity = "Critical"

    # Score importance per missing skill (heuristic: earlier in JD = more important)
    jd_skills = match_result["jd_resolved"]
    missing_detailed = []
    for skill in missing:
        # Higher importance for skills appearing earlier in JD
        idx = jd_skills.index(skill) if skill in jd_skills else len(jd_skills)
        importance = round(max(0.3, 1.0 - idx * 0.08), 2)

        if importance >= 0.7:
            skill_sev = "Critical"
        elif importance >= 0.5:
            skill_sev = "Important"
        else:
            skill_sev = "Nice-to-have"

        missing_detailed.append({
            "skill": skill,
            "importance": importance,
            "severity": skill_sev,
        })

    missing_detailed.sort(key=lambda x: -x["importance"])

    return {
        "gap_score": gap_score,
        "gap_severity": severity,
        "missing_with_severity": missing_detailed,
    }
