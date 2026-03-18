"""
MoSSGA — Multi-Modal Fusion Engine
====================================
Combines resume-extracted data with GitHub profile analysis into a
unified candidate profile. Skills validated by both sources receive
higher confidence scores.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fusion Configuration
# ---------------------------------------------------------------------------

DEFAULT_FUSION_WEIGHTS = {
    "resume": 0.60,
    "github": 0.40,
}


# ---------------------------------------------------------------------------
# Fusion Engine
# ---------------------------------------------------------------------------

def fuse_skill_profiles(
    resume_skills: list,
    github_skills: list,
    resume_weight: float = 0.60,
    github_weight: float = 0.40,
) -> dict:
    """
    Fuses skills from resume and GitHub into a single unified profile.

    Skills found in both sources get high confidence; single-source
    skills get lower confidence. Deduplication is case-insensitive.

    Args:
        resume_skills: Skills extracted from the candidate's resume.
        github_skills: Skills inferred from GitHub profile analysis.
        resume_weight: Weight for resume-sourced skills (0-1).
        github_weight: Weight for GitHub-sourced skills (0-1).

    Returns:
        {
            "fused_skills": [list of all unique skills],
            "skill_sources": {skill: ["resume", "github"] or ["resume"] or ["github"]},
            "confidence_scores": {skill: float (0-1)},
            "validation_summary": {
                "both_sources": [...],
                "resume_only": [...],
                "github_only": [...],
            },
            "fusion_stats": {
                "total_unique": int,
                "validated_count": int,
                "validation_rate": float,
            },
        }
    """
    # Normalise to lower case for comparison
    resume_set = {s.lower().strip() for s in resume_skills if s.strip()}
    github_set = {s.lower().strip() for s in github_skills if s.strip()}

    # Classify by source
    both_sources = resume_set & github_set
    resume_only = resume_set - github_set
    github_only = github_set - resume_set
    all_skills = resume_set | github_set

    # Calculate confidence per skill
    confidence = {}
    sources = {}

    for skill in both_sources:
        # Validated by both → high confidence
        confidence[skill] = round(min(resume_weight + github_weight, 1.0), 2)
        sources[skill] = ["resume", "github"]

    for skill in resume_only:
        confidence[skill] = round(resume_weight * 0.8, 2)  # slight penalty for single source
        sources[skill] = ["resume"]

    for skill in github_only:
        confidence[skill] = round(github_weight * 0.75, 2)  # GitHub-only slightly less trusted
        sources[skill] = ["github"]

    # Sort by confidence
    fused_list = sorted(all_skills, key=lambda s: -confidence[s])

    validated_count = len(both_sources)
    total = len(all_skills) if all_skills else 1

    return {
        "fused_skills": fused_list,
        "skill_sources": sources,
        "confidence_scores": confidence,
        "validation_summary": {
            "both_sources": sorted(both_sources),
            "resume_only": sorted(resume_only),
            "github_only": sorted(github_only),
        },
        "fusion_stats": {
            "total_unique": len(all_skills),
            "validated_count": validated_count,
            "validation_rate": round(validated_count / total * 100, 1),
            "resume_skill_count": len(resume_set),
            "github_skill_count": len(github_set),
        },
    }


def generate_fusion_summary(fusion_result: dict) -> str:
    """Generates a human-readable summary of the fusion analysis."""
    stats = fusion_result["fusion_stats"]
    val = fusion_result["validation_summary"]

    both = len(val["both_sources"])
    r_only = len(val["resume_only"])
    g_only = len(val["github_only"])

    summary_parts = [
        f"📊 **Multi-Modal Fusion Summary:**\n",
        f"- **{stats['total_unique']}** unique skills identified across both sources",
        f"- **{both}** skills validated by both resume and GitHub (highest confidence)",
    ]

    if r_only:
        summary_parts.append(
            f"- **{r_only}** skills found only in resume — consider adding GitHub projects to validate them"
        )
    if g_only:
        summary_parts.append(
            f"- **{g_only}** skills found only on GitHub — consider adding them to your resume"
        )

    summary_parts.append(
        f"- **Validation rate:** {stats['validation_rate']}% of skills confirmed by multiple sources"
    )

    # Overall assessment
    rate = stats["validation_rate"]
    if rate >= 50:
        summary_parts.append(
            "\n✅ **Strong cross-validation** — Your resume and GitHub profile are well-aligned."
        )
    elif rate >= 25:
        summary_parts.append(
            "\n🟡 **Moderate alignment** — Some skills lack cross-validation. "
            "Add GitHub projects for resume-only skills, or update your resume to include GitHub technologies."
        )
    else:
        summary_parts.append(
            "\n🔴 **Low alignment** — Your resume and GitHub profile show very different skill sets. "
            "This may raise authenticity concerns with recruiters."
        )

    return "\n".join(summary_parts)


def compute_weighted_skill_score(
    fusion_result: dict,
    match_result: dict,
) -> float:
    """
    Computes an overall MoSSGA skill score that factors in multi-modal confidence.

    Skills matched semantically AND validated by both sources score higher
    than single-source or keyword-only matches.

    Args:
        fusion_result: Output from fuse_skill_profiles().
        match_result: Output from semantic_skill_match().

    Returns:
        Weighted skill score as a percentage (0-100).
    """
    matched = match_result.get("matched_skills", [])
    jd_total = len(match_result.get("jd_resolved", []))

    if jd_total == 0:
        return 0.0

    confidence_scores = fusion_result.get("confidence_scores", {})
    weighted_sum = 0.0

    for m in matched:
        cand_skill = m["candidate_skill"].lower()
        sim = m["similarity"]
        conf = confidence_scores.get(cand_skill, 0.5)

        # Weighted contribution: similarity × confidence
        weighted_sum += sim * conf

    # Normalise by JD count
    score = (weighted_sum / jd_total) * 100
    return round(min(score, 100.0), 1)
