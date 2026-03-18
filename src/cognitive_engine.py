"""
NeuroHire Cognitive Engine — Research-Level AI Modules
=======================================================
Novel contributions:
  1. Cognitive Skill Graph — Skill dependency mapping & next-best-skill prediction
  2. Career Trajectory Prediction — ML-based timeline modeling
  3. Skill Authenticity Detection — Heuristic validation of claimed skills
  4. Enhanced Explainable AI — Factor-by-factor score reasoning
"""

import re
import random
import math
from collections import defaultdict


# ============================================================================
# 1. COGNITIVE SKILL GRAPH — Novel Research Contribution
# ============================================================================

# Directed graph: skill -> [skills it unlocks / leads to]
SKILL_DEPENDENCY_GRAPH = {
    # Programming foundations
    "python": ["pandas", "numpy", "flask", "django", "fastapi", "machine learning", "data visualization"],
    "javascript": ["react", "node.js", "typescript", "express", "vue", "angular"],
    "c++": ["data structures", "algorithms", "system design", "competitive programming"],
    "java": ["spring boot", "android", "microservices", "system design"],
    "html": ["css", "javascript", "responsive design"],
    "css": ["tailwind css", "responsive design", "ui/ux design"],

    # Data Science path
    "pandas": ["data visualization", "feature engineering", "data analysis"],
    "numpy": ["scikit-learn", "tensorflow", "pytorch"],
    "statistics": ["machine learning", "a/b testing", "data analysis"],
    "machine learning": ["deep learning", "nlp", "computer vision", "mlops"],
    "deep learning": ["tensorflow", "pytorch", "nlp", "computer vision", "generative ai"],
    "nlp": ["transformers", "llm fine-tuning", "text mining"],
    "scikit-learn": ["machine learning", "feature engineering", "model evaluation"],
    "tensorflow": ["keras", "model deployment", "mlops"],
    "pytorch": ["model deployment", "research", "mlops"],

    # Web Development path
    "react": ["next.js", "redux", "typescript", "testing"],
    "node.js": ["express", "rest apis", "graphql", "microservices"],
    "typescript": ["angular", "next.js", "type-safe apis"],
    "express": ["rest apis", "middleware", "authentication"],
    "mongodb": ["mongoose", "database design", "nosql"],
    "sql": ["postgresql", "database design", "data modeling", "query optimization"],

    # DevOps/Cloud path
    "git": ["ci/cd", "github actions", "collaboration"],
    "docker": ["kubernetes", "ci/cd", "microservices", "cloud deployment"],
    "kubernetes": ["helm", "service mesh", "cloud architecture"],
    "linux": ["bash scripting", "docker", "devops", "system administration"],
    "aws": ["cloud architecture", "serverless", "devops"],

    # Soft skills
    "communication": ["leadership", "project management", "stakeholder management"],
    "agile": ["scrum", "kanban", "project management"],
    "teamwork": ["leadership", "mentoring", "cross-functional collaboration"],
}

# Reverse mapping: what skills are prerequisites for this skill
SKILL_PREREQUISITES = defaultdict(list)
for parent, children in SKILL_DEPENDENCY_GRAPH.items():
    for child in children:
        SKILL_PREREQUISITES[child.lower()].append(parent.lower())

# Skill complexity weights (higher = more advanced)
SKILL_COMPLEXITY = {
    "html": 1, "css": 1, "git": 1, "excel": 1,
    "python": 2, "javascript": 2, "java": 2, "c++": 2, "sql": 2,
    "react": 3, "node.js": 3, "pandas": 3, "numpy": 3, "flask": 3, "django": 3,
    "typescript": 3, "mongodb": 3, "postgresql": 3, "rest apis": 3,
    "machine learning": 4, "docker": 4, "scikit-learn": 4, "express": 4,
    "deep learning": 5, "nlp": 5, "tensorflow": 5, "pytorch": 5,
    "kubernetes": 5, "system design": 5, "microservices": 5,
    "mlops": 6, "cloud architecture": 6, "distributed systems": 6,
    "generative ai": 7, "llm fine-tuning": 7, "transformers": 7,
}


def build_cognitive_skill_graph(candidate_skills):
    """
    Build a cognitive skill graph showing:
    - Current skills and their relationships
    - Predicted next-best skills to learn
    - Skill clusters and progression paths

    Returns: dict with nodes, edges, next_best_skills, skill_clusters
    """
    skills_lower = {s.lower() for s in candidate_skills}

    # Build nodes (current skills + recommended next)
    nodes = []
    edges = []
    next_best = []

    # Current skills as nodes
    for skill in skills_lower:
        complexity = SKILL_COMPLEXITY.get(skill, 2)
        nodes.append({
            "id": skill,
            "label": skill.title(),
            "status": "current",
            "complexity": complexity,
            "size": 15 + complexity * 5,
        })

        # Find what this skill unlocks
        unlocked = SKILL_DEPENDENCY_GRAPH.get(skill, [])
        for target in unlocked:
            target_lower = target.lower()
            if target_lower in skills_lower:
                # Both known — show mastered connection
                edges.append({"from": skill, "to": target_lower, "type": "mastered"})
            else:
                # Target is a potential next skill
                edges.append({"from": skill, "to": target_lower, "type": "potential"})

                # Score this as a next-best-skill candidate
                prereqs = SKILL_PREREQUISITES.get(target_lower, [])
                prereqs_met = sum(1 for p in prereqs if p in skills_lower)
                prereq_ratio = prereqs_met / len(prereqs) if prereqs else 0.5
                target_complexity = SKILL_COMPLEXITY.get(target_lower, 3)

                # Readiness score: how ready the candidate is to learn this
                readiness = prereq_ratio * 100

                next_best.append({
                    "skill": target.title(),
                    "readiness_score": round(readiness, 1),
                    "complexity": target_complexity,
                    "prerequisites_met": f"{prereqs_met}/{len(prereqs)}",
                    "unlocked_by": [s.title() for s in prereqs if s in skills_lower],
                    "reason": _build_next_skill_reason(target, prereqs, skills_lower, readiness),
                })

    # Deduplicate and sort next_best by readiness
    seen = set()
    unique_next = []
    for nb in sorted(next_best, key=lambda x: x["readiness_score"], reverse=True):
        if nb["skill"].lower() not in seen:
            seen.add(nb["skill"].lower())
            unique_next.append(nb)

    # Identify skill clusters
    clusters = _identify_skill_clusters(skills_lower)

    return {
        "nodes": nodes,
        "edges": edges,
        "next_best_skills": unique_next[:8],
        "skill_clusters": clusters,
        "graph_density": round(len(edges) / max(len(nodes), 1), 2),
        "avg_complexity": round(sum(SKILL_COMPLEXITY.get(s, 2) for s in skills_lower) / max(len(skills_lower), 1), 1),
    }


def _build_next_skill_reason(skill, prereqs, current_skills, readiness):
    """Build XAI reason for why this skill is recommended next."""
    met = [p for p in prereqs if p in current_skills]
    if readiness >= 80:
        return f"You have {len(met)} prerequisites mastered. You're highly ready to learn {skill.title()}."
    elif readiness >= 50:
        return f"Good foundation. Learning {skill.title()} would expand your capability significantly."
    else:
        return f"Emerging pathway. {skill.title()} would open new career directions for you."


def _identify_skill_clusters(skills):
    """Groups skills into domain clusters."""
    clusters = {
        "🌐 Web Development": {"html", "css", "javascript", "react", "node.js", "typescript",
                                "express", "vue", "angular", "next.js", "tailwind css"},
        "🧠 AI / ML": {"machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
                        "scikit-learn", "keras", "computer vision", "generative ai"},
        "📊 Data": {"python", "pandas", "numpy", "sql", "mongodb", "postgresql",
                     "data visualization", "statistics", "excel", "tableau", "power bi"},
        "☁️ DevOps / Cloud": {"docker", "kubernetes", "aws", "azure", "gcp", "linux",
                               "ci/cd", "terraform", "git"},
        "🏗️ Architecture": {"system design", "microservices", "rest apis", "graphql",
                             "distributed systems"},
        "🤝 Professional": {"communication", "teamwork", "leadership", "agile", "scrum",
                             "project management"},
    }

    active_clusters = {}
    for cluster_name, cluster_skills in clusters.items():
        matched = skills & cluster_skills
        if matched:
            active_clusters[cluster_name] = {
                "skills": [s.title() for s in matched],
                "coverage": f"{len(matched)}/{len(cluster_skills)}",
                "percentage": round(len(matched) / len(cluster_skills) * 100, 1),
            }

    return active_clusters


# ============================================================================
# 2. CAREER TRAJECTORY PREDICTION — Novel Research Contribution
# ============================================================================

CAREER_TRAJECTORIES = {
    "data scientist": {
        "milestones": [
            {"role": "Data Analyst", "months": 0, "skills_needed": ["sql", "python", "data visualization"]},
            {"role": "Junior Data Scientist", "months": 4, "skills_needed": ["machine learning", "statistics", "scikit-learn"]},
            {"role": "Data Scientist", "months": 10, "skills_needed": ["deep learning", "nlp", "feature engineering"]},
            {"role": "Senior Data Scientist", "months": 20, "skills_needed": ["mlops", "system design", "leadership"]},
            {"role": "ML Engineering Lead", "months": 36, "skills_needed": ["cloud architecture", "mentoring", "stakeholder management"]},
        ],
    },
    "web developer": {
        "milestones": [
            {"role": "Junior Frontend Dev", "months": 0, "skills_needed": ["html", "css", "javascript"]},
            {"role": "Frontend Developer", "months": 3, "skills_needed": ["react", "typescript", "rest apis"]},
            {"role": "Full Stack Developer", "months": 8, "skills_needed": ["node.js", "mongodb", "docker"]},
            {"role": "Senior Full Stack", "months": 18, "skills_needed": ["system design", "ci/cd", "microservices"]},
            {"role": "Tech Lead / Architect", "months": 30, "skills_needed": ["cloud architecture", "leadership", "mentoring"]},
        ],
    },
    "software engineer": {
        "milestones": [
            {"role": "Junior Software Engineer", "months": 0, "skills_needed": ["programming", "git", "data structures"]},
            {"role": "Software Engineer", "months": 6, "skills_needed": ["system design", "databases", "testing"]},
            {"role": "Senior Software Engineer", "months": 18, "skills_needed": ["distributed systems", "cloud", "mentoring"]},
            {"role": "Staff Engineer", "months": 36, "skills_needed": ["architecture", "cross-team leadership", "technical strategy"]},
        ],
    },
    "ml engineer": {
        "milestones": [
            {"role": "Junior ML Engineer", "months": 0, "skills_needed": ["python", "machine learning", "sql"]},
            {"role": "ML Engineer", "months": 6, "skills_needed": ["tensorflow", "docker", "model deployment"]},
            {"role": "Senior ML Engineer", "months": 16, "skills_needed": ["mlops", "kubernetes", "system design"]},
            {"role": "Principal ML Engineer", "months": 30, "skills_needed": ["research", "architecture", "leadership"]},
        ],
    },
    "devops engineer": {
        "milestones": [
            {"role": "Junior DevOps", "months": 0, "skills_needed": ["linux", "git", "docker"]},
            {"role": "DevOps Engineer", "months": 5, "skills_needed": ["kubernetes", "ci/cd", "terraform"]},
            {"role": "Senior DevOps", "months": 14, "skills_needed": ["cloud architecture", "monitoring", "security"]},
            {"role": "Platform Lead", "months": 28, "skills_needed": ["sre", "strategy", "leadership"]},
        ],
    },
}


def predict_career_trajectory(skills, target_role):
    """
    Predicts the candidate's career trajectory with timeline.
    Returns milestones with estimated months and readiness percentages.
    """
    skills_lower = {s.lower() for s in skills}
    role_lower = target_role.lower() if target_role else ""

    # Match to best trajectory
    best_trajectory = None
    for traj_name, traj_data in CAREER_TRAJECTORIES.items():
        if role_lower and role_lower in traj_name:
            best_trajectory = (traj_name, traj_data)
            break

    if not best_trajectory:
        # Infer from skills
        best_score = 0
        for traj_name, traj_data in CAREER_TRAJECTORIES.items():
            all_needed = set()
            for ms in traj_data["milestones"]:
                all_needed.update(s.lower() for s in ms["skills_needed"])
            score = len(skills_lower & all_needed)
            if score > best_score:
                best_score = score
                best_trajectory = (traj_name, traj_data)

    if not best_trajectory:
        best_trajectory = ("software engineer", CAREER_TRAJECTORIES["software engineer"])

    traj_name, traj_data = best_trajectory

    # Calculate readiness for each milestone
    trajectory_output = []
    current_position = 0

    for i, milestone in enumerate(traj_data["milestones"]):
        needed = set(s.lower() for s in milestone["skills_needed"])
        have = skills_lower & needed
        readiness = round(len(have) / max(len(needed), 1) * 100, 1)

        if readiness >= 70:
            current_position = i
            status = "✅ Ready"
        elif readiness >= 40:
            status = "🟡 Partially Ready"
        else:
            status = "🔴 Skills Gap"

        # Estimate actual months considering current readiness
        base_months = milestone["months"]
        adjustment = max(0, (100 - readiness) / 100 * 3)  # Extra months for gap
        estimated_months = round(base_months + adjustment)

        trajectory_output.append({
            "role": milestone["role"],
            "base_months": base_months,
            "estimated_months": estimated_months,
            "readiness": readiness,
            "status": status,
            "skills_needed": milestone["skills_needed"],
            "skills_you_have": [s.title() for s in have],
            "skills_to_learn": [s.title() for s in (needed - have)],
        })

    return {
        "trajectory_name": traj_name.title(),
        "current_position": current_position,
        "milestones": trajectory_output,
        "prediction": _build_trajectory_prediction(trajectory_output, current_position, traj_name),
    }


def _build_trajectory_prediction(milestones, current_pos, traj_name):
    """Builds a natural language prediction."""
    if current_pos >= len(milestones) - 1:
        return f"🎯 You are well-positioned for senior {traj_name.title()} roles. Focus on leadership and architectural skills."

    next_milestone = milestones[current_pos + 1]
    gap_skills = next_milestone["skills_to_learn"]

    if gap_skills:
        return (
            f"📈 Your next milestone: **{next_milestone['role']}** "
            f"(estimated ~{next_milestone['estimated_months']} months). "
            f"Bridge the gap by mastering: {', '.join(gap_skills)}."
        )
    else:
        return f"📈 You're ready for **{next_milestone['role']}**! Consider applying now."


# ============================================================================
# 3. SKILL AUTHENTICITY DETECTION — Novel Research Contribution
# ============================================================================

def detect_skill_authenticity(resume_text, claimed_skills):
    """
    Heuristic-based skill authenticity scoring.
    Checks if claimed skills have supporting evidence in resume text.

    For each skill, looks for:
    - Project mentions using the skill
    - Experience descriptions mentioning the skill
    - Certifications related to the skill
    - Quantifiable metrics tied to the skill

    Returns: authenticity report per skill.
    """
    text_lower = resume_text.lower()
    report = []

    # Evidence patterns
    project_patterns = [
        r"(?:built|developed|created|designed|implemented|deployed)\s+.*?{skill}",
        r"{skill}\s+.*?(?:project|application|system|tool|platform)",
    ]
    experience_patterns = [
        r"(?:worked|experience|used|utilized|leveraged)\s+.*?{skill}",
        r"{skill}\s+.*?(?:years?|months?|professional)",
    ]
    cert_patterns = [
        r"(?:certified|certification|certificate)\s+.*?{skill}",
        r"{skill}\s+.*?(?:certified|certification|certificate)",
    ]
    metric_patterns = [
        r"{skill}\s+.*?\d+\s*%",
        r"\d+\s*(?:users?|clients?|projects?).*?{skill}",
    ]

    for skill in claimed_skills:
        skill_lower = skill.lower()
        evidence = {
            "project_evidence": False,
            "experience_evidence": False,
            "certification_evidence": False,
            "metric_evidence": False,
        }

        # Check each evidence type
        for pattern in project_patterns:
            if re.search(pattern.format(skill=re.escape(skill_lower)), text_lower):
                evidence["project_evidence"] = True
                break

        for pattern in experience_patterns:
            if re.search(pattern.format(skill=re.escape(skill_lower)), text_lower):
                evidence["experience_evidence"] = True
                break

        for pattern in cert_patterns:
            if re.search(pattern.format(skill=re.escape(skill_lower)), text_lower):
                evidence["certification_evidence"] = True
                break

        for pattern in metric_patterns:
            if re.search(pattern.format(skill=re.escape(skill_lower)), text_lower):
                evidence["metric_evidence"] = True
                break

        # Simple mention check
        mentioned = skill_lower in text_lower

        # Calculate authenticity score
        evidence_count = sum(evidence.values())
        if evidence_count >= 3:
            score = 95
            verdict = "✅ Strongly Verified"
        elif evidence_count >= 2:
            score = 80
            verdict = "✅ Verified"
        elif evidence_count >= 1:
            score = 60
            verdict = "🟡 Partially Verified"
        elif mentioned:
            score = 40
            verdict = "🟡 Mentioned Only"
        else:
            score = 15
            verdict = "🔴 Unverified"

        report.append({
            "skill": skill,
            "authenticity_score": score,
            "verdict": verdict,
            "evidence": evidence,
            "recommendation": _auth_recommendation(skill, evidence, score),
        })

    # Overall authenticity
    avg_score = round(sum(r["authenticity_score"] for r in report) / max(len(report), 1), 1)

    return {
        "overall_authenticity": avg_score,
        "skill_reports": report,
        "summary": _auth_summary(avg_score, report),
    }


def _auth_recommendation(skill, evidence, score):
    """XAI: explains why a skill got its authenticity score."""
    if score >= 80:
        return f"{skill} is well-supported with project/experience evidence in your resume."
    elif score >= 40:
        tips = []
        if not evidence["project_evidence"]:
            tips.append(f"Add a project that uses {skill}")
        if not evidence["metric_evidence"]:
            tips.append("Include quantifiable results")
        return f"Strengthen {skill} by: {'; '.join(tips)}."
    else:
        return f"Add concrete evidence for {skill}: projects, experience descriptions, or certifications."


def _auth_summary(avg_score, reports):
    """Builds overall authenticity summary."""
    verified = sum(1 for r in reports if r["authenticity_score"] >= 60)
    total = len(reports)
    if avg_score >= 70:
        return f"✅ Strong profile authenticity ({verified}/{total} skills verified). Resume evidence supports claimed skills."
    elif avg_score >= 45:
        return f"🟡 Moderate authenticity ({verified}/{total} skills verified). Add project descriptions and metrics to strengthen claims."
    else:
        return f"🔴 Low authenticity ({verified}/{total} skills verified). Most skills lack supporting evidence — add projects and detailed experience."


# ============================================================================
# 4. ENHANCED EXPLAINABLE AI — Deep Score Reasoning
# ============================================================================

def generate_deep_explanation(candidate_data, jd_skills, alpha):
    """
    Generates a comprehensive, factor-by-factor explanation of why
    a candidate received their RJAS score.
    """
    breakdown = candidate_data.get("Breakdown", {})
    rjas = candidate_data.get("RJAS", 0)
    matched = candidate_data.get("Matched Skills", [])
    missing = candidate_data.get("Missing Skills", [])
    bias = candidate_data.get("Detected Bias", [])
    sbert = candidate_data.get("SBERT Score", 0)
    nlp = candidate_data.get("NLP Score", 0)

    factors = []

    # Semantic factor
    sem_score = breakdown.get("Semantic", 0)
    if sem_score > 70:
        factors.append(f"🟢 **Semantic Match ({sem_score}%):** Your resume's meaning strongly aligns with the JD. SBERT detected deep contextual overlap beyond just keywords.")
    elif sem_score > 40:
        factors.append(f"🟡 **Semantic Match ({sem_score}%):** Moderate alignment. Your resume covers some themes but misses core JD concepts. Rewrite your summary to mirror the JD's language.")
    else:
        factors.append(f"🔴 **Semantic Match ({sem_score}%):** Weak alignment. Your resume talks about different topics than the JD. Tailor your resume's objective and descriptions.")

    # Skills factor
    skill_score = breakdown.get("Skills", 0)
    if len(matched) > 0:
        factors.append(f"🛠️ **Skill Match ({skill_score}%):** Matched {len(matched)}/{len(jd_skills)} JD skills: {', '.join(matched[:5])}.")
    else:
        factors.append(f"🛠️ **Skill Match ({skill_score}%):** No direct skill overlap detected with the JD requirements.")

    if missing:
        factors.append(f"❌ **Missing Skills:** {', '.join(missing[:5])}. Adding these would significantly boost your score.")

    # Experience factor
    exp_score = breakdown.get("Experience", 0)
    if exp_score > 50:
        factors.append(f"💼 **Experience ({exp_score}%):** Relevant experience detected. Quantify your achievements for even higher scores.")
    else:
        factors.append(f"💼 **Experience ({exp_score}%):** Limited experience signal. Add years of experience, company names, and measurable results.")

    # Education factor
    edu_score = breakdown.get("Education", 0)
    if edu_score > 50:
        factors.append(f"🎓 **Education ({edu_score}%):** Relevant degree detected.")
    else:
        factors.append(f"🎓 **Education ({edu_score}%):** Education details are weak or missing. Include degree, institution, and year.")

    # Fairness
    if bias:
        factors.append(f"⚖️ **Fairness Penalty:** {len(bias)} bias-inducing entities detected (names, locations, dates). A small penalty was applied to ensure fair ranking (α={alpha}).")
    else:
        factors.append(f"⚖️ **Fairness:** No bias entities detected — clean profile. No penalty applied.")

    # Overall verdict
    if rjas >= 75:
        verdict = "🏆 **Verdict: STRONG FIT** — This candidate is highly aligned with the role requirements."
    elif rjas >= 50:
        verdict = "🎯 **Verdict: MODERATE FIT** — Good potential with room for improvement in specific areas."
    else:
        verdict = "📋 **Verdict: DEVELOPING FIT** — Significant gaps exist. Focus on the missing skills identified above."

    return {
        "factors": factors,
        "verdict": verdict,
        "improvement_priority": _prioritize_improvements(breakdown, missing),
    }


def _prioritize_improvements(breakdown, missing):
    """Ranks which area to improve first based on lowest scores."""
    areas = [
        ("Semantic Match", breakdown.get("Semantic", 0), "Rewrite resume summary to match JD language"),
        ("Skills", breakdown.get("Skills", 0), f"Learn: {', '.join(missing[:3])}" if missing else "Good skill coverage"),
        ("Experience", breakdown.get("Experience", 0), "Add quantifiable achievements and years of experience"),
        ("Education", breakdown.get("Education", 0), "Include degree details clearly"),
    ]
    areas.sort(key=lambda x: x[1])
    return [{"area": a[0], "score": a[1], "action": a[2]} for a in areas[:3]]
