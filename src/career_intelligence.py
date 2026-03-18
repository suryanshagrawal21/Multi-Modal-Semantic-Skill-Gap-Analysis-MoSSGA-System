"""
NeuroHire Career Intelligence Engine
=====================================
Provides job recommendations, skill gap analysis, learning roadmaps,
career guidance, and structured JSON output for the NeuroHire system.
"""
from __future__ import annotations


import json
import re


# ---------------------------------------------------------------------------
# Role Skill Database (Research-Grade Knowledge Base)
# ---------------------------------------------------------------------------

ROLE_SKILL_DB = {
    "data scientist": {
        "core": ["python", "sql", "statistics", "machine learning", "data visualization"],
        "intermediate": ["scikit-learn", "pandas", "numpy", "tensorflow", "a/b testing", "feature engineering"],
        "advanced": ["deep learning", "nlp", "mlops", "docker", "cloud (aws/gcp)", "spark"],
    },
    "data analyst": {
        "core": ["sql", "excel", "python", "data visualization", "statistics"],
        "intermediate": ["tableau", "power bi", "pandas", "reporting", "etl"],
        "advanced": ["machine learning", "cloud", "dbt", "airflow"],
    },
    "web developer": {
        "core": ["html", "css", "javascript", "git", "responsive design"],
        "intermediate": ["react", "node.js", "rest apis", "typescript", "mongodb"],
        "advanced": ["docker", "ci/cd", "kubernetes", "system design", "graphql"],
    },
    "full stack developer": {
        "core": ["html", "css", "javascript", "git", "sql"],
        "intermediate": ["react", "node.js", "express", "mongodb", "rest apis"],
        "advanced": ["docker", "kubernetes", "ci/cd", "system design", "microservices"],
    },
    "frontend developer": {
        "core": ["html", "css", "javascript", "responsive design"],
        "intermediate": ["react", "typescript", "tailwind css", "testing"],
        "advanced": ["next.js", "performance optimization", "accessibility", "design systems"],
    },
    "backend developer": {
        "core": ["python", "sql", "git", "rest apis"],
        "intermediate": ["django", "flask", "postgresql", "redis", "docker"],
        "advanced": ["kubernetes", "microservices", "system design", "message queues"],
    },
    "ml engineer": {
        "core": ["python", "machine learning", "sql", "git"],
        "intermediate": ["tensorflow", "pytorch", "scikit-learn", "docker", "mlflow"],
        "advanced": ["kubernetes", "model serving", "distributed training", "mlops"],
    },
    "devops engineer": {
        "core": ["linux", "git", "docker", "scripting"],
        "intermediate": ["kubernetes", "ci/cd", "terraform", "monitoring"],
        "advanced": ["cloud architecture", "service mesh", "security", "sre practices"],
    },
    "software engineer": {
        "core": ["programming (python/java/c++)", "git", "data structures", "algorithms"],
        "intermediate": ["system design", "databases", "rest apis", "testing"],
        "advanced": ["distributed systems", "cloud", "performance", "architecture patterns"],
    },
    "project manager": {
        "core": ["communication", "agile", "scrum", "jira"],
        "intermediate": ["risk management", "stakeholder management", "budgeting"],
        "advanced": ["pmp certification", "scaled agile", "portfolio management"],
    },
}


# ---------------------------------------------------------------------------
# Level Estimation
# ---------------------------------------------------------------------------

def estimate_level(skills):
    """Estimates candidate level based on skill count and complexity signals."""
    if not skills:
        return "Entry Level"

    advanced_signals = {"docker", "kubernetes", "system design", "microservices",
                        "deep learning", "mlops", "distributed systems", "terraform",
                        "ci/cd", "cloud architecture", "pytorch", "tensorflow"}
    intermediate_signals = {"react", "node.js", "django", "flask", "scikit-learn",
                            "pandas", "sql", "mongodb", "postgresql", "rest apis",
                            "typescript", "redis", "express"}

    skills_lower = {s.lower() for s in skills}
    adv_count = len(skills_lower & advanced_signals)
    int_count = len(skills_lower & intermediate_signals)

    if adv_count >= 2 or len(skills) >= 12:
        return "Advanced"
    elif int_count >= 2 or len(skills) >= 6:
        return "Intermediate"
    else:
        return "Beginner"


# ---------------------------------------------------------------------------
# Job Recommendations
# ---------------------------------------------------------------------------

def generate_job_recommendations(skills, target_role=""):
    """Generates personalized job recommendations based on skills and target role."""
    skills_lower = {s.lower() for s in skills}
    recs = []

    # Score each role by how many of its core+intermediate skills the candidate has
    for role_name, role_skills in ROLE_SKILL_DB.items():
        all_role_skills = set(role_skills["core"] + role_skills["intermediate"])
        overlap = skills_lower & all_role_skills
        match_pct = len(overlap) / len(all_role_skills) * 100 if all_role_skills else 0

        if match_pct > 15:  # At least some relevance
            missing_core = [s for s in role_skills["core"] if s not in skills_lower]
            recs.append({
                "role": role_name.title(),
                "match_percent": round(match_pct, 1),
                "reason": _build_recommendation_reason(role_name, overlap, missing_core, match_pct),
                "required_skills": role_skills["core"],
                "your_matching_skills": list(overlap),
                "missing_core_skills": missing_core,
            })

    # Sort by match percentage descending
    recs.sort(key=lambda x: x["match_percent"], reverse=True)

    # If target role specified, prioritize it
    if target_role:
        target_lower = target_role.lower()
        prioritized = [r for r in recs if target_lower in r["role"].lower()]
        others = [r for r in recs if target_lower not in r["role"].lower()]
        recs = prioritized + others

    return recs[:5]


def _build_recommendation_reason(role, overlap, missing_core, match_pct):
    """Builds a natural language explanation for why a role was recommended."""
    if match_pct > 70:
        return f"Excellent fit! You already have {len(overlap)} matching skills. {'Focus on: ' + ', '.join(missing_core[:2]) if missing_core else 'You are well-prepared for this role.'}"
    elif match_pct > 40:
        return f"Good potential. You match {len(overlap)} skills. Bridge the gap by learning: {', '.join(missing_core[:3])}."
    else:
        return f"Emerging fit with {len(overlap)} overlapping skills. This could be a growth target if you invest in: {', '.join(missing_core[:3])}."


# ---------------------------------------------------------------------------
# Skill Gap Analysis & Learning Roadmap
# ---------------------------------------------------------------------------

def generate_skill_gap_and_roadmap(resume_skills, target_role=""):
    """Generates missing skills list and a structured learning roadmap."""
    skills_lower = {s.lower() for s in resume_skills}
    role_lower = target_role.lower() if target_role else ""

    # Find the best matching role in our DB
    best_role = None
    for role_name in ROLE_SKILL_DB:
        if role_lower and role_lower in role_name:
            best_role = role_name
            break

    if not best_role:
        # Try to infer from skills
        best_score = 0
        for role_name, role_skills in ROLE_SKILL_DB.items():
            all_s = set(role_skills["core"] + role_skills["intermediate"] + role_skills["advanced"])
            score = len(skills_lower & all_s)
            if score > best_score:
                best_score = score
                best_role = role_name

    if not best_role:
        best_role = "software engineer"

    role_data = ROLE_SKILL_DB[best_role]

    # Compute missing skills per tier
    missing_core = [s for s in role_data["core"] if s not in skills_lower]
    missing_intermediate = [s for s in role_data["intermediate"] if s not in skills_lower]
    missing_advanced = [s for s in role_data["advanced"] if s not in skills_lower]

    all_missing = missing_core + missing_intermediate + missing_advanced

    roadmap = []
    if missing_core:
        roadmap.append({
            "level": "🟢 Beginner",
            "focus": f"Master Core {best_role.title()} Foundations",
            "technologies": missing_core,
            "timeline": f"{len(missing_core) * 2}-{len(missing_core) * 3} Weeks",
            "resources": _get_resources(missing_core),
            "priority": "HIGH — Start here immediately",
        })
    if missing_intermediate:
        roadmap.append({
            "level": "🟡 Intermediate",
            "focus": f"Build Professional {best_role.title()} Skills",
            "technologies": missing_intermediate,
            "timeline": f"{len(missing_intermediate) * 2}-{len(missing_intermediate) * 3} Weeks",
            "resources": _get_resources(missing_intermediate),
            "priority": "MEDIUM — After core skills are solid",
        })
    if missing_advanced:
        roadmap.append({
            "level": "🔴 Advanced",
            "focus": f"Specialize & Stand Out as {best_role.title()}",
            "technologies": missing_advanced,
            "timeline": "8-16 Weeks",
            "resources": _get_resources(missing_advanced),
            "priority": "LONG-TERM — Career differentiator",
        })

    return all_missing, roadmap


def _get_resources(skills):
    """Maps skills to practical learning resources."""
    resource_map = {
        "python": "Python.org Tutorial, Automate the Boring Stuff",
        "sql": "SQLBolt.com, Kaggle SQL Micro-Course",
        "machine learning": "Andrew Ng's ML Specialization (Coursera)",
        "deep learning": "DeepLearning.AI, Fast.ai",
        "docker": "Docker Getting Started (docs.docker.com), Play with Docker",
        "kubernetes": "Kubernetes.io Tutorials, KodeKloud",
        "react": "React.dev Official Tutorial, Scrimba React Course",
        "node.js": "NodeSchool.io, The Odin Project",
        "git": "Git-SCM Book, GitHub Skills",
        "tensorflow": "TensorFlow.org Tutorials",
        "pytorch": "PyTorch.org Tutorials",
        "scikit-learn": "Scikit-learn User Guide, Kaggle Learn",
        "typescript": "TypeScript Handbook (typescriptlang.org)",
        "system design": "System Design Primer (GitHub), Grokking System Design",
        "ci/cd": "GitHub Actions Docs, GitLab CI Tutorial",
    }
    resources = []
    for skill in skills[:3]:
        if skill.lower() in resource_map:
            resources.append(resource_map[skill.lower()])
        else:
            resources.append(f"Search '{skill} tutorial' on Coursera/YouTube")
    return " | ".join(resources) if resources else "Explore Coursera, Udemy, or YouTube for targeted courses"


# ---------------------------------------------------------------------------
# Career Guidance
# ---------------------------------------------------------------------------

def generate_career_guidance(level, role):
    """Generates personalized, actionable career advice."""
    role_title = role if role else "your target field"

    if level == "Beginner":
        return (
            f"🎯 **Action Plan for {role_title}:**\n"
            f"1. Build 2-3 end-to-end projects and deploy them publicly (GitHub + live demo)\n"
            f"2. Contribute to open-source projects in {role_title} to build credibility\n"
            f"3. Network on LinkedIn — connect with {role_title} professionals and share your learning journey\n"
            f"4. Target internships or junior positions to gain real-world experience\n"
            f"5. Get at least one relevant certification to validate your skills\n\n"
            f"⏱ **Timeline to first role:** 3-6 months with consistent effort"
        )
    elif level == "Intermediate":
        return (
            f"🎯 **Action Plan for {role_title}:**\n"
            f"1. Deepen your system design and architecture knowledge\n"
            f"2. Build a portfolio project that solves a real business problem with measurable impact\n"
            f"3. Practice technical interviews (LeetCode, system design mock interviews)\n"
            f"4. Consider specializing in a niche area within {role_title}\n"
            f"5. Start mentoring beginners — it deepens your own understanding\n\n"
            f"⏱ **Timeline to mid-level role:** 2-4 months of focused preparation"
        )
    else:
        return (
            f"🎯 **Action Plan for {role_title}:**\n"
            f"1. Focus on demonstrating leadership and driving measurable business impact\n"
            f"2. Publish technical blogs or speak at meetups/conferences\n"
            f"3. Consider transitioning to senior/lead roles or architecture positions\n"
            f"4. Mentor team members and build cross-functional collaboration skills\n"
            f"5. Stay current with industry trends and emerging technologies\n\n"
            f"⏱ **You are well-positioned for senior roles**"
        )


# ---------------------------------------------------------------------------
# Resume Section Extraction Helpers
# ---------------------------------------------------------------------------

def extract_education_from_text(text):
    """Extracts education entries from resume text using pattern matching."""
    edu_keywords = [
        r"b\.?tech", r"b\.?sc", r"b\.?e\b", r"bachelor", r"m\.?tech", r"m\.?sc",
        r"master", r"mba", r"ph\.?d", r"doctorate", r"diploma", r"12th", r"10th",
        r"cgpa", r"gpa", r"percentage", r"university", r"college", r"institute",
    ]
    lines = text.split("\n")
    edu_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        if any(re.search(kw, line_lower) for kw in edu_keywords) and len(line.strip()) > 5:
            edu_lines.append(line.strip())
    return edu_lines if edu_lines else ["No education details detected"]


def extract_experience_from_text(text):
    """Extracts experience entries from resume text."""
    exp_keywords = [
        r"\d+\s*\+?\s*years?", r"intern", r"developer", r"engineer", r"analyst",
        r"manager", r"present", r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec",
        r"20\d{2}", r"company", r"organization", r"worked",
    ]
    lines = text.split("\n")
    exp_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        if any(re.search(kw, line_lower) for kw in exp_keywords) and len(line.strip()) > 10:
            exp_lines.append(line.strip())
    return exp_lines[:10] if exp_lines else ["No experience details detected"]


def extract_projects_from_text(text):
    """Extracts project entries from resume text."""
    proj_keywords = [
        r"project", r"developed", r"built", r"created", r"implemented",
        r"designed", r"system", r"application", r"app", r"website", r"tool",
    ]
    lines = text.split("\n")
    proj_lines = []
    in_project_section = False
    for line in lines:
        line_lower = line.lower().strip()
        if "project" in line_lower and len(line.strip()) < 30:
            in_project_section = True
            continue
        if in_project_section and line.strip():
            proj_lines.append(line.strip())
            if len(proj_lines) > 8:
                break
        if any(re.search(kw, line_lower) for kw in proj_keywords) and len(line.strip()) > 15:
            if line.strip() not in proj_lines:
                proj_lines.append(line.strip())
    return proj_lines[:8] if proj_lines else ["No project details detected"]


# ---------------------------------------------------------------------------
# Strengths & Weaknesses Generator
# ---------------------------------------------------------------------------

def analyze_strengths_weaknesses(skills, readability, education_lines, experience_lines, project_lines):
    """Generates detailed strengths and weaknesses based on extracted resume data."""
    strengths = []
    weaknesses = []

    # Skills analysis
    if len(skills) >= 8:
        strengths.append(f"Diverse skill set with {len(skills)} identified technologies — shows versatility.")
    elif len(skills) >= 4:
        strengths.append(f"Solid foundation with {len(skills)} relevant skills detected.")
    else:
        weaknesses.append(f"Only {len(skills)} skills detected. Add more technical skills to improve ATS matching.")

    # Readability
    if readability >= 0.6:
        strengths.append("Resume has good readability and clear structure — ATS-friendly formatting.")
    else:
        weaknesses.append("Resume readability is below optimal. Consider shorter sentences and clearer bullet points.")

    # Education
    if education_lines and education_lines[0] != "No education details detected":
        strengths.append("Education section is present and detectable by parsing systems.")
    else:
        weaknesses.append("Education section is missing or poorly formatted — add degree, university, and year clearly.")

    # Experience
    if experience_lines and experience_lines[0] != "No experience details detected":
        strengths.append(f"Experience section detected with {len(experience_lines)} relevant entries.")
    else:
        weaknesses.append("No clear experience entries found. Add company names, dates, and quantifiable achievements.")

    # Projects
    if project_lines and project_lines[0] != "No project details detected":
        strengths.append(f"{len(project_lines)} project entries detected — demonstrates hands-on capability.")
    else:
        weaknesses.append("No project section detected. Adding 2-3 projects with links significantly boosts your profile.")

    # General best-practice checks
    weaknesses.append("Consider adding quantifiable metrics (e.g., 'improved efficiency by 30%') to experience bullets.")

    return strengths, weaknesses


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

def format_neurohire_json(
    mode,
    resume_score="N/A",
    estimated_level="N/A",
    strengths=None,
    weaknesses=None,
    skills=None,
    missing_skills=None,
    job_match_score="N/A",
    recommended_jobs=None,
    learning_path=None,
    interview_score="N/A",
    interview_feedback="N/A",
    career_advice="N/A",
    education=None,
    experience=None,
    projects=None,
):
    """Assembles all components into the STRICT JSON format."""
    return {
        "mode": mode,
        "resume_score": str(resume_score),
        "estimated_level": estimated_level,
        "strengths": strengths or [],
        "weaknesses": weaknesses or [],
        "skills": skills or [],
        "education": education or [],
        "experience": experience or [],
        "projects": projects or [],
        "missing_skills": missing_skills or [],
        "job_match_score": str(job_match_score),
        "recommended_jobs": recommended_jobs or [],
        "learning_path": learning_path or [],
        "interview_score": str(interview_score),
        "interview_feedback": str(interview_feedback),
        "career_advice": str(career_advice),
    }
