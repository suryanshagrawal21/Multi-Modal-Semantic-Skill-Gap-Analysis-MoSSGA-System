import os
import tempfile
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.adaptive_engine import RLRankingAgent
from src.experiment import ExperimentLab
from src.matcher import calculate_similarity, calculate_sbert_similarity, find_missing_skills
from src.parser import extract_text
from src.preprocessing import clean_text, extract_skills, preprocess_text, COMMON_SKILLS
from src.research_lab import (
    analyze_fairness,
    calculate_statistics,
    compare_algorithms,
    detect_bias_entities,
    generate_pareto_frontier,
    simulate_rl_convergence,
)
from src.scoring import calculate_readability, calculate_composite_score
from src.career_intelligence import (
    generate_job_recommendations,
    generate_skill_gap_and_roadmap,
    estimate_level,
    generate_career_guidance,
    format_neurohire_json,
    extract_education_from_text,
    extract_experience_from_text,
    extract_projects_from_text,
    analyze_strengths_weaknesses,
)
from src.cognitive_engine import (
    build_cognitive_skill_graph,
    predict_career_trajectory,
    detect_skill_authenticity,
    generate_deep_explanation,
)
from src.mossga_engine import run_mossga_pipeline
from src.github_analyzer import analyze_github_profile
from src.semantic_skill_matcher import semantic_skill_match, calculate_semantic_gap_score
from src.multimodal_fusion import fuse_skill_profiles, generate_fusion_summary

# ---------------------------------------------------------------------------
# Session & Page Setup
# ---------------------------------------------------------------------------

if "adaptive_engine" not in st.session_state:
    st.session_state["adaptive_engine"] = RLRankingAgent()

st.set_page_config(
    page_title="NeuroHire MoSSGA — AI Career Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #111827 40%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #818cf8 !important;
        -webkit-text-fill-color: #818cf8 !important;
        background: none !important;
    }

    /* Gradient Headers */
    .main h1 {
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
        font-size: 2.8rem !important;
    }
    .main h2 {
        background: linear-gradient(135deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .main h3 {
        background: linear-gradient(135deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }

    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin: 4px 0;
    }
    .mode-full { background: linear-gradient(135deg, #059669, #10b981); color: white; }
    .mode-partial { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; }
    .mode-guidance { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; }

    /* Status bar */
    .status-bar {
        display: flex;
        gap: 16px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 12px 20px;
        margin: 16px 0;
        flex-wrap: wrap;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* JSON export box */
    .json-export {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        font-family: 'Fira Code', 'Cascadia Code', monospace;
        font-size: 0.82rem;
        color: #c9d1d9;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 500px;
        overflow-y: auto;
    }

    /* Score gauge */
    .score-gauge {
        text-align: center;
        padding: 12px;
    }
    .score-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .score-green { color: #22c55e; }
    .score-yellow { color: #eab308; }
    .score-red { color: #ef4444; }

    /* Roadmap steps */
    .roadmap-step {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .roadmap-beginner { border-color: #22c55e; }
    .roadmap-intermediate { border-color: #eab308; }
    .roadmap-advanced { border-color: #ef4444; }

    /* Metric cards override */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(99, 102, 241, 0.15);
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
    }

    /* Button styling */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header + System Status Bar
# ---------------------------------------------------------------------------

st.title("🧠 NeuroHire + MoSSGA")
st.markdown("##### Multi-Modal Semantic Skill Gap Analysis · AI-Powered Recruitment & Career Intelligence")

st.markdown("""
<div class="status-bar">
    <div class="status-item"><div class="status-dot"></div> SBERT Model Active</div>
    <div class="status-item"><div class="status-dot"></div> RL Agent Online</div>
    <div class="status-item"><div class="status-dot"></div> NLP Pipeline Ready</div>
    <div class="status-item"><div class="status-dot"></div> Bias Filter Enabled</div>
    <div class="status-item"><div class="status-dot"></div> Career Engine Loaded</div>
    <div class="status-item"><div class="status-dot"></div> MoSSGA Engine Ready</div>
    <div class="status-item"><div class="status-dot"></div> GitHub Analyzer Active</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📋 Input Panel")

    st.markdown("---")
    st.markdown("### 📄 Resume Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX resumes", type=["pdf", "docx"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) loaded")

    st.markdown("---")
    st.markdown("### 🐙 GitHub Profile")
    github_input = st.text_input(
        "GitHub URL or Username",
        placeholder="e.g. https://github.com/username or just username",
    )
    if github_input:
        st.success("✅ GitHub profile will be analyzed")

    st.markdown("---")
    st.markdown("### 🎯 Target Profile")
    target_role = st.text_input("Target Role", placeholder="e.g. Data Scientist, Web Developer")
    manual_skills = st.text_area(
        "Your Skills (comma separated)",
        placeholder="e.g. Python, SQL, React, Machine Learning",
        height=80,
    )

    st.markdown("---")
    st.markdown("### 📌 Job Description")
    jd_input = st.text_area("Paste Job Description here", height=120, label_visibility="collapsed",
                            placeholder="Paste a job description to get semantic matching...")

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    bias_filter = st.checkbox("Enable Bias-Aware Filtering", value=True)
    use_sbert = st.checkbox("Use Deep Semantic (SBERT)", value=True)
    alpha_input = st.slider(
        "Fairness ↔ Accuracy Balance",
        min_value=0.0, max_value=1.0, value=0.9, step=0.1,
        help="1.0 = Pure Accuracy | 0.0 = Pure Fairness",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run MoSSGA + NeuroHire Analysis", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Helper: Score color class
# ---------------------------------------------------------------------------

def score_color(val):
    """Returns CSS class for score coloring."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "score-yellow"
    if v >= 70:
        return "score-green"
    elif v >= 40:
        return "score-yellow"
    return "score-red"


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

if run_btn:
    has_resumes = bool(uploaded_files)
    has_role_skills = bool(target_role or manual_skills)
    has_jd = bool(jd_input)
    has_github = bool(github_input)

    # --- GUIDANCE MODE ---
    if not has_resumes and not has_role_skills and not has_jd and not has_github:
        st.markdown("""
        <div class="glass-card">
            <span class="mode-badge mode-guidance">🔮 GUIDANCE MODE</span>
            <h3>I couldn't find any input data!</h3>
            <p>I need at least one of:</p>
            <ul>
                <li>📄 <b>Resume</b> — Upload a PDF or DOCX for full analysis</li>
                <li>🎯 <b>Target Role + Skills</b> — Enter in the sidebar for partial analysis</li>
                <li>📌 <b>Job Description</b> — Paste a JD for matching</li>
            </ul>
            <p>💡 <b>Tip:</b> Even just entering your target role and a few skills will give you personalized job recommendations and a learning roadmap!</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # --- Determine Mode ---
    if has_resumes:
        mode = "Full Analysis Mode"
        badge_class = "mode-full"
    elif has_github:
        mode = "GitHub Profile Mode"
        badge_class = "mode-partial"
    else:
        mode = "Partial Profile Mode"
        badge_class = "mode-partial"

    st.markdown(f'<span class="mode-badge {badge_class}">🔬 {mode.upper()}</span>', unsafe_allow_html=True)

    # --- MoSSGA Pipeline ---
    mossga_report = None

    # --- Progress Animation ---
    progress_bar = st.progress(0, text="🔄 Initializing NeuroHire engines...")

    # --- Pre-processing ---
    jd_text = clean_text(jd_input) if jd_input else ""
    jd_processed = preprocess_text(jd_text) if jd_text else ""
    jd_skills = extract_skills(jd_input) if jd_input else []
    progress_bar.progress(15, text="📝 Processing job description...")

    rl_agent = st.session_state["adaptive_engine"]
    role_context = rl_agent.detect_role(jd_text) if jd_text else (target_role or "General")
    current_weights = rl_agent.get_weights(explore=True)
    progress_bar.progress(25, text="🧠 RL Agent configured...")

    results = []
    neurohire_jsons = []

    # ===========================================================
    # FULL ANALYSIS MODE
    # ===========================================================
    if has_resumes:
        cleaned_texts = []
        for i, file in enumerate(uploaded_files):
            progress_bar.progress(30 + int(20 * (i / len(uploaded_files))),
                                  text=f"📄 Parsing resume {i+1}/{len(uploaded_files)}: {file.name}...")

            file_ext = file.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            text = extract_text(tmp_path)
            os.remove(tmp_path)

            if not text:
                continue

            detected_bias = detect_bias_entities(text) if bias_filter else []
            bias_penalty = 1.0 if detected_bias else 0.0

            cleaned = clean_text(text)
            processed = preprocess_text(cleaned)
            cleaned_texts.append(processed)
            extracted_skills = extract_skills(text)

            # Extract structured sections
            education = extract_education_from_text(text)
            experience = extract_experience_from_text(text)
            projects = extract_projects_from_text(text)
            readability = calculate_readability(text)

            results.append({
                "Name": file.name,
                "Raw Text": text,
                "Cleaned Text": processed,
                "Skills": extracted_skills,
                "Education": education,
                "Experience": experience,
                "Projects": projects,
                "Detected Bias": detected_bias,
                "Bias Penalty": bias_penalty,
                "Readability": readability,
                "Processing Status": "Success",
            })

        progress_bar.progress(55, text="🧮 Computing SBERT semantic similarity...")

        # Score computation (if JD exists)
        if jd_input and results:
            valid_cleaned = [r["Cleaned Text"] for r in results]
            tfidf_scores = calculate_similarity(valid_cleaned, jd_processed)
            sbert_scores = (
                calculate_sbert_similarity(valid_cleaned, jd_processed)
                if use_sbert
                else [0] * len(results)
            )
            progress_bar.progress(70, text="📊 Computing RJAS composite scores...")

            for idx, res in enumerate(results):
                matched_skills = list(set(res["Skills"]).intersection(set(jd_skills)))
                semantic_score = sbert_scores[idx] if use_sbert else tfidf_scores[idx]

                final_score, breakdown = calculate_composite_score(
                    resume_text=res["Raw Text"],
                    jd_text=jd_input,
                    skill_match_count=len(matched_skills),
                    total_jd_skills=len(jd_skills),
                    nlp_similarity=tfidf_scores[idx],
                    sbert_similarity=semantic_score,
                    weights=current_weights,
                    bias_penalty=res["Bias Penalty"],
                    alpha=alpha_input,
                )
                res.update({
                    "NLP Score": round(tfidf_scores[idx], 4),
                    "SBERT Score": round(sbert_scores[idx], 4),
                    "Final Score": final_score,
                    "RJAS": final_score,
                    "Matched Skills": matched_skills,
                    "Missing Skills": find_missing_skills(res["Raw Text"], jd_input, COMMON_SKILLS),
                    "Breakdown": breakdown,
                })

        progress_bar.progress(80, text="🔀 Running MoSSGA multi-modal analysis...")

        # Run MoSSGA pipeline for the first resume
        if results:
            first_resume_text = results[0].get("Raw Text", "")
            mossga_report = run_mossga_pipeline(
                resume_text=first_resume_text,
                github_input=github_input if has_github else "",
                jd_text=jd_input if has_jd else "",
                target_role=target_role or role_context,
            )

        progress_bar.progress(90, text="🎯 Generating career intelligence...")

        # Build NeuroHire JSON for each resume
        for res in results:
            skills_list = res["Skills"]
            level = estimate_level(skills_list)
            missing, roadmap = generate_skill_gap_and_roadmap(skills_list, role_context)
            recs = generate_job_recommendations(skills_list, role_context) if not has_jd else []

            strengths, weaknesses = analyze_strengths_weaknesses(
                skills_list, res["Readability"],
                res["Education"], res["Experience"], res["Projects"]
            )

            ats_score = round(res.get("Readability", 0.5) * 100, 1)
            match_score = str(round(res["RJAS"], 2)) if "RJAS" in res else "N/A"

            n_json = format_neurohire_json(
                mode=mode,
                resume_score=str(ats_score),
                estimated_level=level,
                strengths=strengths,
                weaknesses=weaknesses,
                skills=skills_list,
                missing_skills=res.get("Missing Skills", missing),
                job_match_score=match_score,
                recommended_jobs=recs,
                learning_path=roadmap,
                career_advice=generate_career_guidance(level, role_context),
                education=res["Education"],
                experience=res["Experience"],
                projects=res["Projects"],
            )
            neurohire_jsons.append((res["Name"], n_json, res))

    # ===========================================================
    # PARTIAL PROFILE MODE
    # ===========================================================
    elif has_role_skills or has_github:
        progress_bar.progress(40, text="🎯 Analyzing your profile...")

        parsed_skills = [s.strip() for s in manual_skills.split(",")] if manual_skills else []

        # Run MoSSGA for GitHub + manual skills
        mossga_report = run_mossga_pipeline(
            resume_text="",
            github_input=github_input if has_github else "",
            jd_text=jd_input if has_jd else "",
            target_role=target_role or "General",
        )

        # Merge GitHub-discovered skills with manually entered ones
        if mossga_report and mossga_report.get("github_analysis"):
            gh_skills = mossga_report["github_analysis"].get("skills", [])
            parsed_skills = list(set(parsed_skills + gh_skills))

        level = estimate_level(parsed_skills)
        missing, roadmap = generate_skill_gap_and_roadmap(parsed_skills, target_role)
        recs = generate_job_recommendations(parsed_skills, target_role)

        progress_bar.progress(80, text="📋 Building recommendations...")

        n_json = format_neurohire_json(
            mode=mode,
            resume_score="N/A",
            estimated_level=level,
            strengths=[f"Active learning across {len(parsed_skills)} skill areas."],
            weaknesses=["No resume provided — upload one for a detailed ATS score and section analysis."],
            skills=parsed_skills,
            missing_skills=missing,
            job_match_score="N/A",
            recommended_jobs=recs,
            learning_path=roadmap,
            career_advice=generate_career_guidance(level, target_role),
        )
        neurohire_jsons.append(("Your Profile", n_json, {}))

    progress_bar.progress(100, text="✅ Analysis complete!")

    # ===================================================================
    # VISUALIZATION TABS
    # ===================================================================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🧠 NeuroHire Insights",
        "🐙 GitHub Analysis",
        "🔀 Multi-Modal Fusion",
        "🎯 MoSSGA Skill Gap",
        "🕸️ Cognitive Skill Graph",
        "📈 Career Trajectory",
        "📊 RJAS Ranking",
        "📢 Explainability (XAI)",
        "🧪 Research Lab",
        "🔄 RL Feedback",
    ])

    # --- TAB 1: NeuroHire Insights ---
    with tab1:
        for name, n_json, res_data in neurohire_jsons:
            st.markdown(f"## 📋 Analysis: {name}")

            # Top metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                sc = n_json["resume_score"]
                st.markdown(f"""
                <div class="score-gauge">
                    <div class="score-value {score_color(sc)}">{sc}</div>
                    <div class="score-label">ATS / Resume Score</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                ms = n_json["job_match_score"]
                st.markdown(f"""
                <div class="score-gauge">
                    <div class="score-value {score_color(ms)}">{ms}</div>
                    <div class="score-label">Job Match Score</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="score-gauge">
                    <div class="score-value" style="color: #a78bfa;">{n_json["estimated_level"]}</div>
                    <div class="score-label">Estimated Level</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="score-gauge">
                    <div class="score-value" style="color: #60a5fa;">{len(n_json["skills"])}</div>
                    <div class="score-label">Skills Detected</div>
                </div>""", unsafe_allow_html=True)

            # Skills detected
            st.markdown("### 🛠️ Skills Detected")
            if n_json["skills"]:
                skill_html = " ".join(
                    [f'<span style="background:rgba(99,102,241,0.2); color:#a5b4fc; padding:4px 12px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; border:1px solid rgba(99,102,241,0.3);">{s}</span>'
                     for s in n_json["skills"]]
                )
                st.markdown(f'<div style="margin:8px 0;">{skill_html}</div>', unsafe_allow_html=True)

            # Extracted sections (Full mode only)
            if n_json.get("education") or n_json.get("experience") or n_json.get("projects"):
                sec1, sec2, sec3 = st.columns(3)
                with sec1:
                    st.markdown("### 🎓 Education")
                    for e in (n_json.get("education") or ["Not detected"]):
                        st.markdown(f"- {e}")
                with sec2:
                    st.markdown("### 💼 Experience")
                    for e in (n_json.get("experience") or ["Not detected"])[:5]:
                        st.markdown(f"- {e}")
                with sec3:
                    st.markdown("### 🔨 Projects")
                    for p in (n_json.get("projects") or ["Not detected"])[:5]:
                        st.markdown(f"- {p}")

            # Strengths / Weaknesses
            sw1, sw2 = st.columns(2)
            with sw1:
                st.markdown("### ✅ Strengths")
                for s in n_json["strengths"]:
                    st.success(s)
            with sw2:
                st.markdown("### ⚠️ Weaknesses")
                for w in n_json["weaknesses"]:
                    st.warning(w)

            # Missing skills
            if n_json["missing_skills"]:
                st.markdown("### 🔍 Missing Skills for Target Role")
                miss_html = " ".join(
                    [f'<span style="background:rgba(239,68,68,0.15); color:#fca5a5; padding:4px 12px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; border:1px solid rgba(239,68,68,0.3);">{s}</span>'
                     for s in n_json["missing_skills"]]
                )
                st.markdown(f'<div style="margin:8px 0;">{miss_html}</div>', unsafe_allow_html=True)

            # Learning Roadmap
            if n_json["learning_path"]:
                st.markdown("### 🗺️ Personalized Learning Roadmap")
                for step in n_json["learning_path"]:
                    level_key = "beginner" if "Beginner" in step["level"] else ("intermediate" if "Intermediate" in step["level"] else "advanced")
                    st.markdown(f"""
                    <div class="roadmap-step roadmap-{level_key}">
                        <b>{step['level']}</b> — {step['focus']}<br/>
                        <span style="color:#94a3b8;">⏱ {step['timeline']} | 📦 {', '.join(step['technologies'])}</span><br/>
                        <span style="color:#64748b; font-size:0.8rem;">📚 {step['resources']}</span>
                    </div>""", unsafe_allow_html=True)

            # Job Recommendations
            if n_json["recommended_jobs"]:
                st.markdown("### 💼 Recommended Job Roles")
                for job in n_json["recommended_jobs"]:
                    match_pct = job.get("match_percent", "—")
                    st.markdown(f"""
                    <div class="glass-card">
                        <b style="color:#a5b4fc; font-size:1.1rem;">{job['role']}</b>
                        <span style="float:right; color:#22c55e; font-weight:600;">{match_pct}% match</span><br/>
                        <span style="color:#94a3b8;">{job['reason']}</span>
                    </div>""", unsafe_allow_html=True)

            # Career Advice
            st.markdown("### 🎯 Career Guidance")
            st.info(n_json["career_advice"])

            # JSON Export
            with st.expander("📦 Export Full JSON Report", expanded=False):
                st.markdown(f'<div class="json-export">{json.dumps(n_json, indent=2, ensure_ascii=False)}</div>',
                            unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(n_json, indent=2, ensure_ascii=False),
                    file_name=f"neurohire_report_{name.replace(' ', '_')}.json",
                    mime="application/json",
                )

    # --- TAB 2: GitHub Analysis ---
    with tab2:
        st.markdown("## 🐙 GitHub Profile Analysis")
        st.markdown("*Analyzing GitHub repositories to validate practical skills and project experience.*")

        if mossga_report and mossga_report.get("github_analysis"):
            gh = mossga_report["github_analysis"]

            if gh.get("status") == "success":
                profile = gh.get("profile", {})

                # Profile header
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; align-items:center; gap:16px;">
                        <img src="{profile.get('avatar_url', '')}" style="width:64px; height:64px; border-radius:50%; border:2px solid rgba(99,102,241,0.5);" />
                        <div>
                            <b style="font-size:1.3rem; color:#a5b4fc;">{profile.get('name', profile.get('username', ''))}</b><br/>
                            <span style="color:#94a3b8;">@{profile.get('username', '')} · {profile.get('public_repos', 0)} public repos · {profile.get('followers', 0)} followers</span><br/>
                            <span style="color:#64748b; font-size:0.85rem;">{profile.get('bio', '') or 'No bio'}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                metrics = gh.get("contribution_metrics", {})
                gm1, gm2, gm3, gm4 = st.columns(4)
                gm1.metric("Repos Analyzed", gh.get("repos_analyzed", 0))
                gm2.metric("Total Stars", metrics.get("total_stars", 0))
                gm3.metric("Languages", metrics.get("language_diversity", 0))
                gm4.metric("Avg Complexity", f"{metrics.get('avg_project_complexity', 0)}%")

                # Language distribution chart
                languages = gh.get("languages", {})
                if languages:
                    st.markdown("### 📊 Language Distribution")
                    lang_df = pd.DataFrame([
                        {"Language": lang, "Percentage": pct}
                        for lang, pct in list(languages.items())[:10]
                    ])
                    fig_lang = px.bar(
                        lang_df, x="Language", y="Percentage",
                        color="Percentage", color_continuous_scale="Viridis",
                        title="Top Languages by Code Volume",
                    )
                    fig_lang.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_lang, use_container_width=True)

                # Skills from GitHub
                gh_skills = gh.get("skills", [])
                if gh_skills:
                    st.markdown("### 🛠️ Skills Inferred from GitHub")
                    skill_sources = gh.get("skill_sources", {})
                    src_cols = st.columns(3)
                    with src_cols[0]:
                        st.markdown("**From Languages:**")
                        for s in skill_sources.get("from_languages", []):
                            st.markdown(f"- {s}")
                    with src_cols[1]:
                        st.markdown("**From Topics:**")
                        for s in skill_sources.get("from_topics", []):
                            st.markdown(f"- {s}")
                    with src_cols[2]:
                        st.markdown("**From Descriptions:**")
                        for s in skill_sources.get("from_descriptions", []):
                            st.markdown(f"- {s}")

                # Top projects by complexity
                projects = gh.get("projects", [])
                if projects:
                    st.markdown("### 🏗️ Top Projects by Complexity")
                    for proj in projects[:6]:
                        complexity = proj.get("complexity_score", 0)
                        bar_color = '#22c55e' if complexity >= 50 else ('#eab308' if complexity >= 25 else '#94a3b8')
                        st.markdown(f"""
                        <div class="glass-card" style="padding:14px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <b style="color:#a5b4fc;">{proj['name']}</b>
                                    <span style="color:#94a3b8; margin-left:8px; font-size:0.85rem;">⭐ {proj.get('stargazers_count', 0)} · 🍴 {proj.get('forks_count', 0)}</span>
                                </div>
                                <span style="color:{bar_color}; font-weight:600;">Complexity: {complexity}%</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.3); border-radius:8px; height:8px; margin:8px 0;">
                                <div style="background:{bar_color}; width:{complexity}%; height:100%; border-radius:8px;"></div>
                            </div>
                            <span style="color:#64748b; font-size:0.85rem;">{proj.get('description', 'No description')[:120]}</span>
                        </div>""", unsafe_allow_html=True)

            elif gh.get("status") == "no_repos":
                st.warning("GitHub profile found but no public repositories to analyze.")
            else:
                st.error("Failed to fetch GitHub profile. Check the username/URL and try again.")
        else:
            st.markdown("""
            <div class="glass-card">
                <p style="color:#94a3b8;">🐙 Enter a GitHub URL or username in the sidebar to analyze repositories, languages, and validate practical skills.</p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 3: Multi-Modal Fusion ---
    with tab3:
        st.markdown("## 🔀 Multi-Modal Fusion")
        st.markdown("*Combining resume and GitHub data into a unified skill profile with confidence scoring.*")

        if mossga_report and mossga_report.get("fusion_result"):
            fusion = mossga_report["fusion_result"]
            stats = fusion.get("fusion_stats", {})
            val = fusion.get("validation_summary", {})

            # Fusion metrics
            fm1, fm2, fm3, fm4 = st.columns(4)
            fm1.metric("Total Unique Skills", stats.get("total_unique", 0))
            fm2.metric("Cross-Validated", stats.get("validated_count", 0))
            fm3.metric("Validation Rate", f"{stats.get('validation_rate', 0)}%")
            fm4.metric("Sources", f"R:{stats.get('resume_skill_count', 0)} G:{stats.get('github_skill_count', 0)}")

            # Fusion summary
            st.markdown(mossga_report.get("fusion_summary", ""))

            # Side-by-side comparison
            comp1, comp2, comp3 = st.columns(3)
            with comp1:
                st.markdown("### ✅ Both Sources")
                for s in val.get("both_sources", []):
                    conf = fusion.get("confidence_scores", {}).get(s, 0)
                    st.markdown(f"""
                    <div style="background:rgba(34,197,94,0.15); padding:6px 12px; border-radius:8px; margin:4px 0;">
                        <span style="color:#86efac;">{s.title()}</span>
                        <span style="float:right; color:#22c55e; font-weight:600;">{conf}</span>
                    </div>""", unsafe_allow_html=True)
                if not val.get("both_sources"):
                    st.markdown("*No cross-validated skills*")

            with comp2:
                st.markdown("### 📄 Resume Only")
                for s in val.get("resume_only", []):
                    conf = fusion.get("confidence_scores", {}).get(s, 0)
                    st.markdown(f"""
                    <div style="background:rgba(99,102,241,0.15); padding:6px 12px; border-radius:8px; margin:4px 0;">
                        <span style="color:#a5b4fc;">{s.title()}</span>
                        <span style="float:right; color:#6366f1; font-weight:600;">{conf}</span>
                    </div>""", unsafe_allow_html=True)
                if not val.get("resume_only"):
                    st.markdown("*No resume-only skills*")

            with comp3:
                st.markdown("### 🐙 GitHub Only")
                for s in val.get("github_only", []):
                    conf = fusion.get("confidence_scores", {}).get(s, 0)
                    st.markdown(f"""
                    <div style="background:rgba(234,179,8,0.15); padding:6px 12px; border-radius:8px; margin:4px 0;">
                        <span style="color:#fde68a;">{s.title()}</span>
                        <span style="float:right; color:#eab308; font-weight:600;">{conf}</span>
                    </div>""", unsafe_allow_html=True)
                if not val.get("github_only"):
                    st.markdown("*No GitHub-only skills*")

            # Confidence distribution chart
            conf_scores = fusion.get("confidence_scores", {})
            if conf_scores:
                st.markdown("### 📊 Skill Confidence Distribution")
                conf_df = pd.DataFrame([
                    {"Skill": s.title(), "Confidence": c, "Source": "Both" if s in val.get("both_sources", []) else ("Resume" if s in val.get("resume_only", []) else "GitHub")}
                    for s, c in sorted(conf_scores.items(), key=lambda x: -x[1])
                ][:15]
                )
                fig_conf = px.bar(
                    conf_df, x="Skill", y="Confidence", color="Source",
                    color_discrete_map={"Both": "#22c55e", "Resume": "#6366f1", "GitHub": "#eab308"},
                    title="Skill Confidence Scores (Multi-Modal)",
                )
                fig_conf.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <p style="color:#94a3b8;">🔀 Provide both a resume and GitHub profile to see multi-modal fusion analysis. Single-source analysis is also available.</p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 4: MoSSGA Skill Gap ---
    with tab4:
        st.markdown("## 🎯 MoSSGA — Semantic Skill Gap Analysis")
        st.markdown("*Semantic matching powered by Sentence-BERT ensures similar skills are correctly interpreted.*")

        if mossga_report and mossga_report.get("semantic_match"):
            match = mossga_report["semantic_match"]
            gap = mossga_report.get("gap_analysis", {})

            # Top metrics
            sm1, sm2, sm3, sm4 = st.columns(4)
            match_pct = match.get("match_percentage", 0)
            gap_score = gap.get("gap_score", 0)
            gap_severity = gap.get("gap_severity", "N/A")
            mossga_score = mossga_report.get("mossga_score", 0)

            sm1.markdown(f"""
            <div class="score-gauge">
                <div class="score-value {score_color(match_pct)}">{match_pct}%</div>
                <div class="score-label">Semantic Match Rate</div>
            </div>""", unsafe_allow_html=True)
            sm2.markdown(f"""
            <div class="score-gauge">
                <div class="score-value {score_color(100 - gap_score)}">{gap_score}%</div>
                <div class="score-label">Skill Gap Score</div>
            </div>""", unsafe_allow_html=True)
            sm3.markdown(f"""
            <div class="score-gauge">
                <div class="score-value" style="color:{'#22c55e' if gap_severity in ('Low',) else '#eab308' if gap_severity in ('Medium',) else '#ef4444'};">{gap_severity}</div>
                <div class="score-label">Gap Severity</div>
            </div>""", unsafe_allow_html=True)
            sm4.markdown(f"""
            <div class="score-gauge">
                <div class="score-value {score_color(mossga_score)}">{mossga_score}</div>
                <div class="score-label">MoSSGA Score</div>
            </div>""", unsafe_allow_html=True)

            # Matched skills
            matched_skills = match.get("matched_skills", [])
            if matched_skills:
                st.markdown("### ✅ Semantically Matched Skills")
                for m in matched_skills:
                    sim = m["similarity"]
                    match_type = m.get("match_type", "semantic")
                    sim_color = '#22c55e' if sim >= 0.8 else ('#eab308' if sim >= 0.6 else '#94a3b8')
                    type_badge = '🎯 Exact' if match_type == 'exact' else '🧠 Semantic'
                    st.markdown(f"""
                    <div style="background:rgba(30,41,59,0.5); padding:10px 16px; border-radius:10px; margin:4px 0; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <b style="color:#a5b4fc;">{m['jd_skill'].title()}</b>
                            <span style="color:#64748b; margin:0 8px;">→</span>
                            <span style="color:#e2e8f0;">{m['candidate_skill'].title()}</span>
                            <span style="color:#94a3b8; font-size:0.8rem; margin-left:8px;">{type_badge}</span>
                        </div>
                        <span style="color:{sim_color}; font-weight:600;">{sim:.1%}</span>
                    </div>""", unsafe_allow_html=True)

            # Missing skills
            missing_detailed = gap.get("missing_with_severity", [])
            if missing_detailed:
                st.markdown("### ❌ Missing Skills (by priority)")
                for item in missing_detailed:
                    sev = item["severity"]
                    sev_color = '#ef4444' if sev == 'Critical' else ('#eab308' if sev == 'Important' else '#94a3b8')
                    st.markdown(f"""
                    <div class="glass-card" style="padding:12px 16px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <b style="color:#fca5a5;">{item['skill'].title()}</b>
                            <span style="color:{sev_color}; font-weight:600; font-size:0.85rem;">{sev} (Priority: {item['importance']})</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # Semantic similarity heatmap
            sim_matrix = match.get("similarity_matrix", [])
            jd_resolved = match.get("jd_resolved", [])
            cand_resolved = match.get("candidate_resolved", [])
            if sim_matrix and len(sim_matrix) <= 20 and len(sim_matrix[0]) <= 20:
                st.markdown("### 🔥 Semantic Similarity Heatmap")
                import numpy as np
                fig_heat = go.Figure(data=go.Heatmap(
                    z=sim_matrix,
                    x=[s.title() for s in cand_resolved[:20]],
                    y=[s.title() for s in jd_resolved[:20]],
                    colorscale='Viridis',
                    text=[[f"{v:.2f}" for v in row] for row in sim_matrix],
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                ))
                fig_heat.update_layout(
                    template='plotly_dark', height=500,
                    title='JD Skills vs Candidate Skills — Semantic Similarity',
                    xaxis_title='Candidate Skills', yaxis_title='JD Required Skills',
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            # Recommendations
            recs = mossga_report.get("recommendations", {})
            if recs:
                skill_recs = recs.get("skill_recommendations", [])
                if skill_recs:
                    st.markdown("### 📚 Personalized Learning Recommendations")
                    for rec in skill_recs:
                        imp_color = '#ef4444' if rec['importance'] == 'Critical' else ('#eab308' if rec['importance'] == 'Important' else '#94a3b8')
                        courses_html = ""
                        for course in rec.get("courses", []):
                            courses_html += f'<a href="{course.get("url", "#")}" style="color:#60a5fa; text-decoration:none;">{course["title"]}</a> ({course["platform"]}) · '

                        st.markdown(f"""
                        <div class="glass-card" style="padding:14px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b style="color:#a5b4fc; font-size:1.05rem;">{rec['skill'].title()}</b>
                                <span style="color:{imp_color}; font-weight:600; font-size:0.85rem;">{rec['importance']}</span>
                            </div>
                            <p style="color:#94a3b8; font-size:0.9rem; margin:4px 0;">{rec['action']}</p>
                            <span style="font-size:0.85rem;">📖 {courses_html.rstrip(' · ')}</span>
                        </div>""", unsafe_allow_html=True)

                # Career steps
                career_steps = recs.get("career_steps", [])
                if career_steps:
                    st.markdown("### 🚀 Career Improvement Steps")
                    for step in career_steps:
                        st.markdown(step)

                # Improvement areas
                impr_areas = recs.get("improvement_areas", [])
                if impr_areas:
                    st.markdown("### 🎯 Strategic Improvement Areas")
                    for area in impr_areas:
                        st.markdown(area)

        elif mossga_report and mossga_report.get("fusion_result"):
            st.info("📌 Paste a Job Description to see semantic skill gap analysis. Currently showing your fused skill profile.")
            fused = mossga_report["fusion_result"].get("fused_skills", [])
            if fused:
                skill_html = " ".join(
                    [f'<span style="background:rgba(99,102,241,0.2); color:#a5b4fc; padding:4px 12px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; border:1px solid rgba(99,102,241,0.3);">{s.title()}</span>'
                     for s in fused]
                )
                st.markdown(f'<div style="margin:8px 0;">{skill_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <p style="color:#94a3b8;">🎯 Upload a resume or enter a GitHub profile, plus a job description, to see the MoSSGA semantic skill gap analysis with personalized recommendations.</p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 5: Cognitive Skill Graph ---
    with tab5:
        st.markdown("## 🕸️ Cognitive Skill Graph")
        st.markdown("*Novel Research Contribution: Mapping skill dependencies and predicting your optimal next skill.*")

        for name, n_json, res_data in neurohire_jsons:
            skills_for_graph = n_json["skills"]
            if skills_for_graph:
                graph_data = build_cognitive_skill_graph(skills_for_graph)

                # Metrics
                g1, g2, g3 = st.columns(3)
                g1.metric("Skills in Graph", len(graph_data["nodes"]))
                g2.metric("Connections", len(graph_data["edges"]))
                g3.metric("Avg Complexity", graph_data["avg_complexity"])

                # Skill Clusters
                st.markdown("### 🧩 Skill Domain Clusters")
                clusters = graph_data["skill_clusters"]
                cluster_cols = st.columns(min(len(clusters), 3)) if clusters else []
                for idx, (cluster_name, cluster_info) in enumerate(clusters.items()):
                    with cluster_cols[idx % len(cluster_cols)]:
                        pct = cluster_info['percentage']
                        color = '#22c55e' if pct > 40 else ('#eab308' if pct > 20 else '#94a3b8')
                        st.markdown(f"""
                        <div class="glass-card">
                            <b style="font-size:1.1rem;">{cluster_name}</b><br/>
                            <span style="color:{color}; font-weight:600;">{cluster_info['coverage']} skills ({pct}%)</span><br/>
                            <span style="color:#94a3b8; font-size:0.85rem;">{', '.join(cluster_info['skills'])}</span>
                        </div>""", unsafe_allow_html=True)

                # Next Best Skills to Learn (AI Prediction)
                st.markdown("### 🎯 AI-Predicted Next Best Skills")
                st.markdown("*Based on skill dependency graph analysis and prerequisite readiness scoring.*")
                next_skills = graph_data["next_best_skills"]
                if next_skills:
                    for ns in next_skills[:6]:
                        readiness = ns['readiness_score']
                        bar_color = '#22c55e' if readiness >= 70 else ('#eab308' if readiness >= 40 else '#ef4444')
                        st.markdown(f"""
                        <div class="glass-card" style="padding:16px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b style="color:#a5b4fc; font-size:1.05rem;">{ns['skill']}</b>
                                <span style="color:{bar_color}; font-weight:700;">Readiness: {readiness}%</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.3); border-radius:8px; height:8px; margin:8px 0;">
                                <div style="background:{bar_color}; width:{readiness}%; height:100%; border-radius:8px;"></div>
                            </div>
                            <span style="color:#94a3b8; font-size:0.85rem;">Prerequisites: {ns['prerequisites_met']} met | Unlocked by: {', '.join(ns['unlocked_by']) if ns['unlocked_by'] else 'Foundation skill'}</span><br/>
                            <span style="color:#64748b; font-size:0.8rem;">{ns['reason']}</span>
                        </div>""", unsafe_allow_html=True)

                # Interactive Skill Network Plot
                st.markdown("### 🔗 Skill Relationship Network")
                import networkx as nx
                try:
                    G = nx.DiGraph()
                    for node in graph_data["nodes"]:
                        G.add_node(node["id"], label=node["label"], size=node["size"])
                    for edge in graph_data["edges"]:
                        G.add_edge(edge["from"], edge["to"], type=edge["type"])

                    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                    # Edges
                    edge_x, edge_y = [], []
                    for e in G.edges():
                        x0, y0 = pos[e[0]]
                        x1, y1 = pos[e[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                            line=dict(width=1, color='rgba(148,163,184,0.3)'),
                                            hoverinfo='none')

                    # Nodes
                    node_x = [pos[n][0] for n in G.nodes()]
                    node_y = [pos[n][1] for n in G.nodes()]
                    node_labels = [G.nodes[n].get('label', n) for n in G.nodes()]
                    node_sizes = [G.nodes[n].get('size', 15) for n in G.nodes()]
                    node_colors = ['#6366f1' if n in {s.lower() for s in skills_for_graph} else '#374151' for n in G.nodes()]

                    node_trace = go.Scatter(
                        x=node_x, y=node_y, mode='markers+text',
                        text=node_labels, textposition='top center',
                        textfont=dict(size=10, color='#e2e8f0'),
                        marker=dict(size=node_sizes, color=node_colors,
                                    line=dict(width=2, color='rgba(99,102,241,0.5)')),
                        hoverinfo='text',
                    )

                    fig_graph = go.Figure(data=[edge_trace, node_trace])
                    fig_graph.update_layout(
                        template='plotly_dark', showlegend=False, height=500,
                        title='Cognitive Skill Dependency Network',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(10, 14, 23, 0.9)',
                    )
                    st.plotly_chart(fig_graph, use_container_width=True)
                except ImportError:
                    st.info("Install `networkx` for the interactive skill graph visualization.")
            else:
                st.warning("No skills detected. Enter skills in the sidebar to see the Cognitive Skill Graph.")

    # --- TAB 6: Career Trajectory Prediction ---
    with tab6:
        st.markdown("## 📈 Career Trajectory Prediction")
        st.markdown("*Novel Research Contribution: ML-based career path modeling with timeline milestones.*")

        for name, n_json, res_data in neurohire_jsons:
            skills_for_traj = n_json["skills"]
            traj_role = target_role if target_role else role_context

            if skills_for_traj:
                traj_data = predict_career_trajectory(skills_for_traj, traj_role)

                st.markdown(f"### 🛤️ Predicted Path: {traj_data['trajectory_name']}")
                st.info(traj_data["prediction"])

                # Timeline visualization
                milestones = traj_data["milestones"]
                for i, ms in enumerate(milestones):
                    is_current = (i == traj_data["current_position"])
                    border_color = '#22c55e' if ms['readiness'] >= 70 else ('#eab308' if ms['readiness'] >= 40 else '#ef4444')
                    bg = 'rgba(99, 102, 241, 0.15)' if is_current else 'rgba(30, 41, 59, 0.5)'
                    marker = '🔵' if is_current else ('✅' if ms['readiness'] >= 70 else '⬜')

                    have_html = ' '.join(
                        [f'<span style="background:rgba(34,197,94,0.2); color:#86efac; padding:2px 8px; border-radius:12px; font-size:0.8rem; margin:2px; display:inline-block;">{s}</span>'
                         for s in ms['skills_you_have']]
                    ) if ms['skills_you_have'] else '<span style="color:#64748b;">None yet</span>'

                    need_html = ' '.join(
                        [f'<span style="background:rgba(239,68,68,0.15); color:#fca5a5; padding:2px 8px; border-radius:12px; font-size:0.8rem; margin:2px; display:inline-block;">{s}</span>'
                         for s in ms['skills_to_learn']]
                    ) if ms['skills_to_learn'] else '<span style="color:#22c55e;">All covered! ✅</span>'

                    st.markdown(f"""
                    <div style="background:{bg}; border-left:4px solid {border_color}; border-radius:0 12px 12px 0; padding:16px 20px; margin:8px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.1rem;"><b>{marker} {ms['role']}</b></span>
                            <span style="color:{border_color}; font-weight:600;">{ms['status']} ({ms['readiness']}%)</span>
                        </div>
                        <span style="color:#94a3b8;">⏱ ~{ms['estimated_months']} months from start</span><br/>
                        <span style="color:#64748b; font-size:0.85rem;">✅ Have: {have_html}</span><br/>
                        <span style="color:#64748b; font-size:0.85rem;">📚 Need: {need_html}</span>
                    </div>""", unsafe_allow_html=True)

                # Readiness chart
                import plotly.graph_objects as go
                fig_traj = go.Figure(
                    data=go.Bar(
                        x=[ms['role'] for ms in milestones],
                        y=[ms['readiness'] for ms in milestones],
                        marker_color=[('#22c55e' if ms['readiness'] >= 70 else '#eab308' if ms['readiness'] >= 40 else '#ef4444') for ms in milestones],
                        text=[f"{ms['readiness']}%" for ms in milestones],
                        textposition='outside',
                    )
                )
                fig_traj.update_layout(
                    template='plotly_dark', title='Milestone Readiness (%)',
                    yaxis=dict(range=[0, 110]), height=350,
                )
                st.plotly_chart(fig_traj, use_container_width=True)

                # Skill authenticity (only for full analysis with resume text)
                if res_data and res_data.get("Raw Text"):
                    st.markdown("---")
                    st.markdown("### 🔍 Skill Authenticity Detection")
                    st.markdown("*Novel: Validates if claimed skills have supporting evidence in the resume.*")
                    auth_report = detect_skill_authenticity(res_data["Raw Text"], skills_for_traj)

                    st.metric("Overall Authenticity Score", f"{auth_report['overall_authenticity']}%")
                    st.markdown(auth_report["summary"])

                    for sr in auth_report["skill_reports"]:
                        vcolor = '#22c55e' if sr['authenticity_score'] >= 60 else ('#eab308' if sr['authenticity_score'] >= 40 else '#ef4444')
                        ev = sr['evidence']
                        badges = []
                        if ev['project_evidence']: badges.append('📦 Project')
                        if ev['experience_evidence']: badges.append('💼 Experience')
                        if ev['certification_evidence']: badges.append('📜 Cert')
                        if ev['metric_evidence']: badges.append('📊 Metrics')
                        badge_str = ' '.join(badges) if badges else '❌ No evidence'

                        st.markdown(f"""
                        <div style="background:rgba(30,41,59,0.5); padding:10px 16px; border-radius:10px; margin:4px 0; display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <b style="color:#e2e8f0;">{sr['skill']}</b>
                                <span style="color:#94a3b8; font-size:0.8rem; margin-left:12px;">{badge_str}</span>
                            </div>
                            <div>
                                <span style="color:{vcolor}; font-weight:600;">{sr['verdict']}</span>
                                <span style="color:#94a3b8; margin-left:8px;">{sr['authenticity_score']}%</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.warning("Enter skills to see career trajectory prediction.")

    # --- TAB 7: RJAS Ranking ---
    with tab7:
        st.markdown("## 📊 Intelligent Ranking (RJAS)")
        if has_resumes and has_jd and results:
            df = pd.DataFrame([r for r in results if r.get("Processing Status") == "Success"])
            st.markdown(f"**Role Context:** `{role_context}` &nbsp;|&nbsp; **Alpha:** `{alpha_input}`")

            if not df.empty and "RJAS" in df.columns:
                display_cols = ["Name", "RJAS", "SBERT Score", "NLP Score"]
                df_ranked = df[display_cols].sort_values(by="RJAS", ascending=False)
                st.dataframe(df_ranked.style.highlight_max(axis=0), use_container_width=True)

                fig = px.bar(
                    df_ranked, x="Name", y="RJAS", color="RJAS",
                    color_continuous_scale="Viridis",
                    title="Resume-Job Alignment Score (RJAS) Ranking",
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Comparison chart
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="SBERT", x=df_ranked["Name"], y=df_ranked["SBERT Score"], marker_color="#6366f1"))
                fig2.add_trace(go.Bar(name="TF-IDF", x=df_ranked["Name"], y=df_ranked["NLP Score"], marker_color="#f59e0b"))
                fig2.update_layout(barmode="group", template="plotly_dark", title="SBERT vs TF-IDF Comparison", height=350)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <p style="color:#94a3b8;">📌 RJAS Ranking requires both <b>resumes</b> and a <b>job description</b> to compute semantic match scores.</p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 8: Explainability (XAI) ---
    with tab8:
        st.markdown("## 📢 Explainable AI (XAI)")
        st.markdown("*Every score has a reason. Every recommendation has an explanation.*")
        if has_resumes and has_jd and results:
            df = pd.DataFrame([r for r in results if r.get("Processing Status") == "Success"])
            selected_candidate = st.selectbox("Select Candidate to Explain", df["Name"].tolist())

            if selected_candidate:
                cand = df[df["Name"] == selected_candidate].iloc[0]
                rjas_val = cand.get("RJAS", 0)

                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="color:#a5b4fc;">Why did {cand['Name']} score {rjas_val:.2f}?</h3>
                    <ul style="color:#cbd5e1;">
                        <li><b>Semantic Match (SBERT):</b> {cand.get('SBERT Score', 0):.4f} — Measures how closely the resume's meaning aligns with the JD, not just keywords.</li>
                        <li><b>TF-IDF Score:</b> {cand.get('NLP Score', 0):.4f} — Traditional statistical text similarity.</li>
                        <li><b>Skills Matched:</b> {len(cand.get('Matched Skills', []))} out of {len(jd_skills)} JD skills.</li>
                        <li><b>Bias Entities:</b> {len(cand.get('Detected Bias', []))} detected {'(penalty applied)' if cand.get('Detected Bias') else '(clean — no penalty)'}.</li>
                    </ul>
                    <p style="color:#94a3b8; margin-top:12px;">The RJAS formula blends Semantic + Skill + Experience + Education scores using RL-optimized weights, then applies the fairness-accuracy trade-off (α={alpha_input}).</p>
                </div>
                """, unsafe_allow_html=True)

                # Radar chart
                breakdown_scores = cand.get("Breakdown", {})
                if breakdown_scores:
                    categories = ["Semantic", "Skills", "Experience", "Education"]
                    radar_values = [
                        breakdown_scores.get("Semantic", 0) / 100,
                        breakdown_scores.get("Skills", 0) / 100,
                        breakdown_scores.get("Experience", 0) / 100,
                        breakdown_scores.get("Education", 0) / 100,
                    ]
                    radar_values.append(radar_values[0])
                    categories.append(categories[0])

                    fig_radar = go.Figure(
                        data=go.Scatterpolar(
                            r=radar_values, theta=categories, fill="toself",
                            line=dict(color="#8b5cf6"),
                            fillcolor="rgba(139, 92, 246, 0.2)",
                        )
                    )
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(15, 23, 42, 0.8)",
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(148, 163, 184, 0.2)"),
                            angularaxis=dict(gridcolor="rgba(148, 163, 184, 0.2)"),
                        ),
                        template="plotly_dark", title="Score Breakdown Radar", height=400,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # Deep XAI from cognitive_engine
                deep_xai = generate_deep_explanation(cand.to_dict(), jd_skills, alpha_input)
                st.markdown("### 🔬 Factor-by-Factor Deep Analysis")
                for factor in deep_xai["factors"]:
                    st.markdown(factor)
                st.markdown(deep_xai["verdict"])

                st.markdown("### 🎯 Priority Improvement Areas")
                for imp in deep_xai["improvement_priority"]:
                    imp_color = '#ef4444' if imp['score'] < 40 else '#eab308'
                    st.markdown(f"""
                    <div style="background:rgba(30,41,59,0.5); padding:10px 16px; border-radius:10px; margin:4px 0;">
                        <b style="color:{imp_color};">{imp['area']}</b> — Score: {imp['score']}%<br/>
                        <span style="color:#94a3b8;">🔧 {imp['action']}</span>
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="glass-card">
                <p style="color:#94a3b8;">📌 Upload resumes and a JD to see detailed XAI explanations with radar charts and deep factor analysis.</p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 9: Research Lab ---
    with tab9:
        st.markdown("## 🧪 Research Lab — Comparative Study")
        if has_resumes and has_jd and results:
            st.info("📊 Comparing: Traditional Keyword (Jaccard) vs TF-IDF vs Novel RJAS with SBERT")

            # Algorithm comparison
            comparison_data = []
            for res in results:
                if res.get("Processing Status") == "Success":
                    comp = compare_algorithms(res["Raw Text"], jd_input, res.get("NLP Score", 0), res.get("RJAS", 0))
                    comp["Name"] = res["Name"]
                    comparison_data.append(comp)

            study_df = pd.DataFrame(comparison_data)
            if not study_df.empty:
                st.dataframe(study_df, use_container_width=True)

                melted = study_df.melt(id_vars=["Name"], var_name="Algorithm", value_name="Score")
                fig_comp = px.bar(melted, x="Name", y="Score", color="Algorithm", barmode="group",
                                  title="Algorithm Efficiency — Jaccard vs TF-IDF vs RJAS",
                                  color_discrete_sequence=["#ef4444", "#f59e0b", "#22c55e"])
                fig_comp.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_comp, use_container_width=True)

            # Statistical Validation
            st.markdown("---")
            st.markdown("### 📈 Statistical Validation")
            df_stat = pd.DataFrame([r for r in results if r.get("Processing Status") == "Success"])
            stats_result = calculate_statistics(df_stat)
            if stats_result:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("RJAS Mean (μ)", f"{stats_result.get('RJAS Mean', 0):.2f}",
                          f"σ={stats_result.get('RJAS Std', 0):.2f}")
                s2.metric("NLP Mean (μ)", f"{stats_result.get('NLP Mean', 0):.2f}",
                          f"σ={stats_result.get('NLP Std', 0):.2f}")
                t_val = stats_result.get("T-Statistic")
                p_val = stats_result.get("P-Value")
                if t_val is not None:
                    s3.metric("T-Statistic", f"{t_val:.4f}")
                    sig = "✅ Yes" if stats_result.get("Significant") else "❌ No"
                    s4.metric("Significant (p<0.05)?", sig)

            # Pareto Frontier
            st.markdown("---")
            st.markdown("### ⚖️ Pareto Frontier (Accuracy vs Fairness)")
            pareto_df = generate_pareto_frontier(results, current_weights)
            if not pareto_df.empty:
                fig_pareto = px.scatter(
                    pareto_df, x="Global Fairness", y="Global Accuracy",
                    color="Alpha (Preference)", hover_data=["Msg"],
                    title="Pareto Trade-off: Accuracy vs Fairness",
                    color_continuous_scale="Plasma",
                )
                fig_pareto.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_pareto, use_container_width=True)

            # Fairness Analysis
            st.markdown("---")
            st.markdown("### 🛡️ Bias-Aware Fairness Constraints")
            fairness_metrics = analyze_fairness(df_stat)
            if fairness_metrics:
                f1, f2, f3 = st.columns(3)
                f1.metric("Avg RJAS (Bias Detected)", fairness_metrics["Avg Score (Bias Detected)"])
                f2.metric("Avg RJAS (Clean)", fairness_metrics["Avg Score (Clean)"])
                f3.metric("Fairness Gap", fairness_metrics["Gap"], delta_color="inverse")
                if fairness_metrics["Gap"] > 10:
                    st.warning("⚠️ Significant fairness gap detected! Bias penalties are actively working.")
                else:
                    st.success("✅ The ranking appears balanced and fair.")
        else:
            st.markdown("""
            <div class="glass-card">
                <p style="color:#94a3b8;">📌 Upload resumes and a JD to run the comparative research study.</p>
            </div>""", unsafe_allow_html=True)

    # --- TAB 10: RL Feedback Loop ---
    with tab10:
        st.markdown("## 🔄 Reinforcement Learning Feedback Loop")
        st.markdown("Teach NeuroHire! Your hiring decisions update the RL agent's policy in real-time.")

        if has_resumes and has_jd and results:
            df_rl = pd.DataFrame([r for r in results if r.get("Processing Status") == "Success"])
            selected_hire = st.selectbox("Select a candidate to hire", df_rl["Name"].tolist(), key="fb_select")

            if st.button("✅ Confirm Hire — Update RL Policy"):
                hired_data = df_rl[df_rl["Name"] == selected_hire].iloc[0]
                new_weights = st.session_state["adaptive_engine"].update_policy(hired_data, reward=1.0)
                st.balloons()
                st.success(f"RL Agent updated! New weights for '{role_context}': {new_weights}")

            st.markdown("---")
            st.markdown("### 🔬 RL Convergence Simulation")
            if st.button("▶ Run Simulation (100 iterations)"):
                with st.spinner("Simulating user feedback loop..."):
                    sim_data = simulate_rl_convergence(iterations=100, role=role_context)
                    st.success("✅ Simulation complete!")

                    weight_cols = [c for c in sim_data.columns if c not in ["Iteration", "Cumulative Reward"]]
                    fig_rl = px.line(sim_data, x="Iteration", y=weight_cols,
                                     title="RL Weight Evolution Over Time")
                    fig_rl.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_rl, use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <p style="color:#94a3b8;">📌 Upload resumes and a JD to enable the RL feedback loop.</p>
            </div>""", unsafe_allow_html=True)
