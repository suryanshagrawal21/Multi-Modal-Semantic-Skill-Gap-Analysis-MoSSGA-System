
# Multi-Modal Semantic Skill Gap Analysis (MoSSGA) System

A **Research-Grade** Multi-Modal AI System for intelligent resume screening, GitHub profile analysis, semantic skill matching, and personalized career intelligence — powered by **Sentence-BERT**, **Reinforcement Learning**, and **Multi-Modal Fusion**.

![System Architecture](https://img.shields.io/badge/AI-SBERT%20%2B%20RL%20%2B%20MoSSGA-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

## 🧠 System Architecture

```
Input Layer                     Feature Extraction           Intelligence Engine
┌───────────────┐              ┌──────────────────┐         ┌──────────────────────┐
│ Resume (PDF)  │──┐           │ Skills            │         │ Semantic Embedding   │
│ GitHub Profile│──┼──► NLP ──►│ Experience        │──► Fusion ──► (Sentence-BERT) │
│ Job Desc      │──┘  Parsing  │ Projects          │         │ Skill Gap Detection  │
└───────────────┘              └──────────────────┘         │ Recommendation Engine│
                                                            └──────────┬───────────┘
                                                                       │
                                                            ┌──────────▼───────────┐
                                                            │   Output Dashboard   │
                                                            │  (Streamlit - 10 Tabs)│
                                                            └──────────────────────┘
```

## 🚀 Key Features

### MoSSGA — Multi-Modal Semantic Skill Gap Analysis
- **GitHub Profile Analysis**: Fetches repos via GitHub API, extracts languages, technologies, and project complexity scores.
- **Semantic Skill Matching**: SBERT-powered matching recognizes "ML" ≈ "Machine Learning", "JS" ≈ "JavaScript" — not just keywords.
- **Multi-Modal Fusion**: Combines resume + GitHub data with confidence scoring. Skills validated by both sources get highest confidence.
- **Skill Gap Detection**: Identifies missing skills with severity levels (Critical → Nice-to-have) and generates course recommendations.

### NeuroHire — AI Career Intelligence
- **Multi-Objective RJAS Score**: Balances predictive accuracy with algorithmic fairness.
  $$ RJAS = \alpha \cdot (SBERT + Skills + Exp + Edu) + (1 - \alpha) \cdot (1 - P_{bias}) $$
- **Cognitive Skill Graph**: Maps skill dependencies and predicts next-best skills to learn.
- **Career Trajectory Prediction**: ML-based career path modeling with timeline milestones.
- **Skill Authenticity Detection**: Validates resume claims against evidence patterns.

### Research Lab & RL Agent
- **Reinforcement Learning Ranking**: Contextual Bandit adapts scoring weights from recruiter feedback.
- **Statistical Validation**: Paired T-Tests, Pareto frontier analysis (Accuracy vs Fairness).
- **Explainable AI (XAI)**: Factor-by-factor deep reasoning for every score.

## 🛠️ Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/suryanshagrawal21/Intelligent_AI_Resume_Screening_System.git
    cd Intelligent_AI_Resume_Screening_System
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 📊 Usage Guide

1. **Upload Resumes**: Drag & drop PDF/DOCX files in the sidebar.
2. **Enter GitHub Profile**: Paste a GitHub URL or username.
3. **Set Job Description**: Paste the JD text.
4. **Run Analysis**: Click "🚀 Run MoSSGA + NeuroHire Analysis".
5. **Explore 10 Tabs**:
   | Tab | Description |
   |-----|-------------|
   | 🧠 NeuroHire Insights | Full candidate analysis with strengths, weaknesses, roadmap |
   | 🐙 GitHub Analysis | Repository languages, technologies, project complexity |
   | 🔀 Multi-Modal Fusion | Resume vs GitHub skill comparison with confidence scores |
   | 🎯 MoSSGA Skill Gap | Semantic matching heatmap, missing skills, course recommendations |
   | 🕸️ Cognitive Skill Graph | Interactive skill dependency network |
   | 📈 Career Trajectory | ML-predicted career path with readiness scores |
   | 📊 RJAS Ranking | Multi-resume ranking with composite scores |
   | 📢 Explainability (XAI) | Factor-by-factor score reasoning with radar charts |
   | 🧪 Research Lab | Algorithm comparison, statistical validation, Pareto frontier |
   | 🔄 RL Feedback | Teach the AI by hiring/rejecting candidates |

## 📂 Project Structure

```
├── app.py                          # Main Streamlit Dashboard (10 tabs)
├── src/
│   ├── github_analyzer.py          # GitHub API integration & skill extraction
│   ├── semantic_skill_matcher.py   # SBERT semantic skill matching engine
│   ├── multimodal_fusion.py        # Resume + GitHub data fusion
│   ├── mossga_engine.py            # MoSSGA orchestrator pipeline
│   ├── parser.py                   # PDF/DOCX text extraction
│   ├── preprocessing.py            # NLP text cleaning, skill extraction (70+ skills)
│   ├── matcher.py                  # TF-IDF & SBERT similarity
│   ├── scoring.py                  # RJAS composite scoring
│   ├── rjas_metric.py              # Multi-objective RJAS formula
│   ├── adaptive_engine.py          # RL Contextual Bandit agent
│   ├── career_intelligence.py      # Career guidance & recommendations
│   ├── cognitive_engine.py         # Skill graph, trajectory, authenticity
│   ├── research_lab.py             # Statistical tests & Pareto analysis
│   └── experiment.py               # Comparative experiment runner
├── requirements.txt
└── README.md
```

## ✅ Verification

```bash
# Verify RJAS & RL Agent
python verify_framework.py

# Verify Research Components
python verify_research.py
```

## 🔬 Technologies Used

| Category | Technologies |
|----------|-------------|
| **NLP** | SpaCy, Sentence-BERT (all-MiniLM-L6-v2), TF-IDF |
| **ML** | Scikit-learn, NumPy, SciPy |
| **Visualization** | Plotly, Streamlit, NetworkX |
| **Data** | Pandas, PDFMiner, python-docx |
| **API** | GitHub REST API v3 |
| **Fairness** | Bias-aware scoring, Pareto optimization |

---
*Developed by Suryansh Agrawal*
