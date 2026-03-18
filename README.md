
# Multi-Modal Semantic Skill Gap Analysis (MoSSGA) System

A **Research-Grade** Multi-Modal AI System for semantic skill gap analysis — powered by **Sentence-BERT**, **Multi-Modal Fusion**, and an **Intelligent Recommendation Engine**.

![System Architecture](https://img.shields.io/badge/AI-SBERT%20%2B%20Multi--Modal%20Fusion-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

## 🧠 System Architecture

```
User Interface
  Upload Resume | Enter GitHub | Input Job Role / JD
          │
  ┌───────▼──────────────────────────────────────────────┐
  │     MULTI-MODAL DATA ACQUISITION LAYER               │
  │  Resume (PDF/Text)  │  GitHub (API)  │  JD (Skills)  │
  └───────┬─────────────┴────────┬───────┴───────┬───────┘
          │                      │               │
  ┌───────▼──────────┐           │     ┌─────────▼────────────┐
  │ DATA PREPROCESSING│           │     │ SIMILARITY & MATCHING│
  │ - Text Cleaning   │           │     │ - Cosine Similarity  │
  │ - Tokenization    │           │     │ - Skill Match Score  │
  │ - Normalization   │           │     └─────────┬────────────┘
  └───────┬──────────┘           │               │
          │                      │     ┌─────────▼────────────┐
  ┌───────▼──────────────────┐   │     │ SKILL GAP ANALYSIS   │
  │ MULTI-MODAL FEATURE      │   │     │ (CORE NOVELTY)       │
  │ EXTRACTION MODULE        │   │     │ - Missing Skills      │
  │ Resume → Skills, Exp,    │   │     │ - Weak/Partial Skills │
  │          Projects        │   │     │ - Language Usage       │
  │ GitHub → Languages,      │   │     │ - Activity Check      │
  │          Commits,        │   │     └─────────┬────────────┘
  │          Repo Complexity │   │               │
  │ JD → Required Skills     │   │     ┌─────────▼────────────┐
  └───────┬──────────────────┘   │     │ INTELLIGENT          │
          │                      │     │ RECOMMENDATION ENGINE │
  ┌───────▼──────────────────┐   │     │ - Skill Improvement   │
  │ SEMANTIC REPRESENTATION  │   │     │ - Course/Project Recs │
  │ LAYER (TRANSFORMERS)     │   │     │ - Career Path Guidance│
  │ - Sentence-BERT / BERT   │   │     └─────────┬────────────┘
  │ - Contextual Embeddings  │   │               │
  │ - Skill Normalization    │   │     ┌─────────▼────────────┐
  │   (ML ≈ Machine Learning)│   │     │ OUTPUT VISUALIZATION │
  └───────┬──────────────────┘   │     │ - Match Score (%)     │
          │                      │     │ - Matched Skills      │
  ┌───────▼──────────────────┐   │     │ - Missing Skills      │
  │ VECTOR SPACE MODEL &     │───┘     │ - Recommendations     │
  │ SKILL MAPPING            │         └──────────────────────┘
  │ - Candidate Vector Space │
  │ - Skill Matching Score   │
  └──────────────────────────┘
```

## 🚀 Key Features

### MoSSGA — Multi-Modal Semantic Skill Gap Analysis
- **Multi-Modal Data Acquisition**: Accepts resumes (PDF/DOCX), GitHub profiles (API), and job descriptions
- **Semantic Skill Matching**: SBERT-powered matching recognizes "ML" ≈ "Machine Learning", "JS" ≈ "JavaScript" — not just keywords
- **Multi-Modal Fusion**: Combines resume + GitHub data with confidence scoring. Skills validated by both sources get highest confidence
- **Skill Gap Detection**: Identifies missing and weak skills with severity levels (Critical → Nice-to-have)
- **Intelligent Recommendations**: Generates personalized course recommendations, skill improvement suggestions, and career path guidance

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
4. **Run Analysis**: Click "🚀 Run MoSSGA Analysis".
5. **Explore 5 Tabs**:
   | Tab | Description |
   |-----|-------------|
   | 📊 Analysis Results | Skills detected, match scores, extracted resume sections |
   | 🐙 GitHub Analysis | Repository languages, technologies, project complexity |
   | 🔀 Multi-Modal Fusion | Resume vs GitHub skill comparison with confidence scores |
   | 🎯 MoSSGA Skill Gap | Semantic matching heatmap, missing skills with severity |
   | 📋 Recommendations | Skill improvement suggestions, courses, career guidance |

## 📂 Project Structure

```
├── app.py                          # Main Streamlit Dashboard (5 tabs)
├── src/
│   ├── parser.py                   # PDF/DOCX text extraction
│   ├── preprocessing.py            # NLP text cleaning, skill extraction (70+ skills)
│   ├── matcher.py                  # TF-IDF & SBERT similarity
│   ├── semantic_skill_matcher.py   # SBERT semantic skill matching engine
│   ├── github_analyzer.py          # GitHub API integration & skill extraction
│   ├── multimodal_fusion.py        # Resume + GitHub data fusion
│   ├── mossga_engine.py            # MoSSGA orchestrator pipeline
│   └── career_intelligence.py      # Intelligent recommendation engine
├── verify_mossga.py                # Pipeline verification test
├── requirements.txt
└── README.md
```

## ✅ Verification

```bash
# Verify MoSSGA pipeline (semantic matching + fusion + gap analysis)
python verify_mossga.py
```

## 🔬 Technologies Used

| Category | Technologies |
|----------|-------------|
| **NLP** | SpaCy, Sentence-BERT (all-MiniLM-L6-v2), TF-IDF |
| **ML** | Scikit-learn, NumPy |
| **Visualization** | Plotly, Streamlit |
| **Data** | Pandas, PDFMiner, python-docx |
| **API** | GitHub REST API v3 |

---
*Developed by Suryansh Agrawal*
