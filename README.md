# Hybrid Semantic and Multi-Model Approach for AI-Driven Skill Gap Analysis in Workforce Informatics

A **Research-Grade** AI System for automated, high-precision semantic skill gap analysis. This framework transcends traditional keyword matching by introducing a robust **Hybrid Architecture** that fuses Large Language Model (LLM) embeddings, Knowledge Graph ontology, and Machine Learning predictive modeling.

![System Architecture](https://img.shields.io/badge/AI-Hybrid%20Semantic%20%2B%20ML-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

## Abstract & Methodology

The objective of this system is to effectively analyze skill gaps in heterogeneous and unstructured workforce data (e.g., resumes, GitHub profiles, and job descriptions) and support intelligent decision-making in workforce informatics. 

The methodology relies on a tripartite hybrid scoring mechanism:
1. **Semantic Layer (SBERT)**: Encodes skills into high-dimensional vector space using `Sentence-Transformers` to compute cosine similarity, addressing the vocabulary mismatch problem (e.g., identifying that "ML" is "Machine Learning").
2. **Knowledge Graph Layer (NetworkX/Neo4j)**: A directed acyclic graph topology maps skill prerequisites, hierarchical dependencies, and related domains. E.g., learning "Deep Learning" requires "Machine Learning" and "Python".
3. **Machine Learning Layer (XGBoost/RF)**: Predicts skill gap severity non-linearly based on semantic scores, graph proximity, and missing skill counts.

### Final Hybrid Score Formula
$$ \text{Final Score} = (0.4 \times \text{Semantic Score}) + (0.3 \times \text{Graph Score}) + (0.3 \times \text{ML Prediction Component}) $$

---

## 🏗️ System Architecture

```text
       UNSTRUCTURED DATA INGESTION
   (PDF/DOCX Resumes, GitHub API, Raw JDs)
                 │
                 ▼
       MULTI-MODAL FEATURE EXTRACTION
   (NLP Preprocessing, Entity & Skill Extraction)
                 │
                 ▼
   ┌─────────────────────────────────────┐
   │        HYBRID AI ARCHITECTURE       │
   │                                     │
   │  1. Semantic Engine (SBERT)         │
   │  2. Knowledge Graph (NetworkX)      │
   │  3. Multi-Model Integration (XGB)   │
   └─────────────────┬───────────────────┘
                     ▼
      EXPLAINABLE AI & WORKFORCE INFORMATICS
   (Human-Readable Insights, Career Path Prediction)
```

## ✨ Core Modules

- `src/semantic_engine.py`: SBERT embedding generation and semantic similarity.
- `src/knowledge_graph.py`: Graph-based reasoning for prerequisites and related skills.
- `src/ml_model.py`: Random Forest / XGBoost model predicting gap severity.
- `src/hybrid_scorer.py`: Unifies SBERT, Graph, and ML into a single hybrid score.
- `src/explainability.py`: Rule-based NLP engine translating graph and gap data into human-readable insights ("Skill X is missing and is a prerequisite for Skill Y").
- `src/workforce_module.py`: Predicts career trajectories and progression modeling (Beginner → Intermediate → Advanced).

## 🚀 Installation & Usage

### 1. Backend Setup (FastAPI)
The backend requires Python 3.10+ and houses the core AI Engine.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Launch the REST API on port 8000
uvicorn api.main:app --reload --port 8000
```

### 2. Frontend Setup (React/Vite)
The UI is a sleek, modern Single Page Application built on React.
```bash
cd frontend
npm install

# Launch the Application on port 5173
npm run dev
```

### Running the System Evaluation
To compare the baseline SBERT-only approach against the new Hybrid model (metrics output to console):
```bash
python evaluate_hybrid.py
```

## 📊 Evaluation Results

Experiments running simulated candidate data demonstrate that the Hybrid metric provides a more balanced assessment compared to baseline semantic similarity by effectively punishing candidates missing core prerequisites while rewarding candidates with strong adjacent skills.

## 📄 License
This original research code is provided under the MIT License.
