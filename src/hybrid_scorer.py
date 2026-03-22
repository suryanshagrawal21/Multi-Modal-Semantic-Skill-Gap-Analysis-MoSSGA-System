from src.semantic_skill_matcher import calculate_semantic_gap_score
from src.knowledge_graph import skill_kg
from src.ml_model import ml_model
import logging

logger = logging.getLogger(__name__)

class HybridScorer:
    """Combines Semantic, Knowledge Graph, and ML models into a unified scoring system."""

    def __init__(self, semantic_weight=0.4, graph_weight=0.3, ml_weight=0.3):
        self.w_semantic = semantic_weight
        self.w_graph = graph_weight
        self.w_ml = ml_weight
        
    def compute_hybrid_score(self, candidate_skills: set, required_skills: set, semantic_match_result: dict) -> dict:
        """
        Calculates the unified hybrid score.
        Semantic Score: Base NLP match (0-100)
        Graph Score: Skill ontology and exact/partial match (0-100)
        ML Score: RF/XGBoost gap prediction mapped inversely (0-100)
        """
        if not required_skills:
            return {"hybrid_score": 100.0, "semantic_score": 100.0, "graph_score": 100.0, "ml_score": 100.0}

        # 1. Semantic Score (from existing matching output)
        semantic_score = semantic_match_result.get("match_percentage", 0.0)

        # 2. Knowledge Graph Score
        graph_score = skill_kg.calculate_graph_score(candidate_skills, required_skills) * 100.0

        # 3. ML Model Score
        # ML predicts severity (0 to 1). We want a "score" where 100 means no gap (severity 0)
        missing_count = len([m for m in semantic_match_result.get("matched_skills", []) if m["similarity"] < 0.6])
        if not semantic_match_result.get("matched_skills"):
             # approximation if matched_skills isn't formatted as expected
             missing_skills = required_skills - candidate_skills
             missing_count = len(missing_skills)

        severity_prediction = ml_model.predict_severity(
            semantic_score=semantic_score/100.0, 
            graph_score=graph_score/100.0, 
            missing_count=missing_count, 
            total_required=len(required_skills)
        )
        ml_score_component = (1.0 - severity_prediction) * 100.0

        # Final Weighted Score
        final_score = (self.w_semantic * semantic_score) + \
                      (self.w_graph * graph_score) + \
                      (self.w_ml * ml_score_component)

        logger.info(f"Hybrid Score Computed: {final_score:.2f} (Sem:{semantic_score:.1f}, Graph:{graph_score:.1f}, ML:{ml_score_component:.1f})")

        return {
            "hybrid_score": round(final_score, 2),
            "semantic_score": round(semantic_score, 2),
            "graph_score": round(graph_score, 2),
            "ml_score": round(ml_score_component, 2),
            "severity_prediction": severity_prediction
        }

# Global instance
hybrid_scorer = HybridScorer()
