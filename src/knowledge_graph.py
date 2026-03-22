import networkx as nx
import logging

logger = logging.getLogger(__name__)

# Core hardcoded skill taxonomy to build the skill graph
# Defines prerequisites, related skills, and hierarchical relationships
SKILL_TAXONOMY = {
    "python": {"related": ["pandas", "numpy", "django", "flask", "fastapi"], "type": "language"},
    "machine learning": {"prerequisites": ["python", "statistics", "scikit-learn"], "related": ["deep learning", "data science"], "type": "concept"},
    "deep learning": {"prerequisites": ["machine learning", "tensorflow", "pytorch"], "related": ["computer vision", "nlp"], "type": "concept"},
    "react": {"prerequisites": ["javascript", "html", "css"], "related": ["vue", "angular", "next.js"], "type": "framework"},
    "node.js": {"prerequisites": ["javascript"], "related": ["express", "react"], "type": "framework"},
    "sql": {"related": ["mysql", "postgresql", "nosql"], "type": "language"},
    "aws": {"related": ["azure", "gcp", "cloud", "docker"], "type": "platform"},
    "docker": {"prerequisites": ["linux"], "related": ["kubernetes", "ci/cd"], "type": "tool"},
    "kubernetes": {"prerequisites": ["docker"], "related": ["aws", "gcp"], "type": "tool"},
    "data science": {"prerequisites": ["python", "sql", "statistics"], "related": ["machine learning", "data visualization"], "type": "domain"},
    "nlp": {"prerequisites": ["machine learning", "deep learning", "python"], "related": ["transformers", "spacy"], "type": "domain"},
}

class SkillKnowledgeGraph:
    """NetworkX-based Knowledge Graph for Skill Ontology and Reasoning."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Constructs the knowledge graph from the hardcoded taxonomy."""
        for skill, relations in SKILL_TAXONOMY.items():
            self.graph.add_node(skill, type=relations.get("type", "skill"))
            
            # Directed edges for prerequisites (A -> B means A is required for B)
            for prereq in relations.get("prerequisites", []):
                self.graph.add_node(prereq)  # Add if not exists
                self.graph.add_edge(prereq, skill, relation="prerequisite", weight=1.5)
                
            # Undirected (bidirectional) edges for related skills
            for related in relations.get("related", []):
                self.graph.add_node(related)
                self.graph.add_edge(skill, related, relation="related", weight=0.5)
                self.graph.add_edge(related, skill, relation="related", weight=0.5)
        
        logger.info(f"Skill Knowledge Graph initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def calculate_graph_score(self, candidate_skills: set, required_skills: set) -> float:
        """
        Calculates a graph-based similarity score.
        Reward matching exact skills.
        Reward candidate possessing related skills or prerequisites that form a strong foundation.
        """
        if not required_skills:
            return 1.0

        score = 0.0
        max_possible_score = len(required_skills) * 1.0
        
        for req_skill in required_skills:
            req_lower = req_skill.lower()
            
            if req_lower in {s.lower() for s in candidate_skills}:
                score += 1.0
                continue
            
            # Check graph for partial matches
            if req_lower in self.graph:
                # Get related and prereq skills
                neighbors = list(self.graph.successors(req_lower)) + list(self.graph.predecessors(req_lower))
                
                partial_match_found = False
                for n in neighbors:
                    if n in {s.lower() for s in candidate_skills}:
                        partial_match_found = True
                        break
                
                if partial_match_found:
                    score += 0.5 # Partial credit for having adjacent graph skills
        
        return min(1.0, score / max_possible_score) if max_possible_score > 0 else 0.0

    def get_missing_prerequisites(self, target_skill: str, candidate_skills: set) -> list:
        """Identifies which prerequisites a candidate is missing for a target skill."""
        target_lower = target_skill.lower()
        if target_lower not in self.graph:
            return []
            
        prereqs = [n for n, _, d in self.graph.in_edges(target_lower, data=True) if d.get("relation") == "prerequisite"]
        candidate_lower = {s.lower() for s in candidate_skills}
        
        return [p for p in prereqs if p not in candidate_lower]

# Global instance
skill_kg = SkillKnowledgeGraph()
