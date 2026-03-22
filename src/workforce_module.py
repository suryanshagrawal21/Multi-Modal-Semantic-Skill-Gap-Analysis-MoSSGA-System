from src.knowledge_graph import skill_kg

class WorkforceInformatics:
    """Workforce informatics layer for career path prediction and progression."""

    def __init__(self):
        pass

    def predict_career_path(self, current_skills: set, target_role: str):
        """
        Determines the next best skills to learn based on graph proximity.
        """
        next_skills = set()
        for skill in current_skills:
            skill_lower = skill.lower()
            if skill_lower in skill_kg.graph:
                # Add skills that have the current skill as a prerequisite
                for successor in skill_kg.graph.successors(skill_lower):
                    if skill_kg.graph.edges[skill_lower, successor].get("relation") == "prerequisite":
                        if successor not in current_skills:
                            next_skills.add(successor)
                
                # Add related skills
                for neighbor in skill_kg.graph.neighbors(skill_lower):
                    if neighbor not in current_skills:
                        next_skills.add(neighbor)

        # Filter taking target_role into consideration (mock filter for now)
        return list(next_skills)[:5]  # Limit to top 5 recommendations
        
    def model_skill_progression(self, target_skills: list):
        """
        Creates a beginner -> intermediate -> advanced roadmap for a set of target skills.
        """
        progression = {
            "beginner": [],
            "intermediate": [],
            "advanced": []
        }
        
        for skill in target_skills:
            # Simple heuristic based on degree in graph
            if skill.lower() in skill_kg.graph:
                in_degree = skill_kg.graph.in_degree(skill.lower())
                if in_degree == 0:
                    progression["beginner"].append(skill)
                elif in_degree == 1:
                    progression["intermediate"].append(skill)
                else:
                    progression["advanced"].append(skill)
            else:
                progression["beginner"].append(skill) # Default
                
        return progression

workforce_engine = WorkforceInformatics()
