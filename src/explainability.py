from src.knowledge_graph import skill_kg

class ExplainableAI:
    """Generates human-readable explanations for AI-driven skill gaps and recommendations."""
    
    def __init__(self):
        pass

    def generate_explanations(self, missing_skills, candidate_skills, job_role):
        """
        Creates detailed rule-based reasons and insights for missing skills.
        """
        explanations = []
        for missing in missing_skills:
            skill_name = missing.get('skill', '')
            severity = missing.get('severity', 'Medium')
            
            # Use Knowledge Graph to find prerequisites the candidate doesn't have
            missing_prereqs = skill_kg.get_missing_prerequisites(skill_name, candidate_skills)
            
            insight = f"'{skill_name.title()}' is listed as a {severity.lower()} requirement for {job_role}. "
            
            if missing_prereqs:
                insight += f"To master this, you first need to learn its prerequisites: {', '.join([p.title() for p in missing_prereqs])}. "

            if severity == "Critical":
                insight += "This is a core technology for this role; you should prioritize acquiring this skill."
            elif severity == "High":
                insight += "Adding this to your portfolio will significantly increase your match rate."
            else:
                insight += "Consider exploring this as an optional enhancement to your profile."

            explanations.append({
                "skill": skill_name,
                "insight": insight,
                "missing_prerequisites": missing_prereqs
            })
            
        return explanations

explainability_engine = ExplainableAI()
