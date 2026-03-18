"""Quick verification that the MoSSGA pipeline modules work correctly."""

from src.semantic_skill_matcher import semantic_skill_match, calculate_semantic_gap_score
from src.multimodal_fusion import fuse_skill_profiles, generate_fusion_summary

print("=" * 60)
print("MoSSGA Pipeline Verification")
print("=" * 60)

# Test 1: Semantic Skill Matching
print("\n[1] Semantic Skill Match Test")
result = semantic_skill_match(
    candidate_skills=["Python", "ML", "React", "SQL", "Docker"],
    jd_skills=["python", "machine learning", "javascript", "sql", "kubernetes"],
)
print(f"    Match %: {result['match_percentage']}")
for m in result["matched_skills"]:
    print(f"    {m['jd_skill']:20s} -> {m['candidate_skill']:20s} ({m['similarity']:.2f}, {m['match_type']})")
print(f"    Missing: {result['missing_skills']}")
assert result["match_percentage"] > 0, "Match should be > 0"
print("    => PASSED")

# Test 2: Gap Analysis
print("\n[2] Skill Gap Analysis Test")
gap = calculate_semantic_gap_score(result)
print(f"    Gap Score: {gap['gap_score']}%, Severity: {gap['gap_severity']}")
assert gap["gap_severity"] in ("Low", "Medium", "High", "Critical"), "Invalid severity"
print("    => PASSED")

# Test 3: Multi-Modal Fusion
print("\n[3] Multi-Modal Fusion Test")
fusion = fuse_skill_profiles(
    ["python", "react", "sql", "docker"],
    ["python", "javascript", "docker", "linux", "aws"],
)
print(f"    Total unique: {fusion['fusion_stats']['total_unique']}")
print(f"    Validated by both: {fusion['validation_summary']['both_sources']}")
print(f"    Resume only: {fusion['validation_summary']['resume_only']}")
print(f"    GitHub only: {fusion['validation_summary']['github_only']}")
assert fusion["fusion_stats"]["total_unique"] == 7, "Should be 7 unique skills"
assert set(fusion["validation_summary"]["both_sources"]) == {"python", "docker"}, "Both should have python & docker"
print("    => PASSED")

# Test 4: Fusion Summary
print("\n[4] Fusion Summary Test")
summary = generate_fusion_summary(fusion)
assert "Multi-Modal Fusion Summary" in summary, "Summary should contain header"
print("    => PASSED")

print("\n" + "=" * 60)
print("ALL MOSSGA TESTS PASSED!")
print("=" * 60)
