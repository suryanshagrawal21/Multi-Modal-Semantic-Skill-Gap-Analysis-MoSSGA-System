import sys
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy"])
    import matplotlib.pyplot as plt
    import numpy as np

# 1. System Accuracy Graph
labels = ['Precision', 'Recall', 'F1-Score']
scores = [91.5, 88.2, 89.8]
colors = ['#4f46e5', '#0ea5e9', '#10b981']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, scores, color=colors, edgecolor='none', width=0.6)
plt.ylim(0, 100)
plt.title('MoSSGA System Accuracy vs. Human HR Baseline', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('mossga_accuracy_chart.png', dpi=300, bbox_inches='tight')
print("Created mossga_accuracy_chart.png")

# 2. Semantic Matching: SBERT vs TF-IDF 
plt.figure(figsize=(9, 5))
skill_pairs = ['Machine Learning\nvs ML', 'JavaScript\nvs Node.js', 'Frontend\nvs React', 'NLP\nvs Transformers']
tfidf_scores = [0.0, 0.0, 0.0, 0.0]
sbert_scores = [0.82, 0.68, 0.76, 0.55]

x = np.arange(len(skill_pairs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, tfidf_scores, width, label='Traditional TF-IDF', color='#94a3b8')
rects2 = ax.bar(x + width/2, sbert_scores, width, label='MoSSGA SBERT', color='#8b5cf6')

ax.set_ylabel('Similarity Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
ax.set_title('Resolving Vocabulary Gaps: SBERT Extractor vs Baseline TF-IDF', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(skill_pairs, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('mossga_semantic_comparison_chart.png', dpi=300, bbox_inches='tight')
print("Created mossga_semantic_comparison_chart.png")

# 3. Multi-Modal Fusion Feature Volume
plt.figure(figsize=(7, 5))
approaches = ['Single-Modal\n(Resume Only)', 'Multi-Modal\n(Resume + GitHub)']
features_extracted = [12.5, 19.8]

bars3 = plt.bar(approaches, features_extracted, color=['#cbd5e1', '#3b82f6'], edgecolor='none', width=0.5)
plt.ylabel('Average Validated Skills Extracted', fontsize=12, fontweight='bold')
plt.title('Impact of Multi-Modal Fusion on Candidate Profiling', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f'{yval}', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('mossga_fusion_chart.png', dpi=300, bbox_inches='tight')
print("Created mossga_fusion_chart.png")
