import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# KIVE Signal Analysis -- 9-Dimensional Fraud Detection
This notebook evaluates the separability of the 9 microservice signal dimensions that form the
KIVE expert fraud detection system.

**Signal Taxonomy:**

| Signal | Full Name | Type | What It Detects |
|--------|-----------|------|-----------------|
| TAV | Temporal Anchoring Violations | Passive | Impossible timeline overlaps in employment history |
| SVP | Specificity Variance Profile | Passive | Vague vs. concrete skill descriptions |
| FMD | Failure Memory Deficiency | Passive | Inability to recall project failures authentically |
| MDC | Market Demand Correlation | Passive | Suspiciously trending skill claims |
| TSI | Trajectory Smoothness Index | Passive | Unnaturally linear career progression |
| BES | Behavioral Entropy Service | Active | Keystroke dynamics, paste events, typing entropy |
| LQA | Linguistic Quality Assurance | Active | AI-generated phrasing artifacts |
| CCS | Cross-Candidate Similarity | Active | Response overlap across candidates |
| RSL | Response Latency Slope | Active | Suspiciously fast or flat response times |

**Passive** signals are computed from the static profile (resume analysis).
**Active** signals require live behavioral interaction (probing).
"""

code_imports = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#111111",
    "figure.facecolor": "#111111",
    "axes.edgecolor": "#333333",
    "text.color": "white",
})
"""

code_load = """# Resolve project root
cwd = Path.cwd()
project_root = cwd
while not (project_root / "data" / "signal_distributions.csv").exists():
    if project_root.parent == project_root:
        raise FileNotFoundError(
            f"Could not find 'data/signal_distributions.csv' above {cwd}. "
            "Upload the 'data' folder to your session."
        )
    project_root = project_root.parent

df = pd.read_csv(project_root / "data" / "signal_distributions.csv")
print(f"Loaded {len(df)} profiles: {(df['label']=='REAL').sum()} real, {(df['label']=='FRAUD').sum()} fraud")
df.head()
"""

code_kde = """# --- 3x3 KDE Plot: Signal Separability ---
signals = ['tav_score', 'svp_score', 'fmd_score', 'mdc_score', 'tsi_score',
           'bes_score', 'lqa_score', 'ccs_score', 'rsl_score']
titles = [
    'TAV (Temporal Anchoring)', 'SVP (Specificity Variance)', 'FMD (Failure Memory)',
    'MDC (Market Demand)', 'TSI (Trajectory Smoothness)',
    'BES (Behavioral Entropy)', 'LQA (Linguistic Quality)', 
    'CCS (Cross-Candidate)', 'RSL (Response Latency)'
]
types = ['Passive','Passive','Passive','Passive','Passive','Active','Active','Active','Active']

fig, axes = plt.subplots(3, 3, figsize=(24, 18))
axes = axes.flatten()

for i, (sig, title, stype) in enumerate(zip(signals, titles, types)):
    if sig not in df.columns:
        axes[i].set_title(f"{title} [MISSING]", color='yellow')
        continue
    sns.kdeplot(data=df, x=sig, hue='label', fill=True, ax=axes[i],
                palette={'REAL': '#00ffcc', 'FRAUD': '#ff3366'}, alpha=0.6, linewidth=2)
    border_color = '#ffaa00' if stype == 'Active' else '#4488ff'
    for spine in axes[i].spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)
    axes[i].set_title(f"{title} [{stype}]", color=border_color, pad=12, fontsize=13, fontweight='bold')
    axes[i].set_xlim(-0.1, 1.1)
    axes[i].set_xlabel("Score (0=Real, 1=Fraud)", color='gray')
    axes[i].set_ylabel("Density", color='gray')

plt.tight_layout(pad=3.0)
out_path = project_root / "docs" / "signal_distributions.png"
if out_path.parent.exists():
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
"""

code_boxplot = """# --- Box Plot: Signal Score Distribution by Label ---
signals = ['tav_score','svp_score','fmd_score','mdc_score','tsi_score',
           'bes_score','lqa_score','ccs_score','rsl_score']
existing = [s for s in signals if s in df.columns]

melted = df.melt(id_vars=['label'], value_vars=existing, var_name='signal', value_name='score')
melted['signal'] = melted['signal'].str.replace('_score','').str.upper()

fig, ax = plt.subplots(figsize=(16, 6))
sns.boxplot(data=melted, x='signal', y='score', hue='label',
            palette={'REAL': '#00ffcc', 'FRAUD': '#ff3366'}, ax=ax, width=0.6)
ax.set_title("Signal Score Distribution by Label", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Signal Service", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.legend(title="Label", fontsize=10)
plt.tight_layout()
plt.show()
"""

code_sep = """# --- Separability Statistics with Statistical Significance ---
from scipy import stats

signals = ['tav_score','svp_score','fmd_score','mdc_score','tsi_score',
           'bes_score','lqa_score','ccs_score','rsl_score']
existing = [s for s in signals if s in df.columns]

rows = []
for sig in existing:
    real = df[df['label']=='REAL'][sig]
    fraud = df[df['label']=='FRAUD'][sig]
    
    # Basic statistics
    gap = abs(fraud.mean() - real.mean())
    overlap = max(0, min(fraud.quantile(0.75), real.quantile(0.75)) - max(fraud.quantile(0.25), real.quantile(0.25)))
    
    # Statistical tests
    t_stat, p_value = stats.ttest_ind(fraud, real)
    ks_stat, ks_p = stats.ks_2samp(fraud, real)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(real)-1)*real.std()**2 + (len(fraud)-1)*fraud.std()**2) / (len(real)+len(fraud)-2))
    cohens_d = gap / pooled_std if pooled_std > 0 else 0
    
    rows.append({
        'Signal': sig.replace('_score','').upper(),
        'Real μ': f"{real.mean():.3f}",
        'Fraud μ': f"{fraud.mean():.3f}",
        'Gap': f"{gap:.3f}",
        "Cohen's d": f"{cohens_d:.2f}",
        'KS stat': f"{ks_stat:.3f}",
        'p-value': f"{p_value:.2e}",
        'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'
    })

sep_df = pd.DataFrame(rows)
print("\\n=== Signal Separability Report (Statistical Analysis) ===")
print(sep_df.to_string(index=False))
print("\\nEffect Size: |d| < 0.2 (small), 0.2-0.8 (medium), > 0.8 (large)")
print("Significance: *** p<0.001, ** p<0.01, * p<0.05, NS not significant")
"""

code_corr = """# --- Cross-Signal Correlation Heatmaps ---
signals = ['tav_score','svp_score','fmd_score','mdc_score','tsi_score',
           'bes_score','lqa_score','ccs_score','rsl_score']
existing = [s for s in signals if s in df.columns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

real_corr = df[df['label']=='REAL'][existing].corr()
fraud_corr = df[df['label']=='FRAUD'][existing].corr()

labels = [s.replace('_score','').upper() for s in existing]

sns.heatmap(real_corr, annot=True, cmap='viridis', vmin=-1, vmax=1, ax=ax1,
           xticklabels=labels, yticklabels=labels, fmt='.2f')
ax1.set_title("Cross-Signal Correlation (REAL)", color='#00ffcc', fontsize=13, fontweight='bold')

sns.heatmap(fraud_corr, annot=True, cmap='plasma', vmin=-1, vmax=1, ax=ax2,
           xticklabels=labels, yticklabels=labels, fmt='.2f')
ax2.set_title("Cross-Signal Correlation (FRAUD)", color='#ff3366', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# Identify highly correlated signal pairs (potential redundancy)
print("\\n=== Signal Redundancy Analysis ===")
print("Highly correlated pairs (|r| > 0.7) may indicate redundant information:\\n")
for i in range(len(existing)):
    for j in range(i+1, len(existing)):
        r_real = real_corr.iloc[i, j]
        r_fraud = fraud_corr.iloc[i, j]
        if abs(r_real) > 0.7 or abs(r_fraud) > 0.7:
            print(f"{labels[i]} <-> {labels[j]}: REAL r={r_real:.3f}, FRAUD r={r_fraud:.3f}")
"""

code_roc = """# --- ROC Curves and AUC Scores ---
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

signals = ['tav_score','svp_score','fmd_score','mdc_score','tsi_score',
           'bes_score','lqa_score','ccs_score','rsl_score']
existing = [s for s in signals if s in df.columns]
types = ['Passive','Passive','Passive','Passive','Passive','Active','Active','Active','Active']
type_map = {s: t for s, t in zip(signals, types) if s in existing}

# Binary labels (1=FRAUD, 0=REAL)
y_true = (df['label'] == 'FRAUD').astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ROC curves
auc_scores = {}
for sig in existing:
    fpr, tpr, _ = roc_curve(y_true, df[sig])
    roc_auc = auc(fpr, tpr)
    auc_scores[sig] = roc_auc
    color = '#ffaa00' if type_map[sig] == 'Active' else '#4488ff'
    linestyle = '-' if type_map[sig] == 'Active' else '--'
    ax1.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2, 
            label=f"{sig.replace('_score','').upper()} (AUC={roc_auc:.3f})", alpha=0.8)

ax1.plot([0, 1], [0, 1], 'gray', linestyle=':', linewidth=1, label='Random (AUC=0.500)')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves: Signal Discriminative Power', fontsize=14, fontweight='bold', color='white')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(alpha=0.3)

# Precision-Recall curves
for sig in existing:
    precision, recall, _ = precision_recall_curve(y_true, df[sig])
    avg_precision = average_precision_score(y_true, df[sig])
    color = '#ffaa00' if type_map[sig] == 'Active' else '#4488ff'
    linestyle = '-' if type_map[sig] == 'Active' else '--'
    ax2.plot(recall, precision, color=color, linestyle=linestyle, linewidth=2,
            label=f"{sig.replace('_score','').upper()} (AP={avg_precision:.3f})", alpha=0.8)

baseline = y_true.mean()
ax2.axhline(baseline, color='gray', linestyle=':', linewidth=1, label=f'Baseline (AP={baseline:.3f})')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold', color='white')
ax2.legend(loc='lower left', fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# AUC ranking
print("\\n=== Signal Discriminative Power (AUC Ranking) ===")
auc_df = pd.DataFrame([
    {'Signal': sig.replace('_score','').upper(), 'Type': type_map[sig], 'AUC': auc_scores[sig]}
    for sig in existing
]).sort_values('AUC', ascending=False)
print(auc_df.to_string(index=False))
"""

code_pca = """# --- Signal Redundancy via PCA ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

signals = ['tav_score','svp_score','fmd_score','mdc_score','tsi_score',
           'bes_score','lqa_score','ccs_score','rsl_score']
existing = [s for s in signals if s in df.columns]

# Standardize signals
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[existing])

# PCA
pca = PCA()
pca.fit(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Scree plot
ax1.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, 
       color='#00ffcc', alpha=0.7, edgecolor='white')
ax1.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_),
        color='#ff3366', marker='o', linewidth=2, label='Cumulative')
ax1.axhline(0.95, color='gray', linestyle='--', linewidth=1, label='95% threshold')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('PCA Scree Plot: Signal Dimensionality', fontsize=14, fontweight='bold', color='white')
ax1.legend()
ax1.grid(alpha=0.3)

# Component loadings heatmap
loadings = pca.components_[:5, :]  # First 5 PCs
labels = [s.replace('_score','').upper() for s in existing]
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, ax=ax2,
           xticklabels=labels, yticklabels=[f'PC{i+1}' for i in range(5)], fmt='.2f')
ax2.set_title('PCA Loadings: Signal Contributions', fontsize=14, fontweight='bold', color='white')

plt.tight_layout()
plt.show()

# Dimensionality report
n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"\\n=== Dimensionality Analysis ===")
print(f"Original dimensions: {len(existing)}")
print(f"Components for 95% variance: {n_components_95}")
print(f"Dimensionality reduction: {(1 - n_components_95/len(existing))*100:.1f}%")
print(f"\\nInterpretation: {'Signals are highly redundant' if n_components_95 <= 3 else 'Signals carry independent information' if n_components_95 >= 7 else 'Moderate redundancy'}")
"""

code_mutual_info = """# --- Mutual Information Analysis ---
from sklearn.feature_selection import mutual_info_classif

signals = ['tav_score','svp_score','fmd_score','mdc_score','tsi_score',
           'bes_score','lqa_score','ccs_score','rsl_score']
existing = [s for s in signals if s in df.columns]

y = (df['label'] == 'FRAUD').astype(int)
X = df[existing]

# Compute mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)

fig, ax = plt.subplots(figsize=(14, 6))
labels = [s.replace('_score','').upper() for s in existing]
colors = ['#ffaa00' if 'bes' in s or 'lqa' in s or 'ccs' in s or 'rsl' in s else '#4488ff' 
         for s in existing]

bars = ax.bar(labels, mi_scores, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
ax.set_xlabel('Signal', fontsize=12)
ax.set_ylabel('Mutual Information (bits)', fontsize=12)
ax.set_title('Mutual Information: Signal-Label Dependency', fontsize=14, fontweight='bold', color='white')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, mi_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{score:.3f}', ha='center', va='bottom', fontsize=9, color='white')

plt.tight_layout()
plt.show()

# MI ranking
mi_df = pd.DataFrame({
    'Signal': labels,
    'MI Score': mi_scores
}).sort_values('MI Score', ascending=False)
print("\\n=== Mutual Information Ranking ===")
print(mi_df.to_string(index=False))
print("\\nHigher MI = stronger dependency with fraud label")
"""

code_bootstrap = """# --- Bootstrap Confidence Intervals ---
from scipy.stats import bootstrap

signals = ['tav_score','svp_score','fmd_score','mdc_score','tsi_score',
           'bes_score','lqa_score','ccs_score','rsl_score']
existing = [s for s in signals if s in df.columns]

def mean_diff(real, fraud):
    return np.abs(np.mean(fraud) - np.mean(real))

fig, ax = plt.subplots(figsize=(14, 6))

results = []
for sig in existing:
    real = df[df['label']=='REAL'][sig].values
    fraud = df[df['label']=='FRAUD'][sig].values
    
    # Bootstrap CI for mean difference
    rng = np.random.default_rng(42)
    res = bootstrap((real, fraud), lambda r, f: mean_diff(r, f), 
                   n_resamples=1000, confidence_level=0.95, random_state=rng)
    
    point_est = mean_diff(real, fraud)
    ci_low, ci_high = res.confidence_interval
    
    results.append({
        'signal': sig,
        'point': point_est,
        'ci_low': ci_low,
        'ci_high': ci_high
    })

# Sort by point estimate
results = sorted(results, key=lambda x: x['point'], reverse=True)
labels = [r['signal'].replace('_score','').upper() for r in results]
points = [r['point'] for r in results]
ci_lows = [r['ci_low'] for r in results]
ci_highs = [r['ci_high'] for r in results]

y_pos = np.arange(len(labels))
ax.barh(y_pos, points, color='#00ffcc', alpha=0.7, edgecolor='white')

# Compute error bar magnitudes (must be positive)
lower_errors = [abs(p - l) for p, l in zip(points, ci_lows)]
upper_errors = [abs(h - p) for h, p in zip(ci_highs, points)]

ax.errorbar(points, y_pos, xerr=[lower_errors, upper_errors],
           fmt='none', ecolor='#ff3366', linewidth=2, capsize=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('Mean Score Difference (FRAUD - REAL)', fontsize=12)
ax.set_title('Bootstrap 95% Confidence Intervals for Signal Separability', fontsize=14, fontweight='bold', color='white')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n=== Bootstrap CI Analysis ===")
for r in results:
    sig = r['signal'].replace('_score','').upper()
    print(f"{sig:5s}: {r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell("## 1. Signal Distribution Analysis (KDE)\nEach subplot shows the probability density for REAL vs FRAUD profiles. **Blue borders** = passive signals (free at reset). **Orange borders** = active signals (require probing)."),
    nbf.v4.new_code_cell(code_kde),
    nbf.v4.new_markdown_cell("## 2. Box Plot Comparison\nCompact view of all 9 signal distributions side-by-side."),
    nbf.v4.new_code_cell(code_boxplot),
    nbf.v4.new_markdown_cell("## 3. Statistical Separability Analysis\nQuantitative assessment with t-tests, KS tests, and effect sizes (Cohen's d). Production ML requires statistical rigor beyond visual inspection."),
    nbf.v4.new_code_cell(code_sep),
    nbf.v4.new_markdown_cell("## 4. ROC & Precision-Recall Curves\nROC AUC measures discriminative power (0.5=random, 1.0=perfect). PR curves are critical for imbalanced datasets. These metrics guide feature selection in production."),
    nbf.v4.new_code_cell(code_roc),
    nbf.v4.new_markdown_cell("## 5. Signal Redundancy Analysis (Correlation)\nCorrelation heatmaps reveal whether signals carry independent information or are redundant. High correlation (|r|>0.7) suggests one signal may be dropped."),
    nbf.v4.new_code_cell(code_corr),
    nbf.v4.new_markdown_cell("## 6. Dimensionality Reduction (PCA)\nPCA reveals intrinsic dimensionality. If 95% variance is captured in <5 components, signals are redundant. Loadings show which signals contribute to each PC."),
    nbf.v4.new_code_cell(code_pca),
    nbf.v4.new_markdown_cell("## 7. Mutual Information Analysis\nMI measures non-linear dependency between signals and labels. Unlike correlation, MI captures complex relationships. Higher MI = more informative signal."),
    nbf.v4.new_code_cell(code_mutual_info),
    nbf.v4.new_markdown_cell("## 8. Bootstrap Confidence Intervals\nBootstrap resampling provides uncertainty estimates for signal separability. Narrow CIs = reliable estimates. Wide CIs = need more data or signal is noisy."),
    nbf.v4.new_code_cell(code_bootstrap),
]

with open('notebooks/01_signal_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Signal analysis notebook generated.")
