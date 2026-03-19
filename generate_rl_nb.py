import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# KIVE RL Agent Training Analysis
This notebook evaluates the performance of the DQN Orchestrator agent across the vetting POMDP.
The agent learns to dynamically PROBE based on signal ambiguity to reduce uncertainty, balancing the cost of probing against the high penalty of admitting a fraud profile.
"""

code_imports = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#111111", "figure.facecolor": "#111111", "axes.edgecolor": "#333333"})
"""

code_load = """df = pd.read_csv('../artifacts/training/learning_curve.csv')
print(f"Loaded {len(df)} evaluation windows.")
df.tail()
"""

code_plot = """fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = '#00ffcc'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward', color=color1)
ax1.plot(df['episode'], df['reward_mean'], color=color1, linewidth=2, label='Mean Reward')
ax1.fill_between(df['episode'], df['reward_mean'] - df['reward_std'], df['reward_mean'] + df['reward_std'], color=color1, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#ff3366'
ax2.set_ylabel('FN Rate', color=color2)
ax2.plot(df['episode'], df['fn_rate'], color=color2, linewidth=2, linestyle='--', label='False Negative Rate')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.axhline(0.10, color='gray', linestyle=':', label='Target FN limit (10%)')

fig.tight_layout()
plt.title("RL Orchestrator Convergence", color='white', pad=15, fontsize=14)
ax1.legend(loc='lower left')
ax2.legend(loc='lower right')
plt.savefig('../docs/learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()
"""

code_traces = """traces = pd.read_csv('../artifacts/training/sample_traces.csv')
traces.head(6)
"""

code_probe_dist = """sns.histplot(data=traces, x='actions_taken', hue='true_label', multiple='stack', 
             palette={'REAL': '#00ffcc', 'FRAUD': '#ff3366'})
plt.title("Probe Frequency Distribution", color='white')
plt.xlabel("Number of Actions Taken (Probes + Terminal)")
plt.ylabel("Count")
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_plot),
    nbf.v4.new_code_cell(code_traces),
    nbf.v4.new_code_cell(code_probe_dist)
]

with open('notebooks/02_rl_training.ipynb', 'w') as f:
    nbf.write(nb, f)
print("RL notebook generated successfully.")
