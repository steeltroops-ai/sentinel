import nbformat as nbf

# 1. Update Signal Analysis Notebook
nb = nbf.read('notebooks/01_signal_analysis.ipynb', as_version=4)

with open('data/signal_distributions.csv', 'r') as f:
    csv_data = f.read()

nb.cells[2].source = '''import io
csv_string = """''' + csv_data + '''"""
df = pd.read_csv(io.StringIO(csv_string))
print(f"Loaded {len(df)} profiles.")
df.head()
'''
nbf.write(nb, 'notebooks/01_signal_analysis.ipynb')

# 2. Update RL Training Notebook
nb_rl = nbf.read('notebooks/02_rl_training.ipynb', as_version=4)

with open('artifacts/training/learning_curve.csv', 'r') as f:
    lc_csv = f.read()
    
with open('artifacts/training/sample_traces.csv', 'r') as f:
    st_csv = f.read()

nb_rl.cells[2].source = '''import io
csv_string = """''' + lc_csv + '''"""
df = pd.read_csv(io.StringIO(csv_string))
print(f"Loaded {len(df)} evaluation windows.")
df.tail()
'''

nb_rl.cells[4].source = '''import io
csv_string2 = """''' + st_csv + '''"""
traces = pd.read_csv(io.StringIO(csv_string2))
traces.head(6)
'''

# ALSO strip the plt.savefig out of both notebooks because it'll fail on remote colab if docs/ doesn't exist
# We'll just let them display inline.
nb.cells[3].source = nb.cells[3].source.replace("out_path = '../docs/signal_distributions.png' if os.path.exists('../docs') else 'docs/signal_distributions.png'", "")
nb.cells[3].source = nb.cells[3].source.replace("plt.savefig(out_path, dpi=300, bbox_inches='tight')", "")

nb_rl.cells[3].source = nb_rl.cells[3].source.replace("out_path = '../docs/learning_curve.png' if os.path.exists('../docs') else 'docs/learning_curve.png'", "")
nb_rl.cells[3].source = nb_rl.cells[3].source.replace("plt.savefig(out_path, dpi=300, bbox_inches='tight')", "")

nbf.write(nb, 'notebooks/01_signal_analysis.ipynb')
nbf.write(nb_rl, 'notebooks/02_rl_training.ipynb')

print('Notebooks successfully rewritten with embedded CSVs!')
