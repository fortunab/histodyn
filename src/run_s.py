# from train_sims import s
from results import train_sims
print("\n[1/5] Loading proper datasets...")

print("="*50)
print("HistoDyn::BioGater")
print("="*50)

print("\n[2/5] Training models...")
train_sims.s()
print("\n[3/5] Evaluating...")

print("\n[4/5] Generating figures...")

print("\n[5/5] Exporting results...")

print("\nPipeline completed successfully.")

files = [
    "src/ts.py",
    "src/fig.py",
    "src/gate_figures.py",
    "src/federated_and_ablation.py",
    "src/activation_grid.py",
]