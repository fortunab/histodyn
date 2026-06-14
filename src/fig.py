import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from metrics_utils import metrics_from_cm

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

with open("configs/figure6_confusion_matrices.json", "r") as f:
    DATASETS = json.load(f)

rows = []

for dataset, models in DATASETS.items():
    for model, cm in models.items():
        row = {
            "Dataset": dataset,
            "Model": model,
            **metrics_from_cm(cm),
            "Confusion Matrix": cm
        }
        rows.append(row)

df = pd.DataFrame(rows)

df.to_csv("results/figure6_all_datasets.csv", index=False)

print("\nFigure 6 all datasets")
print(df.to_string(index=False))

pivot = df.pivot(
    index="Dataset",
    columns="Model",
    values="Accuracy (%)"
).loc[
    ["Colorectal", "PolypGen", "Cervical", "Colonoscopy"]
]

order = [
    "RandomOp Mixer",
    "ConvMixer",
    "Pruned ResNet-18",
    "BioGater"
]

pivot = pivot[order]

ax = pivot.plot(
    kind="bar",
    figsize=(9, 5),
    ylim=(88, 100),
    rot=0
)

ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Datasets")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
        fontsize=8,
        rotation=90,
        padding=3
    )

plt.tight_layout()
plt.savefig("figures/figure6_accuracy_comparison.png", dpi=300)
plt.show()