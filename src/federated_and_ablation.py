import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

with open("configs/communication_cost.json", "r") as f:
    communication_cost = json.load(f)

with open("configs/ablation_results.json", "r") as f:
    ablation_results = json.load(f)

df_comm = pd.DataFrame({
    "Model": communication_cost.keys(),
    "Communication Cost (%)": communication_cost.values()
})

df_comm.to_csv("results/figure9_communication_cost.csv", index=False)

print("\nFigure 9 communication cost")
print(df_comm.to_string(index=False))

plt.figure(figsize=(7, 3))
plt.hlines(
    y=df_comm["Model"],
    xmin=0,
    xmax=df_comm["Communication Cost (%)"]
)
plt.plot(df_comm["Communication Cost (%)"], df_comm["Model"], "o")
plt.xlabel("Communication Cost (%)")
plt.title("Communication Cost per Step in Federated Settings")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/figure9_communication_cost.png", dpi=300)
plt.show()


df_ab = pd.DataFrame({
    "Configuration": ablation_results.keys(),
    "Accuracy (%)": ablation_results.values()
})

full_accuracy = ablation_results["Full BioGater"]
df_ab["Drop from Full (%)"] = round(full_accuracy - df_ab["Accuracy (%)"], 1)

df_ab.to_csv("results/figure10_ablation.csv", index=False)

print("\nFigure 10 ablation")
print(df_ab.to_string(index=False))

plt.figure(figsize=(10, 5))
bars = plt.bar(df_ab["Configuration"], df_ab["Accuracy (%)"])

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.05,
        f"{h:.1f}",
        ha="center",
        fontsize=9
    )

plt.ylabel("Accuracy (%)")
plt.xlabel("Ablation configuration")
plt.ylim(88, 98)
plt.xticks(rotation=30, ha="right")
plt.title("Ablation Study of HistoDyn::BioGater")
plt.tight_layout()
plt.savefig("figures/figure10_ablation.png", dpi=300)
plt.show()
