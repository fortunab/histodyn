import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

with open("configs/figure11_activation_maps.json", "r") as f:
    activation_maps = json.load(f)

scale_large = np.array(activation_maps["scale_large"])
sign = np.array(activation_maps["sign"])

np.savetxt(
    "results/figure11_scale_large_grid.csv",
    scale_large,
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    "results/figure11_sign_grid.csv",
    sign,
    delimiter=",",
    fmt="%d"
)

print("scale_large")
print(scale_large)

print("\nsign")
print(sign)

fig, ax = plt.subplots(figsize=(5, 5))

for i in range(6):
    ax.axhline(i, color="gray", alpha=0.4)
    ax.axvline(i, color="gray", alpha=0.4)

for r in range(5):
    for c in range(5):

        if scale_large[r, c]:
            ax.add_patch(
                plt.Rectangle(
                    (c, 4-r),
                    1,
                    1,
                    color="red",
                    alpha=0.35,
                    label="scale_large"
                    if (r, c) == (1, 2)
                    else None
                )
            )

        if sign[r, c]:
            ax.add_patch(
                plt.Rectangle(
                    (c, 4-r),
                    1,
                    1,
                    color="blue",
                    alpha=0.35,
                    label="sign"
                    if (r, c) == (2, 3)
                    else None
                )
            )

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)

ax.set_xticks([])
ax.set_yticks([])

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=2
)

plt.title("Activation grid (Patch space)")
plt.tight_layout()

plt.savefig(
    "figures/figure11_activation_grid.png",
    dpi=300
)

plt.show()
