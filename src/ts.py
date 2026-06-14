import os
import json
import numpy as np
import pandas as pd
from metrics_utils import metrics_from_cm

os.makedirs("results", exist_ok=True)

with open("configs/tables_config.json", "r") as f:
    TABLES = json.load(f)

def build_table(name, spec):
    rows = []

    for model, data in spec["rows"].items():
        cm = np.array(data["cm"])

        row = {
            "Model": model,
            **metrics_from_cm(cm)
        }

        for k, v in data.items():
            if k != "cm":
                row[k] = v

        row["Confusion Matrix"] = cm.tolist()
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"results/{name}.csv", index=False)

    print(f"\n{name}: {spec['caption']}")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    for name, spec in TABLES.items():
        build_table(name, spec)