# HistoDyn::BioGater Project 

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run everything

From the repository root:

```bash
python src/run_all.py
```

This generates all CSV files in `results/` and all plots in `figures/`.

These files contain:

- model name
- accuracy
- precision
- recall
- F1-score
- ECE
- energy/image
- confusion matrix

The accuracies are:

| Dataset | RandomOp Mixer | ConvMixer | Pruned ResNet-18 | BioGater |
|---|---:|---:|---:|---:|
| Colorectal | 92.8 | 94.3 | 92.5 | 96.2 |
| PolypGen | 91.4 | 93.2 | 92.5 | 95.8 |
| Cervical | 89.7 | 91.5 | 90.8 | 94.3 |
| Colonoscopy | 98.5 | 99.4 | 95.4 | 99.5 |


Figure 7 uses final normalized gate weights:

| Operation | Gate Weight (%) |
|---|---:|
| mean_subtract | 34.38 |
| abs | 27.35 |
| reverse | 23.56 |
| sign | 7.62 |
| scale_large | 4.49 |
| others | 2.60 |

Table 4 reports:

| Model | GADS |
|---|---:|
| RandomOp Mixer | 2.91 |
| ConvMixer (static) | 1.87 |
| HistoDyn::BioGater (Ours) | 3.22 |


Figure 9 values:

| Model | Communication Cost (%) |
|---|---:|
| RandomOp Mixer | 96 |
| ConvMixer | 78 |
| BioGater | 52 |

Figure 10 values:

| Configuration | Accuracy (%) |
|---|---:|
| Full BioGater | 96.7 |
| w/o Top-k | 94.8 |
| w/o L1 | 95.1 |
| w/o Stain Norm. | 93.9 |
| w/o Contrast | 94.5 |
| w/o Edge Ops | 94.1 |
| w/o EMA | 95.4 |


The grid encodes:

- `scale_large`: red activation regions
- `sign`: blue activation regions

## Notes about ECE and Energy/Image

Accuracy, precision, recall, and F1 are computed from the confusion matrices.

ECE and Energy/Image are included as reported scalar values, because:


- ECE requires predicted probabilities/confidences, not only class labels.
- Energy/Image requires power and timing measurements.
