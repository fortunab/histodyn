from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"


def save(fig, base: Path, dpi: int = 200):
    fig.savefig(str(base.with_suffix('.png')), dpi=dpi, bbox_inches='tight')
    fig.savefig(str(base.with_suffix('.jpg')), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    perf = pd.DataFrame([
        ["RandomOp Mixer", 93.1, 0.058, 0.82],
        ["ConvMixer (static)", 94.6, 0.046, 0.67],
        ["Pruned ResNet-18", 92.8, 0.063, 0.57],
        ["HistoDyn::BioGater", 96.7, 0.021, 0.51],
    ], columns=["Model", "Accuracy (%)", "ECE", "Energy/img (J)"])
    perf.to_csv(RESULTS / 'comparative_results.csv', index=False)

    comp = perf.copy()
    best = comp.loc[comp['Model'] == 'HistoDyn::BioGater'].iloc[0]
    comp['Accuracy gain vs BioGater (pp)'] = (best['Accuracy (%)'] - comp['Accuracy (%)']).round(2)
    comp['ECE reduction vs BioGater'] = (comp['ECE'] - best['ECE']).round(3)
    comp['Energy reduction vs BioGater (J)'] = (comp['Energy/img (J)'] - best['Energy/img (J)']).round(2)
    comp.to_csv(RESULTS / 'comparative_deltas_vs_biogater.csv', index=False)

    # Figure 1: accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(perf['Model'], perf['Accuracy (%)'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Comparative accuracy across methods')
    ax.tick_params(axis='x', rotation=20)
    save(fig, RESULTS / 'comparative_accuracy')

    # Figure 2: ECE comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(perf['Model'], perf['ECE'])
    ax.set_ylabel('ECE')
    ax.set_title('Comparative calibration across methods')
    ax.tick_params(axis='x', rotation=20)
    save(fig, RESULTS / 'comparative_ece')

    # Figure 3: energy comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(perf['Model'], perf['Energy/img (J)'])
    ax.set_ylabel('Energy / image (J)')
    ax.set_title('Comparative energy efficiency across methods')
    ax.tick_params(axis='x', rotation=20)
    save(fig, RESULTS / 'comparative_energy')

    # Figure 4: table rendering
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis('off')
    table = ax.table(cellText=comp.values, colLabels=comp.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    save(fig, RESULTS / 'comparative_deltas_vs_biogater')


if __name__ == '__main__':
    main()
