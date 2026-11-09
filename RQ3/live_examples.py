"""Replay fixed live examples and render English labels without changing inputs."""
import json
from pathlib import Path
from hashlib import sha256

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from research.feature_review import CONNECTIONS, hand_crop
from research.studies import gh_live_diagnostics


def main():
    # The fixed numerical replay still checks original model/input/capture hashes.
    gh_live_diagnostics.main()
    out = gh_live_diagnostics.OUT
    rows = pd.read_csv(out / 'predictions.csv', keep_default_na=False)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), layout='constrained')
    for row, ax in zip(rows.to_dict('records'), axes.flat):
        folder = Path('captures/confusions') / row['id']
        record = json.loads((folder / 'capture.json').read_text())
        points = np.asarray(record['detection']['image_landmarks'])
        with Image.open(folder / 'model_input.png') as source:
            crop, box = hand_crop(source, points, overlay=False)
            xy = points[:, :2] * source.size - np.array(box[:2])
        ax.imshow(crop)
        for a, b in CONNECTIONS:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color='#00ba89', lw=1.5)
        ax.scatter(xy[:, 0], xy[:, 1], s=13, c='#ffcc33', edgecolors='none', zorder=3)
        view = 'palm visible' if row['panel'] == 'D' else 'back visible'
        ax.set_title(f"{row['panel']}. Intended {row['expected']} | {view}\n"
                     f"Reference: {row['reference']} | Rotation: {row['rotation']}", fontsize=11)
        ax.axis('off')
    figure = Path('paper/figures/gh_live_diagnostics.pdf')
    fig.savefig(figure)
    plt.close(fig)
    verification = json.loads((out / 'verification.json').read_text())
    verification['figure_sha256'] = sha256(figure.read_bytes()).hexdigest()
    verification['figure_language'] = 'English'
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    lines = []
    for row in rows.to_dict('records'):
        view = 'Palm visible' if row['panel'] == 'D' else 'Back visible'
        lines.append(f"{row['panel']} & {row['expected']} & {view} & "
                     f"{row['reference']} & {row['rotation']} \\\\")
    Path('paper/gh_live_diagnostics_rows_nl.tex').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
