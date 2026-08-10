import json
import numpy as np
import matplotlib.pyplot as plt
import os


JSON_FILE = "/home/ep42de8l/SCION/scion-main/examples/weight_alignment/alignment_metrics_for_W_and_Z_with_phi_h_lr_0.01_momentum_0.9_1000_steps.json"
SAVE_DIR = "/home/ep42de8l/SCION/scion-main/examples/weight_alignment/alignment_plots_W_and_Z"

os.makedirs(SAVE_DIR, exist_ok=True)

with open(JSON_FILE, "r") as f:
    data = json.load(f)


color = "#0d8cb3"

if "phi_h" in JSON_FILE:
    titles = [
        r'Mean alignement of $W^{(\ell)}$ and $\phi(h^{(\ell -1)})$',
        r'Mean alignement of $Z^{(\ell)}$ and $\phi(h^{(\ell -1)})$',
    ]

    ylabels = [
        r'$\frac{ || W^{(\ell)} \phi(h^{(\ell -1)}) ||_2}{|| W^{(\ell)} ||_{\mathcal{S}_\infty} ||\phi(h^{(\ell -1)}) ||_2}$',
        r'$\frac{ || Z^{(\ell)} \phi(h^{(\ell -1)}) ||_2}{|| Z^{(\ell)} ||_{\mathcal{S}_\infty} ||\phi(h^{(\ell -1)}) ||_2}$',
    ]
else:
    titles = [
        r'Mean alignement of $W^{(\ell)}$ and $h^{(\ell -1)}$',
        r'Mean alignement of $Z^{(\ell)}$ and $h^{(\ell -1)}$',
    ]

    ylabels = [
        r'$\frac{ || W^{(\ell)} h^{(\ell -1)} ||_2}{|| W^{(\ell)} ||_{\mathcal{S}_\infty} ||h^{(\ell -1)} ||_2}$',
        r'$\frac{ || Z^{(\ell)} h^{(\ell -1)} ||_2}{|| Z^{(\ell)} ||_{\mathcal{S}_\infty} ||h^{(\ell -1)} ||_2}$',
    ]



for checkpoint in sorted(data.keys(), key=int):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metric_info = [
        ("alignment_W", titles[0], ylabels[0]),
        ("alignment_Z", titles[1], ylabels[1]),
    ]

    for ax, (metric, title, ylabel) in zip(np.ravel(axes), metric_info):

        metric_data = data[checkpoint][metric]

        depths = sorted(
            [int(d) for d in metric_data.keys()]
        )

        means = []
        stds = []

        for depth in depths:
            vals = np.asarray(metric_data[str(depth)])

            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.asarray(means)
        stds = np.asarray(stds)

        ax.plot(
            depths,
            means,
            color=color,
            marker="o",
            markersize=4,
            linewidth=2,
        )

        ax.fill_between(
            depths,
            means - stds,
            means + stds,
            alpha=0.2,
            color=color,
        )

        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 0.9)

        ax.set_xlabel("Depth", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3)


    plt.suptitle(
        f"Alignment metrics at step {checkpoint}",
        fontsize=14,
    )

    plt.tight_layout()


    plt.savefig(
        os.path.join(
            SAVE_DIR,
            f"checkpoint_{checkpoint}.pdf",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"saved checkpoint_{checkpoint}.pdf")