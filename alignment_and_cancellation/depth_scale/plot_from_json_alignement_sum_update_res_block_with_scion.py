import json
import numpy as np
import matplotlib.pyplot as plt
import os


JSON_FILE = "alignment_sum_metric_update_res_block_with_scion_with_h_apply_Z_lr_0.01_momentum_0.9_1000_steps.json"
SAVE_DIR = "alignment_sum_update_res_block_with_scion"

os.makedirs(SAVE_DIR, exist_ok=True)

with open(JSON_FILE, "r") as f:
    data = json.load(f)


color = "#75e70a"

titles = [
    r'Mean alignment of $|| \Delta_{W^{(\ell)}, Z^{(\ell)}} h^{(\ell)} ||_2^2$ and $|| \Delta Z^{(\ell)} h^{(\ell -1)} ||_2^2 + \frac{1}{L^{2\alpha}} || \Delta W^{(\ell)} \phi(h^{(\ell -1)}) ||_2^2$',
    r'Mean alignment of $||\Delta W^{(\ell)} \phi(h^{(\ell-1)})||_2^2$ and $|| \gamma * W^{(\ell)} \phi(h^{(\ell-1)})||_2^2 + || \gamma * lmo({M_t^{W^{(\ell)}}}) \phi(h^{(\ell-1)})||_2^2$',
    r'Mean alignment of $||\Delta Z^{(\ell)} h^{(\ell-1)}||_2^2 $ and $|| \gamma_Z * Z^{(\ell)} h^{(\ell-1)}||_2^2 + || \gamma_Z *  lmo({M_t^{Z^{(\ell)}}})   h^{(\ell-1)}||_2^2$',
]
ylabels = [
    r'$\frac{|| \Delta_{W^{(\ell)}, Z^{(\ell)}} h^{(\ell)} ||_2^2}{|| \Delta Z^{(\ell)} h^{(\ell -1)} ||_2^2 + \frac{1}{L^{2\alpha}} || \Delta W^{(\ell)} \phi(h^{(\ell -1)}) ||_2^2}$',
    r'$\frac{||\Delta W^{(\ell)} \phi(h^{(\ell-1)})||_2^2}{|| \gamma * W^{(\ell)} \phi(h^{(\ell-1)})||_2^2 + || \gamma * lmo({M_t^{W^{(\ell)}}}) \phi(h^{(\ell-1)})||_2^2}$',
    r'$\frac{||\Delta Z^{(\ell)} h^{(\ell-1)}||_2^2}{(|| \gamma_Z * Z^{(\ell)} h^{(\ell-1)}||_2^2 + || \gamma_Z * lmo({M_t^{Z^{(\ell)}}}) h^{(\ell-1)}||_2^2}$'
    ]


for checkpoint in sorted(data.keys(), key=int):

    fig, axes = plt.subplots(3, 1, figsize=(8, 15))

    metric_info = [
        ("alignment_sum_metric_update_res_block", titles[0], ylabels[0]),
        ("alignment_sum_update_metric_W", titles[1], ylabels[1]),
        ("alignment_sum_update_metric_Z", titles[2], ylabels[2])

    ]

    for ax, (metric, title, ylabel) in zip(np.ravel(axes), metric_info):

        metric_data = data[checkpoint][metric]

        depths = sorted(
            [int(d) for d in metric_data.keys()]
        )
        # print("depths", depths)

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
        ax.set_ylim(0, 2)

        if ax == axes[-1]:
            ax.set_xlabel("Depth L", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3)

        ax.axhline(y=1, linestyle='--', color= "#fdd90d", alpha=0.5, lw=3)


    plt.suptitle(
        f"Alignment metrics at step {checkpoint} \n",
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