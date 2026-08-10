import json
import numpy as np
import matplotlib.pyplot as plt
import os


JSON_FILE = "alignment_sum_metric_update_with_projected_scion_with_lr_0.01_momentum_0.9_1000_steps.json"
SAVE_DIR = "alignment_sum_update_with_projected_scion_separate_plots"

os.makedirs(SAVE_DIR, exist_ok=True)

with open(JSON_FILE, "r") as f:
    data = json.load(f)


color = "#5f0ae7"

titles = [
    r'Mean alignment of $||\Delta h^{(\ell)} ||_2^2$ and $|| \Delta W^{(\ell)} \phi(h^{(\ell -1)}) ||_2^2 + || W^{(\ell)} \Delta \phi(h^{(\ell -1)}) ||_2^2 + || \Delta W^{(\ell)} \Delta \phi(h^{(\ell -1)}) ||_2^2$',
    r'Mean alignment of $||(\text{clip}(W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)}) ) - W^{(\ell)} ) \phi(h^{(\ell-1)})||_2^2$' +"\n and " + r'$||\text{clip}(W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)}) ) \phi(h^{(\ell-1)})||_2^2 + ||W^{(\ell)} \phi(h^{(\ell-1)})||_2^2$' ,
    r'Mean alignment of $||W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)})||_{\mathcal{S}_\infty}^2$ and $||W^{(\ell)}||_{\mathcal{S}_\infty}^2 + ||\gamma \text{lmo}(M_t^{(\ell)})||_{\mathcal{S}_\infty}^2 $',
    r'Norm of $W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)})$'
]
ylabels = [
   r'$\frac{||\Delta h^{(\ell)} ||_2^2}{|| \Delta W^{(\ell)} \phi(h^{(\ell -1)}) ||_2^2 + || W^{(\ell)} \Delta \phi(h^{(\ell -1)}) ||_2^2 + || \Delta W^{(\ell)} \Delta \phi(h^{(\ell -1)}) ||_2^2}$',
    r'$\frac{||(\text{clip}(W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)}) ) - W^{(\ell)} ) \phi(h^{(\ell-1)})||_2^2}{||\text{clip}(W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)}) ) \phi(h^{(\ell-1)})||_2^2 + ||W^{(\ell)} \phi(h^{(\ell-1)})||_2^2}$',
    r'$\frac{||W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)})||_{\mathcal{S}_\infty}^2}{||W^{(\ell)}||_{\mathcal{S}_\infty}^2 + ||\gamma \text{lmo}(M_t^{(\ell)})||_{\mathcal{S}_\infty}^2}$',
    r'$||W^{(\ell)} + \gamma \text{lmo}(M_t^{(\ell)})||_{\mathcal{S}_\infty}^2$'
    ]


for checkpoint in sorted(data.keys(), key=int):

    fig, axes = plt.subplots(3, 1, figsize=(8, 15))

    metric_info = [
        ("alignment_sum_update_metric", titles[0], ylabels[0]),
        # ("alignment_sum_update_spectral", titles[1], ylabels[1]),
        ("alignment_sum_W_lmo", titles[2], ylabels[2]),
        ("norm_W_lmo", titles[3], ylabels[3])

    ]

    for ax, (metric, title, ylabel) in zip(np.ravel(axes), metric_info):

        metric_data = data[checkpoint][metric]

        widths = sorted(
            [int(d) for d in metric_data.keys()]
        )
        # print("depths", depths)

        means = []
        stds = []

        for width in widths:
            vals = np.asarray(metric_data[str(width)])

            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.asarray(means)
        stds = np.asarray(stds)

        ax.plot(
            widths,
            means,
            color=color,
            marker="o",
            markersize=4,
            linewidth=2,
        )

        ax.fill_between(
            widths,
            means - stds,
            means + stds,
            alpha=0.2,
            color=color,
        )

        ax.set_xscale("log", base=2)
        if ax == axes[1] or ax==axes[2]:
            ax.set_ylim(0.7, 1.3)
        else:
            ax.set_ylim(0, 2)
            


        if ax == axes[-1]:
            ax.set_xlabel("Width", fontsize=12)
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


#########################################################################################################
#########################################################################################################

for checkpoint in sorted(data.keys(), key=int):

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    metric_info = [
        # ("alignment_sum_update_metric", titles[0], ylabels[0]),
        ("alignment_sum_update_spectral", titles[1], ylabels[1]),
        # ("alignment_sum_W_lmo", titles[2], ylabels[2]),
        # ("norm_W_lmo", titles[3], ylabels[3])

    ]

    for ax, (metric, title, ylabel) in zip(np.ravel(axes), metric_info):

        metric_data = data[checkpoint][metric]

        widths = sorted(
            [int(d) for d in metric_data.keys()]
        )
        # print("depths", depths)

        means = []
        stds = []

        for width in widths:
            vals = np.asarray(metric_data[str(width)])

            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.asarray(means)
        stds = np.asarray(stds)

        ax.plot(
            widths,
            means,
            color=color,
            marker="o",
            markersize=4,
            linewidth=2,
        )

        ax.fill_between(
            widths,
            means - stds,
            means + stds,
            alpha=0.2,
            color=color,
        )

        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.1)
            


        ax.set_xlabel("Width", fontsize=12)
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
            f"checkpoint_{checkpoint}_non_cancellation.pdf",
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"saved checkpoint_{checkpoint}_non_cancellation.pdf")