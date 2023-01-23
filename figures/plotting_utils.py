import matplotlib.pyplot as plt
import numpy as np


# short fun to colorize boxplots
def color_boxplot(thos, bp, colors, alpha=0.7):
    for key in ["boxes", "medians"]:
        for i, b in enumerate(bp[key]):
            b.set(color=colors[i], alpha=alpha)

    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[idx])
        patch.set_alpha(alpha)

    for key in ["whiskers", "caps"]:
        for i in range(thos.shape[1]):
            bp[key][2 * i].set(color=colors[i], alpha=alpha)
            bp[key][2 * i + 1].set(color=colors[i], alpha=alpha)


# plotting 1D marginals, e.g., for showing posterior predictives.
def custom_marginal_plot(
    ax,
    x,
    points,
    x_label,
    points_label,
    color,
    show_xlabels=True,
    labels=None,
    num_bins=10,
    alpha=0.8,
    histtype="stepfilled",
    plot_legend=True,
    handlelength=0.8,
    bbox_to_anchor=(1, 1),
    points_line_style="-",
):
    # Cross validation
    assert x.shape[1] == 7
    assert isinstance(x, np.ndarray)
    assert not show_xlabels or labels
    for idx in range(x.shape[1]):

        axi = ax[idx]
        plt.sca(axi)
        axi.spines["right"].set_visible(False)
        axi.spines["top"].set_visible(False)
        axi.spines["left"].set_visible(False)
        plt.yticks([])
        if points is not None:
            plt.axvline(
                x=points[0, idx], color="k", label=points_label, ls=points_line_style
            )

        plt.xlim([0, 1])
        plt.xticks([0, 1])
        _, bins, _ = plt.hist(
            x[:1000, idx],
            bins=num_bins,
            alpha=alpha,
            color=color,
            label=x_label,
            histtype=histtype,
        )
        if show_xlabels:
            plt.xlabel(labels[idx])
        # plt.sca(ax[-1, -1])
        # plt.axis("off")
        # plt.sca(ax[-1, -2])
        if idx == 6 and plot_legend:
            plt.legend(handlelength=handlelength, bbox_to_anchor=bbox_to_anchor)
