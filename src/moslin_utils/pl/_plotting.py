from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

from moslin_utils.constants import TIME_KEY
from moslin_utils.settings import settings
from moslin_utils.ul import get_best_runs


def plot_pie_chart(
    adata,
    state: str,
    figsize=(5, 5),
    pctdistance=1.4,
    labeldistance=1.1,
    fontsize=10,
    annot=True,
    save=None,
):
    """Show pie chart over fine cluster distributions."""
    # subset to the target state on a coarse cluster level
    mask = adata.obs["coarse_clusters"] == state

    print(f"Showing the distribution for {state} (n_cells={np.sum(mask)})")

    # caluculate the distribution of this coarse state over sub-types
    value_counts = adata[mask].obs["plot.cell.type"].value_counts()

    # for the main clusters, obtain their colors
    color_mapping = {
        cluster: color for (cluster, color) in zip(adata.obs["clusters"].cat.categories, adata.uns["clusters_colors"])
    }

    # count how many are not among the main clusters
    n_not_found = np.sum(~value_counts.index.isin(color_mapping.keys()))

    # for these, obtain a range of grey values
    range_of_greys = [mcolors.to_hex(c) for c in plt.get_cmap("Greys")(np.linspace(0.1, 0.9, n_not_found))]

    # iterate over all sub-types, if we have a nice color for them, assign it, if not, assign some shade of grey
    colors = []
    i = 0

    for cl in value_counts.index:
        if cl in color_mapping.keys():
            colors.append(color_mapping[cl])
        else:
            colors.append(range_of_greys[i])
            i += 1

    # plot the distribution as a pie chart
    fig, ax = plt.subplots(figsize=figsize)

    # plot the pie chart
    shared_kwargs = {"startangle": 90, "colors": colors}
    if annot:
        ax.pie(
            value_counts,
            labels=value_counts.index,
            autopct="%1.1f%%",
            textprops={"fontsize": fontsize},
            pctdistance=pctdistance,
            labeldistance=labeldistance,
            **shared_kwargs,
        )
    else:
        ax.pie(value_counts, **shared_kwargs)

    ax.axis("equal")  # equal aspect ratio ensures that pie is drawn as a circle

    plt.tight_layout()

    if save is not None and settings.save_figures:
        plt.savefig(save)
    elif save is not None:
        print("Not saving the figure, as `moslin_utils.settings.save_figures` is set to `False`.")

    plt.show()


def plot_ancestor_descendant_error(adata, early_tp: int, late_tp: int, save: Optional[str] = None):
    """Visualize method difference in terms of ancestor and descendant error in an embedding."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    vmax = adata.obs["diff_early_error"].abs().max()

    sc.pl.umap(
        adata,
        vmin=-vmax,
        vmax=vmax,
        ax=ax0,
        show=False,
        frameon=False,
        alpha=0.6,
        size=80,
    )

    sc.pl.umap(
        adata[adata.obs[TIME_KEY] == late_tp],
        color=["diff_early_error"],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        add_outline=True,
        ax=ax0,
        show=False,
        frameon=False,
        alpha=0.6,
        size=80,
    )

    vmax = adata.obs["diff_late_error"].abs().max()
    sc.pl.umap(
        adata,
        vmin=-vmax,
        vmax=vmax,
        ax=ax1,
        show=False,
        frameon=False,
        alpha=0.6,
        size=80,
    )
    sc.pl.umap(
        adata[adata.obs[TIME_KEY] == early_tp],
        color=["diff_late_error"],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        ax=ax1,
        show=False,
        add_outline=True,
        frameon=False,
        alpha=0.6,
        size=80,
    )

    ax0.set_title("$\\Delta$ ancestor error")
    ax1.set_title("$\\Delta$ descendant error")
    fig.tight_layout()

    if save is not None and settings.save_figures:
        plt.savefig(save)
    elif save is not None:
        print("Not saving the figure, as `moslin_utils.settings.save_figures` is set to `False`.")

    return fig


def plot2D_samples_mat(ax, xs, xt, G, thr=1e-8, alpha_scale=1, **kwargs):
    """Visualize lines connecting cells in an embedding."""
    mx = G.max()
    sources = []
    targets = []
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            val = G[i, j] / mx
            if val > thr:
                ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], alpha=alpha_scale * val, **kwargs)
                sources.append(i)
                targets.append(j)
    return sources, targets


def plot_coupling(
    adata,
    coupling: np.ndarray,
    early_tp: int,
    late_tp: int,
    mask: Optional[List[bool]] = None,
    cats: Optional[Union[str, List[str]]] = None,
    key: Optional[str] = None,
    line_color: str = "k",
    dot_color: str = "lineage",
    thr: float = 0.1,
    alpha_scale: float = 0.6,
    ax=None,
) -> None:
    """Visualize predicted couplings via lines in an embedding."""
    if ax is None:
        fig, ax = plt.subplots()

    src_mask = adata.obs[TIME_KEY] == early_tp
    tgt_mask = adata.obs[TIME_KEY] == late_tp
    emb_src = adata.obsm["X_umap"][src_mask]
    emb_tgt = adata.obsm["X_umap"][tgt_mask]

    if mask is None:
        print(f"Using the {key} for subsetting")
        if isinstance(cats, str):
            cats = [cats]
        final_mask = adata[src_mask].obs[key].isin(cats)
    else:
        print("Using the mask for subsetting")
        final_mask = mask[src_mask]

    coupling = coupling[final_mask, :]
    emb_src = emb_src[final_mask]

    sc.pl.umap(adata, frameon=False, alpha=0.6, size=80, ax=ax, show=False, zorder=0)

    # draw the lines
    src, tgt = plot2D_samples_mat(
        ax,
        emb_src,
        emb_tgt,
        coupling,
        thr=thr,
        alpha_scale=alpha_scale,
        zorder=1,
        color=line_color,
    )
    # outline source points
    sc.pl.umap(
        adata[src_mask][final_mask][np.unique(src), :],
        color=dot_color,
        legend_loc="none",
        alpha=0.3,
        frameon=False,
        size=80,
        ax=ax,
        show=False,
        add_outline=True,
    )
    # outline target points
    sc.pl.umap(
        adata[tgt_mask][np.unique(tgt), :],
        color=dot_color,
        legend_loc="none",
        alpha=0.3,
        frameon=False,
        size=80,
        ax=ax,
        show=False,
        add_outline=True,
    )

    if ax is None:
        fig.tight_layout()
        return fig


def gridsearch_heatmap(
    df: pd.DataFrame,
    lineage_info_key: str = "lineage_info",
    lineage_info: Literal["precise", "abpxp"] = "abpxp",
    parameter_1: str = "alpha",
    parameter_2: str = "epsilon",
    metric: str = "mean_error",
    method_key: str = "kind",
    method: Literal["moslin", "W", "GW", "LineageOT"] = "moslin",
    convergence_key: str = "converged",
    converged: Optional[bool] = True,
    save: Optional[str] = None,
    n_col: int = 3,
):
    # Let's print some stuff
    print(f"Lineage info: {lineage_info}")

    # filter based on the method, convergence and lineage info
    mask1 = df[method_key] == method
    if converged is not None:
        mask2 = df[convergence_key] == converged
    else:
        mask2 = df[convergence_key].notnull()
    mask3 = df[lineage_info_key] == lineage_info

    # combine both masks and subset the dataframe
    mask = mask1 & mask2 & mask3
    df = df.loc[mask].copy()

    # get the timepoints and sort them
    timepoints = sorted(df["tp"].unique(), key=lambda x: int(x.split("-")[0]))

    # get the number of rows and columns for the grid
    n_row = int(np.ceil(len(timepoints) / n_col))
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 10, n_row * 8), sharex="col", sharey="row")

    # Create a colormap that goes from blue to red and uses grey for NaN values
    cmap = plt.cm.coolwarm
    cmap.set_bad(color="grey")

    vmin = df["mean_error"].min()
    vmax = df["mean_error"].max()

    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    for ax, timepoint in zip(axs.flat, timepoints):
        group_df = df[df["tp"] == timepoint]

        pivot_df = group_df.pivot(index=parameter_1, columns=parameter_2, values=metric)

        sns.heatmap(
            pivot_df,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            ax=ax,
            cbar_ax=cbar_ax,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Mean error"},
        )
        # im = ax.imshow(pivot_df, cmap=cmap, vmin=vmin, vmax=vmax)

        # Find the coordinates of the cell with the minimum value, ignoring NaN values
        min_val_index = np.unravel_index(np.nanargmin(pivot_df.values), pivot_df.values.shape)

        # Add a rectangle around the cell with the minimum value
        rect = patches.Rectangle(
            (min_val_index[1], min_val_index[0]), 1, 1, linewidth=2, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        ax.set_title(f"{timepoint} min")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Alpha")

    # Remove any unused subplots
    for ax in axs.flat[len(timepoints) :]:
        ax.remove()

    fig.subplots_adjust(right=0.9)

    if save is not None and settings.save_figures:
        plt.savefig(save)
    elif save is not None:
        print("Not saving the figure, as `moslin_utils.settings.save_figures` is set to `False`.")

    plt.show()


def gridsearch_bar(
    df: pd.DataFrame,
    lineage_info: Literal["precise", "abpxp"],
    metric: str = "mean_error",
    aim: Literal["min", "max"] = "min",
    group_key: str = "kind",
    convergence_key: str = "converged",
    converged: Optional[bool] = None,
    alpha_min: Optional[float] = None,
    ylim: Optional[List[float]] = None,
    save: Optional[str] = None,
):
    # filter to the best performing runs
    df = get_best_runs(
        df=df,
        lineage_info=lineage_info,
        metric=metric,
        aim=aim,
        group_key=group_key,
        convergence_key=convergence_key,
        converged=converged,
        alpha_min=alpha_min,
    )

    # print the best-performing runs
    print(df[["alpha", "kind", "tp", "epsilon", "scale_cost", "mean_error"]])

    # plot
    fig, ax = plt.subplots(dpi=400)
    _ = sns.barplot(
        df,
        x="tp",
        y="mean_error",
        hue="kind",
        ax=ax,
        palette=["#40928A", "#85BCD9", "#1A5A29", "#C8AB66"],
    )
    ax.set_title(lineage_info)
    ax.set_xlabel("time point")
    ax.set_ylabel("mean error")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    # Set y-axis range
    if ylim is not None:
        ax.set_ylim(ylim)

    if save is not None and settings.save_figures:
        plt.savefig(save)
    elif save is not None:
        print("Not saving the figure, as `moslin_utils.settings.save_figures` is set to `False`.")

    plt.show()
