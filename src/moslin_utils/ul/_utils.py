from itertools import chain
from typing import Literal, Optional, Tuple, Union

from cellrank.estimators import CFLARE, GPCCA
from cellrank.estimators.mixins._utils import StatesHolder

import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from anndata import AnnData

from moslin_utils.constants import KEY_STATE_COLORS, TIME_KEY


def sort_clusters(
    adata: AnnData,
    color_dict: Optional[dict[str, str]] = None,
) -> AnnData:
    """Reorder "clusters" by "coarse_clusters"."""
    # create a lookup table to link coarse clusters to clusters
    df = (adata.obs.groupby(["clusters", "coarse_clusters"]).size() > 0).unstack()
    lookup = {state: list(df.loc[df[state]].index) for state in adata.obs["coarse_clusters"].cat.categories}

    # use the lookup to get an ordered list of states
    ordered_clusters = list(chain(*list(lookup.values())))

    # for later, record the colors
    if color_dict is None:
        color_dict = {
            state: color for state, color in zip(adata.obs["clusters"].cat.categories, adata.uns["clusters_colors"])
        }

    # actually change cluster order
    adata.obs["clusters"] = adata.obs["clusters"].cat.reorder_categories(ordered_clusters)

    # take care of colors
    adata.uns["clusters_colors"] = [color_dict[state] for state in adata.obs["clusters"].cat.categories]

    return adata


def assign_time_to_bin(bin: str) -> int:
    """Discretize time assignments."""
    if bin == "< 100":
        return 75
    if bin == "> 650":
        # arbitrary choice here
        return 700
    # end of time range
    return 0 * int(bin[:3]) + 1 * int(bin[-3:])


def get_best_runs(
    df: pd.DataFrame,
    lineage_info: Literal["precise", "abpxp"],
    metric: str = "mean_error",
    aim: Literal["min", "max"] = "min",
    group_key: str = "kind",
    group: Optional[str] = None,
    convergence_key: str = "converged",
    converged: Optional[bool] = None,
    alpha_min: Optional[float] = None,
) -> pd.DataFrame:
    """Obtain the best runs from a gridsearch."""
    # subset to the requested type of lineage info
    df = df.loc[df["lineage_info"] == lineage_info.lower()].copy()

    # subset to the requested group
    if group is not None:
        df = df.loc[df[group_key] == group].copy()

    # potentially filter to converged runs
    mask = np.array(df[convergence_key])
    if converged is not None:
        if converged:
            print(f"Removing {np.sum(~mask)}/{df.shape[0]} not converged runs.")
            df = df[mask].copy()
        else:
            print(f"Removing {np.sum(mask)}/{df.shape[0]} converged runs.")
            df = df[~mask].copy()
    else:
        print(f"Returning all {df.shape[0]} runs ({np.sum(~mask)} did not converge).")

    # potentially filter to runs with a minimum alpha
    if alpha_min is not None:
        mask = np.logical_or(df["alpha"] >= alpha_min, df["alpha"] == 0)
        print(f"Removing {np.sum(~mask)}/{df.shape[0]} runs with alpha < {alpha_min}.")
        df = df[mask].copy()

    # for each model and time point, get the index corresponding to the minimal cost
    if aim == "max":
        ixs = df.groupby([group_key, "tp"])[metric].idxmax()
    elif aim == "min":
        ixs = df.groupby([group_key, "tp"])[metric].idxmin()
    else:
        raise ValueError("Unrecognized aim: {aim}.")

    # subset to these entries
    df = df.loc[ixs.reset_index()[metric]]

    return df


def get_adata_and_tmats(
    result_dict: dict[tuple[int, int], AnnData],
) -> Tuple[AnnData, dict[str:AnnData]]:
    """Reformat transition matrices.

    Convert the raw list of AnnDatas into a list of transition matrices and
    large, concatenated AnnData object.
    """
    # get a list of all timepoints
    timepoints = list(np.unique(list(chain(*result_dict.keys()))))

    tmats = {}
    a_dict, b_dict = {}, {}
    for i, ((t1, t2), bdata) in enumerate(result_dict.items()):
        src_mask = bdata.obs[TIME_KEY] == t1
        tgt_mask = bdata.obs[TIME_KEY] == t2
        tmat = bdata.obsp["pred"][src_mask, :][:, tgt_mask]

        # store the transition matrix as an N x M AnnData object
        tmats[t1, t2] = AnnData(tmat, obs=bdata[src_mask].obs, var=bdata[tgt_mask].obs)

        # this makes sure that cells which get zero mass in the target marginal are not included
        # in the subsequent source marginal
        a_dict[i + 1] = bdata[src_mask]
        b_dict[i + 1] = bdata[tgt_mask]

    subsets = [
        a_dict[1],  # all cells from the first source marginal
        b_dict[1][
            b_dict[1].obs_names.intersection(a_dict[2].obs_names)
        ],  # cells in the target marginal which are also in the next source marginal
        b_dict[2][b_dict[2].obs_names.intersection(a_dict[3].obs_names)],
        b_dict[3][b_dict[3].obs_names.intersection(a_dict[4].obs_names)],
        b_dict[4][b_dict[4].obs_names.intersection(a_dict[5].obs_names)],
        b_dict[5][b_dict[5].obs_names.intersection(a_dict[6].obs_names)],
        b_dict[6],  # all cells from the last target marginal
    ]
    for i, (t1, t2) in enumerate(zip(timepoints[:-1], timepoints[1:])):
        tmats[t1, t2] = tmats[t1, t2][subsets[i].obs_names, subsets[i + 1].obs_names]

    # concatenate the AnnData's
    adata = subsets[0].concatenate(subsets[1:], batch_key="time_point")
    adata.obs_names = adata.obs_names.str.split("-").str[:2].str.join("-")
    assert len(set(adata.obs_names)) == adata.n_obs

    # make timepoints categorical
    adata.obs["time_point"] = (
        adata.obs["time_point"]
        .cat.rename_categories(dict(zip(map(str, range(len(timepoints))), timepoints)))
        .astype("category")
    )
    return adata, tmats


def prettify(adata: AnnData):
    """Prettify AnnData objects."""
    # assign time-point colors
    timepoints = adata.obs["time_point"].cat.categories
    adata.uns["time_point_colors"] = [
        mcolors.to_hex(c) for c in plt.get_cmap("gnuplot")(np.linspace(0, 1, len(timepoints)))
    ]

    # make timepoints categorical
    adata.obs["raw.embryo.time"] = adata.obs["raw.embryo.time"].astype("category")
    tps_raw = adata.obs["raw.embryo.time"].cat.categories

    # assign colors to raw time points
    adata.uns["raw.embryo.time_colors"] = [
        mcolors.to_hex(c) for c in plt.get_cmap("gnuplot")(np.linspace(0, 1, len(tps_raw)))
    ]

    # make them categorical again
    adata.obs["coarse_clusters"] = adata.obs["coarse_clusters"].astype("category")

    # write down the desired cluster order
    nice_order = [
        "Ciliated neuron",
        "Ciliated preterminal neuron",
        "Non-ciliated neuron",
        "Non-ciliated preterminal neuron",
        "Glia and excretory",
        "Preterminal glia and excretory",
        "Other terminal cell",
        "Other preterminal cell",
        "Progenitor cell",
    ]
    adata.obs["coarse_clusters"] = adata.obs["coarse_clusters"].cat.reorder_categories(nice_order)

    return adata


def return_drivers(
    adata,
    state: str,
    q_thresh: Optional[float] = None,
    n_genes: int = 20,
    only_tf: bool = True,
):
    """Return a sorted df of driver genes, subsetted to the relevant columns."""
    cols = [
        "Ensembl",
        "highly_variable",
        "Family",
        "TF",
        "Protein",
        f"{state}_pval",
        f"{state}_qval",
        f"{state}_corr",
        "means",
        "dispersions",
    ]

    if q_thresh is not None:
        mask = adata.var[f"{state}_qval"] < q_thresh
    else:
        mask = np.ones(adata.n_vars).astype(bool)

    if only_tf:
        gene_df = (
            adata[:, mask].var.loc[adata.var["TF"]].sort_values(by=f"{state}_corr", ascending=False).head(n_genes)[cols]
        )
    else:
        gene_df = adata[:, mask].var.sort_values(by=f"{state}_corr", ascending=False).head(n_genes)[cols]
    return gene_df


def get_state_purity(adata, estimator, states: Literal["macrostates", "terminal_states"], obs_col: str):
    """Calculate purity of each state of a state type (e.g. each macrostate).

    Note: this code has been adapted from Philipp Weiler, see CellRank2's reproducility repo
    for the original: https://github.com/theislab/cellrank2_reproducibility/tree/main.

    """
    states = getattr(estimator, states)

    max_obs_count_per_state = (
        pd.DataFrame({"states": states, "obs_col": adata.obs[obs_col]})[~states.isnull()]
        .groupby(["states", "obs_col"])
        .size()
        .reset_index()
        .rename(columns={0: "group_counts"})[["states", "group_counts"]]
        .groupby("states")
        .max()["group_counts"]
    )

    return (max_obs_count_per_state / states.value_counts()).to_dict()


def get_state_time(adata: AnnData, estimator: Union[GPCCA, CFLARE], states="macrostates", obs_col="time_point") -> dict:
    """Calculate the mean time of each state of a state type (e.g. each macrostate).

    Note: this code has been adapted from Philipp Weiler, see CellRank2's reproducility repo
    for the original: https://github.com/theislab/cellrank2_reproducibility/tree/main.
    """
    # get marcostates
    states = getattr(estimator, states)

    # create dataframe with the relevant information
    df = (
        pd.DataFrame({"states": states, "obs_col": adata.obs[obs_col]})[~states.isnull()]
        .groupby(["states", "obs_col"])
        .size()
        .reset_index()
        .rename(columns={0: "group_counts"})
    )

    # multiply (timepoint) x (n_cells)
    df["sum"] = df["obs_col"].astype(int) * df["group_counts"]

    # divide by the number of cells per state to get the mean
    df = (
        (df.groupby("states")["sum"].sum() / df.groupby("states")["group_counts"].sum())
        .reset_index()
        .rename(columns={0: "mean"})
        .set_index("states")
    )

    final_dict = df.to_dict(orient="dict")["mean"]

    return final_dict


def sort_and_aggregate_macrostates(
    adata: AnnData,
    estimator: Union[GPCCA, CFLARE],
    cluster_key_l1: str = "coarse_clusters",
    cluster_key_l2: str = "clusters",
) -> Tuple[Union[GPCCA, CFLARE], Union[GPCCA, CFLARE]]:
    estimator_fine = estimator.copy()

    original_macrostates = estimator_fine.macrostates.cat.categories

    # Define macrostate order
    df = (adata.obs.groupby([cluster_key_l2, cluster_key_l1]).size() > 0).unstack()
    lookup = {state: list(df.loc[df[state]].index) for state in adata.obs[cluster_key_l1].cat.categories}
    ordered_clusters = list(chain(*list(lookup.values())))

    # Define a mapping from base categories to derived categories
    category_mapping = {}

    for base_category in ordered_clusters:
        derivatives = [cat for cat in original_macrostates if cat.startswith(base_category)]
        if derivatives:
            category_mapping[base_category] = derivatives

    # Sort macrostates accordingly
    ordered_macrostates = []

    for _, derivatives in category_mapping.items():
        # Sort derivatives by their suffix number (if any) to maintain relative order
        derivatives_sorted = sorted(
            derivatives, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
        )
        ordered_macrostates.extend(derivatives_sorted)

    # Modify the order of macrostates in the GPCCA object accordingly
    estimator_fine._macrostates = StatesHolder(
        assignment=estimator_fine._macrostates.assignment.cat.reorder_categories(ordered_macrostates),
        probs=None,
        memberships=estimator_fine._macrostates.memberships[ordered_macrostates],
        colors=estimator_fine._macrostates.memberships[ordered_macrostates].colors,
    )

    estimator_coarse = estimator_fine.copy()

    # Aggregate macrostates into coarse states
    state_dict = {}

    for group in KEY_STATE_COLORS.keys():
        base_states = lookup[group]
        mask = [any(str(item).startswith(cat) for cat in base_states) for item in estimator_coarse.macrostates]
        state_dict[group] = list(estimator_coarse.macrostates[mask].index)

    # Manually set these to be terminal
    estimator_coarse.set_terminal_states(
        states=state_dict,
        cluster_key=cluster_key_l1,
    )

    # Sort terminal states accordingly
    estimator_coarse._term_states = StatesHolder(
        assignment=estimator_coarse._term_states.assignment.cat.reorder_categories(KEY_STATE_COLORS.keys()),
        probs=None,
        memberships=None,
        colors=KEY_STATE_COLORS.values(),
    )

    return estimator_fine, estimator_coarse


def permute_symmetric_matrix_elements(matrix: np.array, x_percent: float):
    # Create a copy of the matrix to ensure the original is not modified
    matrix_copy = np.copy(matrix)
    n = matrix_copy.shape[0]
    # Adjust to exclude diagonal elements from the total count in the upper triangular part
    total_elements_upper = n * (n + 1) // 2 - n

    # Calculate the number of elements to permute, excluding diagonal
    num_elements_to_permute = int(total_elements_upper * (x_percent / 100))

    # Generate all possible indices for the upper triangular part, excluding diagonal
    upper_tri_indices = np.array([(i, j) for i in range(n) for j in range(i, n) if i != j])

    # Randomly select indices to permute using NumPy
    selected_indices = upper_tri_indices[
        np.random.choice(upper_tri_indices.shape[0], num_elements_to_permute, replace=False)
    ]

    # Extract values to permute
    values_to_permute = matrix_copy[selected_indices[:, 0], selected_indices[:, 1]]

    # Shuffle the values using NumPy
    np.random.shuffle(values_to_permute)

    # Assign the shuffled values back, maintaining symmetry, and track changes
    modified_count = 0
    for (i, j), value in zip(selected_indices, values_to_permute):
        if matrix_copy[i, j] != value:  # Check if the value is actually changing
            modified_count += 2  # Each modification affects two elements
            matrix_copy[i, j] = value
            matrix_copy[j, i] = value

    # Calculate the total number of elements in the matrix, excluding diagonal
    total_elements = n**2 - n

    # Calculate the actual percentage of elements modified, excluding diagonal
    actual_modified_percent = (modified_count / total_elements) * 100

    return matrix_copy, actual_modified_percent
