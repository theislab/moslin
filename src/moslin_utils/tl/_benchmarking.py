import warnings
from functools import partial
from typing import Literal, Optional, Tuple, Union

import lineageot.inference as lot_inf
import moscot
import ot

import networkx as nx
import numpy as np
import scipy.sparse as sp

from anndata import AnnData

from moslin_utils.constants import REP_KEY, TIME_KEY
from moslin_utils.ul import (
    compute_tree_distances,
    CouplingInfo,
    ground_truth_coupling,
    remove_unlisted_leaves,
    sort_clusters,
    TreeInfo,
)


def benchmark_lot(
    adata: AnnData,
    early_time: int,
    late_time: int,
    *,
    epsilon: float,
    scale_cost: Optional[Literal["mean", "median", "max"]] = None,
    tau_a: float = 1.0,
) -> Tuple[np.ndarray, bool, CouplingInfo]:
    """Benchmark LineageOT's ability to recover early-to-late cell couplings."""

    def scale(x: np.ndarray) -> np.ndarray:
        if scale_cost is None:
            return x
        if scale_cost == "max_cost":
            return x / np.max(x)
        if scale_cost == "mean":
            return x / np.mean(x)
        if scale_cost == "median":
            return x / np.median(x)
        if isinstance(scale_cost, (int, float)):
            return x / scale_cost
        raise TypeError(type(scale_cost))

    gt_coupling = ground_truth_coupling(adata, early_time, late_time)

    late_data: TreeInfo = adata.uns["trees"][late_time]
    early_cells = adata.uns["trees"][early_time].good_cells
    keep_late_cells = np.asarray(late_data.good_cells)[gt_coupling.late_cells]

    late_tree = remove_unlisted_leaves(late_data.tree, keep_late_cells, max_depth=15)
    lot_inf.add_nodes_at_time(late_tree, early_time)
    lot_inf.add_times_to_edges(late_tree)
    lot_inf.add_conditional_means_and_variances(late_tree, late_data.good_cells)
    ancestor_info = lot_inf.get_ancestor_data(late_tree, early_time)

    cost_matrix = ot.utils.dist(np.asarray(adata[early_cells].obsm[REP_KEY]), ancestor_info[0]) @ np.diag(
        ancestor_info[1] ** (-1)
    )
    cost_matrix = cost_matrix[gt_coupling.early_cells, :]
    cost_matrix = scale(cost_matrix)

    assert cost_matrix.shape == gt_coupling.coupling.shape, (
        cost_matrix.shape,
        gt_coupling.coupling.shape,
    )

    if tau_a == 1.0:
        ot_fn = partial(ot.sinkhorn, method="sinkhorn_log")
    else:
        rho_a = (epsilon * tau_a) / (1 - tau_a)
        ot_fn = partial(
            ot.unbalanced.sinkhorn_unbalanced,
            reg_m=rho_a,
            method="sinkhorn_reg_scaling",
        )

    with warnings.catch_warnings(record=True) as ws:
        coupling = ot_fn(
            a=gt_coupling.early_marginal,
            b=gt_coupling.late_marginal,
            M=cost_matrix,
            numItermax=10_000,
            stopThr=1e-3,
            reg=epsilon,
            warn=True,
        )
    coupling = coupling.astype(np.float64)
    conv = not [w for w in ws if "did not converge" in str(w.message)]

    return coupling, conv and np.all(np.isfinite(coupling)), gt_coupling


def benchmark_moscot(
    adata: AnnData,
    early_time: int,
    late_time: int,
    *,
    alpha: float,
    epsilon: float,
    scale_cost: Union[str, float] = "mean",
    tau_a: float = 1.0,
    max_inner_iterations: int = 30000,
    max_outer_iterations: int = 50,
    min_outer_iterations: int = 5,
    store_inner_errors: bool = False,
    threshold: float = 1e-3,
    reorder_clusters: bool = False,
) -> Tuple[moscot.problems.time.TemporalProblem, np.ndarray, bool, CouplingInfo]:
    """Benchmark moslin's ability to recover early-to-late cell couplings."""
    # get the ground truth coupling and cost matrices
    gt_coupling, early_dist, late_dist, bdata = prepare_moscot(
        adata=adata,
        early_time=early_time,
        late_time=late_time,
        reorder_clusters=reorder_clusters,
    )

    # write marginals to bdata
    bdata.obs["marginals"] = np.r_[gt_coupling.early_marginal, gt_coupling.late_marginal]

    # write cost matrices to bdata
    bdata.obsp["cost_matrices"] = sp.bmat([[early_dist, None], [None, late_dist]], format="csr")

    if alpha == 0:
        prob = moscot.problems.time.TemporalProblem(bdata)
        prob = prob.prepare(
            time_key=TIME_KEY,
            joint_attr={"attr": "obsm", "key": "X_pca"},
            a="marginals",
            b="marginals",
        )
        prob = prob.solve(
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=1.0,
            scale_cost=scale_cost,
            max_iterations=max_inner_iterations,
            threshold=threshold,
        )
    else:
        if alpha == 1:
            joint_attr = None
        else:
            joint_attr = {"attr": "obsm", "key": "X_pca"}

        print(f"joint_attr = {joint_attr}")

        prob = moscot.problems.time.LineageProblem(bdata)
        prob = prob.prepare(
            time_key=TIME_KEY,
            joint_attr=joint_attr,
            lineage_attr={"attr": "obsp", "key": "cost_matrices", "cost": "custom"},
            a="marginals",
            b="marginals",
        )

        prob = prob.solve(
            epsilon=epsilon,
            alpha=alpha,
            tau_a=tau_a,
            tau_b=1.0,
            scale_cost=scale_cost,
            store_inner_errors=store_inner_errors,
            max_iterations=max_outer_iterations,
            min_iterations=min_outer_iterations,
            linear_solver_kwargs={"max_iterations": max_inner_iterations},
            threshold=threshold,
        )

    output = prob.solutions[early_time, late_time]

    return (
        prob,
        np.asarray(output.transport_matrix),
        bool(output.converged),
        gt_coupling,
    )


def prepare_moscot_tree(tree_info: TreeInfo, gt_coupling: CouplingInfo, *, late: bool, early_time: float) -> nx.DiGraph:
    """Prepare lineage tree for moslin analysis."""
    ixs = gt_coupling.late_cells if late else gt_coupling.early_cells
    keep_cells = np.asarray(tree_info.good_cells)[ixs]

    tree = remove_unlisted_leaves(tree_info.tree, keep_cells, max_depth=15)
    n_added = lot_inf.add_nodes_at_time(tree, early_time)
    if late:
        assert n_added > 0, n_added
    else:
        assert n_added == 0, n_added
    lot_inf.add_times_to_edges(tree)

    return tree


def prepare_moscot(
    adata: AnnData,
    early_time: int,
    late_time: int,
    reorder_clusters: bool = False,
) -> Tuple[CouplingInfo, np.ndarray, np.ndarray]:
    """Compute the ground truth coupling and lineage distance matrices."""
    # get the  ground truth coupling
    gt_coupling = ground_truth_coupling(adata, early_time, late_time)
    n_early, n_late = gt_coupling.coupling.shape

    # prepare lineage trees
    early_tree = prepare_moscot_tree(adata.uns["trees"][early_time], gt_coupling, late=False, early_time=early_time)
    late_tree = prepare_moscot_tree(adata.uns["trees"][late_time], gt_coupling, late=True, early_time=early_time)

    # get early and late lineage tree sizes
    n_early_tree = len(lot_inf.get_leaves(early_tree, include_root=False))
    n_late_tree = len(lot_inf.get_leaves(late_tree, include_root=False))

    assert n_early == n_early_tree, (n_early, n_early_tree)
    assert n_late == n_late, (n_late, n_late_tree)

    # compute lineage distance matrices based on trees
    early_dist = compute_tree_distances(early_tree)
    late_dist = compute_tree_distances(late_tree)

    # make sure cells appear in the right order
    src = np.asarray(adata.uns["trees"][early_time].good_cells)[gt_coupling.early_cells]
    tgt = np.asarray(adata.uns["trees"][late_time].good_cells)[gt_coupling.late_cells]

    bdata = adata[src].concatenate(adata[tgt], index_unique=None)

    if reorder_clusters:
        # record the color mapping
        color_dict = {
            state: color for state, color in zip(adata.obs["clusters"].cat.categories, adata.uns["clusters_colors"])
        }

        # reorder clusters
        bdata = sort_clusters(bdata, color_dict=color_dict)

        # fix coarse clusters colors
        bdata.uns["coarse_clusters_colors"] = adata.uns["coarse_clusters_colors"].copy()

    return gt_coupling, early_dist, late_dist, bdata
