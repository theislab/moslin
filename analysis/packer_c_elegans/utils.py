import jax

jax.config.update("jax_enable_x64", True)
import copy
import warnings
from functools import partial
from typing import Any, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

import lineageot.evaluation as lot_eval
import lineageot.inference as lot_inf
import lineageot.simulation as lot_sim
import moscot
import networkx as nx
import numpy as np
import ot
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

__all__ = ["preprocess", "benchmark_lot", "benchmark_moscot"]

TIME_KEY = "assigned_batch_time"
REP_KEY = "X_pca"


class TreeInfo(NamedTuple):
    tree: nx.DiGraph
    good_cells: List[str]
    bad_cells: List[str]


class CouplingInfo(NamedTuple):
    coupling: np.ndarray
    early_rna_cost: np.ndarray
    late_rna_cost: np.ndarray
    early_cells: List[int]
    late_cells: List[int]

    def cost(
        self, cmat: np.ndarray, *, late: bool, scale: Optional[float] = None
    ) -> float:
        if late:
            cmat = lot_eval.expand_coupling(cmat, self.coupling, self.late_rna_cost)
            cost = lot_inf.OT_cost(cmat, self.late_rna_cost)
            if scale is None:
                scale = self.ind_late_cost
        else:
            cmat = lot_eval.expand_coupling(
                cmat.T, self.coupling.T, self.early_rna_cost
            ).T
            cost = lot_inf.OT_cost(cmat, self.early_rna_cost)
            if scale is None:
                scale = self.ind_early_cost

        return cost / scale

    @property
    def ind_coupling(self) -> np.ndarray:
        return np.outer(self.early_marginal, self.late_marginal)

    @property
    def early_marginal(self) -> np.ndarray:
        return np.sum(self.coupling, axis=1)

    @property
    def late_marginal(self) -> np.ndarray:
        return np.sum(self.coupling, axis=0)

    @property
    def ind_early_cost(self) -> float:
        return self.cost(self.ind_coupling, late=False, scale=1.0)

    @property
    def ind_late_cost(self) -> float:
        return self.cost(self.ind_coupling, late=True, scale=1.0)


def preprocess(
    adata: AnnData,
    ref_tree: nx.DiGraph,
    *,
    lineage_info: Optional[Literal["precise", "abpxp"]],
    seed: int = 0,
) -> AnnData:
    # Removing partially lineage-labeled cells
    if lineage_info == "precise":
        keep_mask = (
            adata.obs["lineage"].to_numpy()
            == adata.obs["random_precise_lineage"].to_numpy()
        )
        adata = adata[keep_mask].copy()
    elif lineage_info == "abpxp":
        keep_mask = adata.obs["lineage"].str.startswith("ABpxp")
        adata = adata[keep_mask].copy()

    adata.obs.index = adata.obs["cell"]
    adata = adata[adata.obs.index.sort_values()].copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    sc.pp.pca(adata)

    adata.obs[TIME_KEY] = adata.obs["embryo.time.bin"].apply(assign_time_to_bin)
    adata.obs[TIME_KEY] = adata.obs[TIME_KEY].astype("category")
    adata.obs = adata.obs.set_index("cell")

    rng = np.random.RandomState(seed)
    trees = {}
    for batch_time in adata.obs[TIME_KEY].cat.categories:
        seed = rng.randint(0, 2**16 - 1)
        trees[batch_time] = create_lineage_tree(
            adata, batch_time=batch_time, reference_tree=ref_tree, seed=seed
        )

    adata.uns["trees"] = trees

    return adata


def assign_time_to_bin(bin: str) -> int:
    if bin == "< 100":
        return 75
    if bin == "> 650":
        # arbitrary choice here
        return 700
    # end of time range
    return 0 * int(bin[:3]) + 1 * int(bin[-3:])


def remove_unlisted_leaves(
    tree: nx.DiGraph, kept_leaves: Sequence[Any], max_depth: int = 10
) -> nx.DiGraph:
    tree = copy.deepcopy(tree)
    for _ in range(max_depth):
        all_leaves = lot_inf.get_leaves(tree, include_root=False)
        for leaf in all_leaves:
            if leaf not in kept_leaves:
                tree.remove_node(leaf)

    return tree


def create_lineage_tree(
    adata: AnnData,
    *,
    batch_time: int,
    reference_tree: nx.DiGraph,
    seed: Optional[int] = None,
) -> TreeInfo:
    selected_cells = adata[adata.obs[TIME_KEY] == batch_time]

    # no need to remove any nodes from the reference
    # (unobserved subtrees have no effect on the inference)
    new_tree = copy.deepcopy(reference_tree)

    rng = np.random.RandomState(seed)
    good_cells, bad_cells = [], []
    for cell in selected_cells.obs.index:
        cell_label = selected_cells.obs["lineage"][cell]
        cell_index = adata.obs.index.get_loc(cell)

        if "x" in cell_label:
            if "/" in cell_label:
                cell_label = rng.choice(cell_label.split("/"))
            if "x" in cell_label:
                cell_label = cell_label.replace("x", rng.choice(["l", "r"]))

        parent = next(reference_tree.predecessors(cell_label))
        if batch_time <= new_tree.nodes[parent]["time"]:
            # filter this cell out
            bad_cells.append(cell)
            continue
        else:
            good_cells.append(cell)

        new_tree.add_node(cell)
        new_tree.add_edge(parent, cell)

        new_tree.nodes[cell]["name"] = cell_label
        new_tree.nodes[cell]["time"] = batch_time
        new_tree.nodes[cell]["time_to_parent"] = (
            batch_time - new_tree.nodes[parent]["time"]
        )
        new_tree.nodes[cell]["cell"] = lot_sim.Cell(
            adata.obsm[REP_KEY][cell_index, :], cell_label
        )

        assert new_tree.nodes[cell]["time_to_parent"] >= 0

    return TreeInfo(new_tree, good_cells, bad_cells)


def ground_truth_coupling(
    adata: AnnData, early_time: int, late_time: int
) -> CouplingInfo:
    def is_ancestor(late_cell: str, early_cell: str) -> bool:
        if late_cell[-1] not in "aplrdvx":
            warnings.warn(
                "Ancestor checking not implemented for cell " + late_cell + " yet."
            )
            return False
        return early_cell in late_cell

    edata: TreeInfo = adata.uns["trees"][early_time]
    ldata: TreeInfo = adata.uns["trees"][late_time]

    n_early = len(edata.good_cells)
    n_late = len(ldata.good_cells)
    coupling = np.zeros([n_early, n_late], dtype=float)

    for i, c_early in enumerate(edata.good_cells):
        for j, c_late in enumerate(ldata.good_cells):
            if is_ancestor(
                ldata.tree.nodes[c_late]["name"], edata.tree.nodes[c_early]["name"]
            ):
                coupling[i, j] = 1.0

    # filter out zero rows and columns
    kept_early_cells = np.where(np.sum(coupling, 1) > 0)[0]
    kept_late_cells = np.where(np.sum(coupling, 0) > 0)[0]
    coupling = coupling[np.ix_(kept_early_cells, kept_late_cells)]

    # normalize to uniform marginal on late cells
    coupling = np.dot(coupling, np.diag(np.sum(coupling, axis=0) ** -1)) / len(
        kept_late_cells
    )

    erep = np.asarray(adata[edata.good_cells].obsm[REP_KEY])
    lrep = np.asarray(adata[ldata.good_cells].obsm[REP_KEY])
    early_rna_cost = ot.utils.dist(erep, erep)[
        np.ix_(kept_early_cells, kept_early_cells)
    ]
    late_rna_cost = ot.utils.dist(lrep, lrep)[np.ix_(kept_late_cells, kept_late_cells)]

    assert early_rna_cost.shape[0] == coupling.shape[0]
    assert late_rna_cost.shape[0] == coupling.shape[1]

    return CouplingInfo(
        coupling, early_rna_cost, late_rna_cost, kept_early_cells, kept_late_cells
    )


def compute_tree_distances(tree: nx.DiGraph) -> np.ndarray:
    leaves = lot_inf.get_leaves(tree, include_root=False)
    num_leaves = len(leaves)
    distances = np.zeros([num_leaves, num_leaves])
    for i, leaf in enumerate(leaves):
        distance_dictionary, tmp = nx.multi_source_dijkstra(
            tree.to_undirected(), [leaf], weight="time"
        )
        for j, target_leaf in enumerate(leaves):
            distances[i, j] = distance_dictionary[target_leaf]
    return distances


def benchmark_lot(
    adata: AnnData,
    early_time: int,
    late_time: int,
    *,
    epsilon: float,
    scale_cost: Optional[Literal["mean", "median", "max"]] = None,
    tau_a: float = 1.0,
) -> Tuple[np.ndarray, bool, CouplingInfo]:
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

    cost_matrix = ot.utils.dist(
        np.asarray(adata[early_cells].obsm[REP_KEY]), ancestor_info[0]
    ) @ np.diag(ancestor_info[1] ** (-1))
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


def prepare_moscot_tree(
    tree_info: TreeInfo, gt_coupling: CouplingInfo, *, late: bool, early_time: float
) -> nx.DiGraph:
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


def benchmark_moscot(
    adata: AnnData,
    early_time: int,
    late_time: int,
    *,
    alpha: float,
    epsilon: float,
    scale_cost: Union[str, float] = "max_cost",
    tau_a: float = 1.0,
) -> Tuple[moscot.problems.time.TemporalProblem, np.ndarray, bool, CouplingInfo]:
    gt_coupling = ground_truth_coupling(adata, early_time, late_time)
    n_early, n_late = gt_coupling.coupling.shape

    early_tree = prepare_moscot_tree(
        adata.uns["trees"][early_time], gt_coupling, late=False, early_time=early_time
    )
    late_tree = prepare_moscot_tree(
        adata.uns["trees"][late_time], gt_coupling, late=True, early_time=early_time
    )

    n_early_tree = len(lot_inf.get_leaves(early_tree, include_root=False))
    n_late_tree = len(lot_inf.get_leaves(late_tree, include_root=False))

    assert n_early == n_early_tree, (n_early, n_early_tree)
    assert n_late == n_late, (n_late, n_late_tree)

    src = np.asarray(adata.uns["trees"][early_time].good_cells)[gt_coupling.early_cells]
    tgt = np.asarray(adata.uns["trees"][late_time].good_cells)[gt_coupling.late_cells]

    bdata = adata[src].concatenate(adata[tgt], index_unique=None)
    bdata.obs["marginals"] = np.r_[
        gt_coupling.early_marginal, gt_coupling.late_marginal
    ]
    bdata.obs["time"] = bdata.obs["assigned_batch_time"].astype(float)

    early_dist = compute_tree_distances(early_tree)
    late_dist = compute_tree_distances(late_tree)
    bdata.obsp["cost_matrices"] = sp.bmat(
        [[early_dist, None], [None, late_dist]], format="csr"
    )

    if alpha == 0:
        prob = moscot.problems.time.TemporalProblem(bdata)
        prob = prob.prepare(
            time_key="time",
            joint_attr={"attr": "obsm", "key": "X_pca"},
            scale_cost=scale_cost,
            a="marginals",
            b="marginals",
        )
    else:
        prob = moscot.problems.time.LineageProblem(bdata)
        prob = prob.prepare(
            time_key="time",
            joint_attr={"attr": "obsm", "key": "X_pca"},
            lineage_attr={"attr": "obsp", "key": "cost_matrices"},
            scale_cost=scale_cost,
            a="marginals",
            b="marginals",
        )
    prob = prob.solve(epsilon=epsilon, alpha=alpha, tau_a=tau_a, tau_b=1.0)
    output = prob.solutions[early_time, late_time]

    return (
        prob,
        np.asarray(output.transport_matrix),
        bool(output.converged),
        gt_coupling,
    )
