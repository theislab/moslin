import copy
import warnings
from typing import Any, List, Literal, NamedTuple, Optional, Sequence

import lineageot.evaluation as lot_eval
import lineageot.inference as lot_inf
import lineageot.simulation as lot_sim
import ot
import tqdm

import networkx as nx
import numpy as np

from anndata import AnnData

from moslin_utils.constants import REP_KEY, TIME_KEY


class TreeInfo(NamedTuple):
    """Hold the tree."""

    tree: nx.DiGraph
    good_cells: List[str]
    bad_cells: List[str]


class CouplingInfo(NamedTuple):
    """Store the OT coupling."""

    coupling: np.ndarray
    early_rna_cost: np.ndarray
    late_rna_cost: np.ndarray
    early_cells: List[int]
    late_cells: List[int]

    def cost(self, cmat: np.ndarray, *, late: bool, scale: Optional[float] = None) -> float:
        """Compute the early or late error of a coupling."""
        if late:
            cmat = lot_eval.expand_coupling(cmat, self.coupling, self.late_rna_cost)
            cost = lot_inf.OT_cost(cmat, self.late_rna_cost)
            if scale is None:
                scale = self.ind_late_cost
        else:
            cmat = lot_eval.expand_coupling(cmat.T, self.coupling.T, self.early_rna_cost).T
            cost = lot_inf.OT_cost(cmat, self.early_rna_cost)
            if scale is None:
                scale = self.ind_early_cost

        return cost / scale

    @property
    def ind_coupling(self) -> np.ndarray:
        """Return the uninformative coupling."""
        return np.outer(self.early_marginal, self.late_marginal)

    @property
    def early_marginal(self) -> np.ndarray:
        """Return the early marginal."""
        return np.sum(self.coupling, axis=1)

    @property
    def late_marginal(self) -> np.ndarray:
        """Return the late marginal."""
        return np.sum(self.coupling, axis=0)

    @property
    def ind_early_cost(self) -> float:
        """Return the early error incurred by the uninformative coupling."""
        return self.cost(self.ind_coupling, late=False, scale=1.0)

    @property
    def ind_late_cost(self) -> float:
        """Return the late error incurred by the uninformative coupling."""
        return self.cost(self.ind_coupling, late=True, scale=1.0)


def remove_unlisted_leaves(tree: nx.DiGraph, kept_leaves: Sequence[Any], max_depth: int = 10) -> nx.DiGraph:
    """Filter leave nodes from tree."""
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
    """Create a lineage tree."""
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
        new_tree.nodes[cell]["time_to_parent"] = batch_time - new_tree.nodes[parent]["time"]
        new_tree.nodes[cell]["cell"] = lot_sim.Cell(adata.obsm[REP_KEY][cell_index, :], cell_label)

        assert new_tree.nodes[cell]["time_to_parent"] >= 0

    return TreeInfo(new_tree, good_cells, bad_cells)


def ground_truth_coupling(adata: AnnData, early_time: int, late_time: int) -> CouplingInfo:
    """Obtain a ground truth OT coupling."""

    def is_ancestor(late_cell: str, early_cell: str) -> bool:
        if late_cell[-1] not in "aplrdvx":
            warnings.warn("Ancestor checking not implemented for cell " + late_cell + " yet.")
            return False
        return early_cell in late_cell

    edata: TreeInfo = adata.uns["trees"][early_time]
    ldata: TreeInfo = adata.uns["trees"][late_time]

    n_early = len(edata.good_cells)
    n_late = len(ldata.good_cells)
    coupling = np.zeros([n_early, n_late], dtype=float)

    for i, c_early in enumerate(edata.good_cells):
        for j, c_late in enumerate(ldata.good_cells):
            if is_ancestor(ldata.tree.nodes[c_late]["name"], edata.tree.nodes[c_early]["name"]):
                coupling[i, j] = 1.0

    # filter out zero rows and columns
    kept_early_cells = np.where(np.sum(coupling, 1) > 0)[0]
    kept_late_cells = np.where(np.sum(coupling, 0) > 0)[0]
    coupling = coupling[np.ix_(kept_early_cells, kept_late_cells)]

    # normalize to uniform marginal on late cells
    coupling = np.dot(coupling, np.diag(np.sum(coupling, axis=0) ** -1)) / len(kept_late_cells)

    erep = np.asarray(adata[edata.good_cells].obsm[REP_KEY])
    lrep = np.asarray(adata[ldata.good_cells].obsm[REP_KEY])
    early_rna_cost = ot.utils.dist(erep, erep)[np.ix_(kept_early_cells, kept_early_cells)]
    late_rna_cost = ot.utils.dist(lrep, lrep)[np.ix_(kept_late_cells, kept_late_cells)]

    assert early_rna_cost.shape[0] == coupling.shape[0]
    assert late_rna_cost.shape[0] == coupling.shape[1]

    return CouplingInfo(coupling, early_rna_cost, late_rna_cost, kept_early_cells, kept_late_cells)


def compute_tree_distances(tree: nx.DiGraph) -> np.ndarray:
    """Compute distances along the lineage tree."""
    leaves = lot_inf.get_leaves(tree, include_root=False)
    num_leaves = len(leaves)
    distances = np.zeros([num_leaves, num_leaves])
    for i, leaf in enumerate(leaves):
        distance_dictionary, tmp = nx.multi_source_dijkstra(tree.to_undirected(), [leaf], weight="time")
        for j, target_leaf in enumerate(leaves):
            distances[i, j] = distance_dictionary[target_leaf]
    return distances


def _compute_w2(
    P_pred: np.array,
    P_gt: np.array,
    C: np.array,
    metric_type: Literal["descendant", "ancestor"],
    scale_by_marginal: bool = True,
) -> List:
    """Compute Wasserstein distances over arrays of distributions.

    Compute EMD distances between the rows or columns of two matrices, given a cost matrix `C`.
    """
    # initialize an empty list to store the errors per cell
    errors = []

    # transpose for ancestors
    if metric_type == "ancestor":
        P_pred = P_pred.T
        P_gt = P_gt.T

    # iterate over all cells
    for i in tqdm.tqdm(range(P_pred.shape[0])):
        # normalize to get distributions
        marginal = P_pred[i].sum()
        p = P_pred[i] / marginal
        g = P_gt[i] / P_gt[i].sum()

        # compute the EMD distance between ground-truth and predicted distibution
        error, log = ot.emd2(p, g, C, log=True)

        # append the EMD (potentially scaled by the marignal)
        if log["warning"] is None:
            if scale_by_marginal:
                errors.append(marginal * error)
            else:
                errors.append(error)
        else:
            errors.append(np.nan)

    return errors


def compute_errors(pred: np.ndarray, gt: CouplingInfo, scale_by_marginal: bool = True):
    """Compute ancestor and descendant errors."""
    # check that the shapes match
    assert pred.shape == gt.coupling.shape

    # compute the ancestor errors
    ancestor_errors = _compute_w2(
        P_pred=pred,
        P_gt=gt.coupling,
        C=gt.early_rna_cost,
        metric_type="ancestor",
        scale_by_marginal=scale_by_marginal,
    )

    # compute the descendant errors
    descendant_errors = _compute_w2(
        P_pred=pred,
        P_gt=gt.coupling,
        C=gt.late_rna_cost,
        metric_type="descendant",
        scale_by_marginal=scale_by_marginal,
    )

    return ancestor_errors, descendant_errors
