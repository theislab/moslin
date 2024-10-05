import jax

jax.config.update("jax_enable_x64", True)

import io
import pickle
import warnings
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import lineageot.core as lot_core
import lineageot.evaluation as lot_eval
import lineageot.inference as lot_inf

from ott.geometry import geometry
import cospar as cs

from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from ott.solvers.linear import sinkhorn

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from Bio import Phylo
from jax import numpy as jnp
from scipy.sparse import issparse
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cluster

CASETTE_SIZE = 4
N_CASETTES = 8

logger = getLogger()


# taken from Cassiopeia
def get_cassettes() -> List[int]:
    cassettes = [(CASETTE_SIZE * j) for j in range(0, N_CASETTES)]
    return cassettes


def silence_cassettes(
        character_array: np.ndarray, silencing_rate: float, missing_state: int = -1
) -> np.ndarray:
    updated_character_array = character_array.copy()
    cassettes = get_cassettes()
    cut_site_by_cassette = np.digitize(range(len(character_array)), cassettes)

    for cassette in range(1, N_CASETTES + 1):
        if np.random.uniform() < silencing_rate:
            indices = np.where(cut_site_by_cassette == cassette)
            left, right = np.min(indices), np.max(indices)
            for site in range(left, right + 1):
                updated_character_array[site] = missing_state

    return updated_character_array


def stochastic_silencing(
        barcodes: np.ndarray,
        stochastic_silencing_rate: float = 1e-2,
        stochastic_missing_data_state: int = -1,
) -> np.ndarray:
    assert 0 <= stochastic_silencing_rate <= 1.0, stochastic_silencing_rate
    barcodes_ss = np.zeros(barcodes.shape)
    for i, bc in enumerate(barcodes):
        barcodes_ss[i, :] = silence_cassettes(
            bc, stochastic_silencing_rate, stochastic_missing_data_state
        )
    return barcodes_ss


def run_cospar(adata, info, barcode_arrays, sample_times, couplings, seed, data_path = None):
    if data_path:
        cs.settings.data_path = f"data_{seed}_{data_path}"
    else:
        cs.settings.data_path = f"data_{seed}"

    if info == "state-only":
        adata_orig = cs.pp.initialize_adata_object(adata)
        adata_res = cs.tmap.infer_Tmap_from_state_info_alone(
            adata_orig,
            compute_new=True,
            max_iter_N=[20, 20],
            epsilon_converge=[0.01, 0.01],
            smooth_array=[20, 15, 10, 5],
            sparsity_threshold=0.1,
            trunca_threshold=[0.001, 0.001]
        )

    else:
        clone_coupling = couplings["true"].copy()

        if info == "fitted-tree":
            ladata = AnnData(
                adata[adata.obs["time_info"] == 1].X.copy(), obsm={"barcodes": barcode_arrays["late"]}, dtype=np.float64
            )
            fitted_tree = lot_core.fit_tree(ladata, sample_times["late"])
            lot_inf.add_node_times_from_division_times(fitted_tree)
            lot_inf.add_nodes_at_time(fitted_tree, sample_times["early"])
            clone_coupling = fitted_couplings(fitted_tree)

        elif info == "barcodes-distance":
            ldist = lot_inf.barcode_distances(barcode_arrays["late"])
            ldist[np.isnan(ldist)] = np.nanmax(ldist)
            clustering = cluster.AgglomerativeClustering(n_clusters=clone_coupling.shape[0])
            clustering.fit(ldist)
            clone_coupling = np.zeros(clone_coupling.shape)
            clone_coupling[clustering.labels_, np.arange(clone_coupling.shape[1])] = 1

        num_early = len(adata[adata.obs["time_info"] == 0].obs_names)
        num_clones = clone_coupling.shape[0]

        adata.obsm["X_clone"] = np.array(np.zeros((adata.n_obs, num_clones)), dtype="bool")

        clone_coupling[clone_coupling > 0] = 1
        adata.obsm["X_clone"][num_early:, :] = np.array(clone_coupling.T, dtype="bool")

        adata_orig = cs.pp.initialize_adata_object(adata)
        adata_res = cs.tmap.infer_Tmap_from_one_time_clones(
            adata_orig,
            initial_time_points=[0],
            later_time_point=[1],
            compute_new=True,
            max_iter_N=[20, 20],
            epsilon_converge=[0.01, 0.01],
            initialize_method="OT",
            OT_cost="GED",
            smooth_array=[20, 15, 10, 5],
            sparsity_threshold=0.1,
            trunca_threshold=[0.001, 0.001]
        )

    return adata_res.uns["transition_map"].A


def run_moslin(
        edist: np.ndarray,
        ldist: np.ndarray,
        rna_dist: np.ndarray,
        epsilon: float,
        alpha: float,
    scale_cost:str
):
    # Compute cost matrices for `cost_matrices_key`

    if alpha == 0:
        geom_xy = geometry.Geometry(
            cost_matrix=rna_dist,
            scale_cost=scale_cost,
            epsilon=epsilon,
        )
        ot_prob = linear_problem.LinearProblem(geom_xy)
        solver = sinkhorn.Sinkhorn()
        # Solve OT problem
        lp = solver(ot_prob)
    elif alpha == 1:
        geom_xx = geometry.Geometry(
            cost_matrix=edist,
            scale_cost=scale_cost,
            epsilon=epsilon,
        )
        geom_yy = geometry.Geometry(
            cost_matrix=ldist,
            scale_cost=scale_cost,
            epsilon=epsilon,
        )
        prob = quadratic_problem.QuadraticProblem(
            geom_xx,
            geom_yy,
        )
        solver = gromov_wasserstein.GromovWasserstein(epsilon=epsilon, max_iterations=1000)
        lp = solver(prob)
    else:
        fused_penalty = (1 - alpha) / alpha

        geom_xy = geometry.Geometry(
            cost_matrix=rna_dist,
            scale_cost=scale_cost,
            epsilon=epsilon,
        )

        geom_xx = geometry.Geometry(
            cost_matrix=edist,
            scale_cost=scale_cost,
            epsilon=epsilon,
        )
        geom_yy = geometry.Geometry(
            cost_matrix=ldist,
            scale_cost=scale_cost,
            epsilon=epsilon,
        )
        prob = quadratic_problem.QuadraticProblem(
            geom_xx=geom_xx,
            geom_yy=geom_yy,
            geom_xy=geom_xy,
            fused_penalty=fused_penalty
        )
        solver = gromov_wasserstein.GromovWasserstein(epsilon=epsilon, max_iterations=1000)
        lp = solver(prob)

    return np.array(lp.matrix), lp.converged


def run_lot(
        barcode_arrays: Mapping[Literal["early", "late"], np.ndarray],
        rna_arrays: Mapping[Literal["early", "late"], np.ndarray],
        sample_times: Mapping[Literal["early", "late"], float],
        *,
        tree: Optional[nx.DiGraph] = None,
        epsilon: float = 0.05,
        normalize_cost: bool = True,
        **kwargs: Any,
) -> Tuple[np.ndarray, bool]:
    """Fits a LineageOT coupling between the cells at time_1 and time_2.

    In the process, annotates the lineage tree with observed and estimated cell states.

    Parameters
    ----------
    tree:
        The lineage tree fitted to cells at time_2. Nodes should already be annotated with times.
        Annotations related to cell state will be added.
    sample_times: Dict
        sampling times of late and early cells
    barcode_arrays: Dict
        barcode arrays of late and early cells
    rna_arrays: Dict
        expression space arrays of late and early cells
    epsilon : float (default 0.05)
        Entropic regularization parameter for optimal transport
    normalize_cost : bool (default True)
        Whether to rescale the cost matrix by its median before fitting a coupling.
        Normalizing this way allows us to choose a reasonable default epsilon for data of any scale

    Returns
    -------
    The coupling and whether marginals are satisfied.
    """
    time_key = "time"

    eadata = AnnData(
        rna_arrays["early"],
        obsm={"barcodes": barcode_arrays["early"]},
        dtype=np.float64,
    )
    ladata = AnnData(
        rna_arrays["late"], obsm={"barcodes": barcode_arrays["late"]}, dtype=np.float64
    )

    adata = eadata.concatenate(ladata, batch_key=time_key)
    adata.obs[time_key] = adata.obs[time_key].cat.rename_categories(
        {"0": sample_times["early"], "1": sample_times["late"]}
    )

    if tree is None:
        tree = lot_core.fit_tree(ladata, sample_times["late"])
    else:
        lot_inf.add_leaf_x(tree, ladata.X)

    with warnings.catch_warnings(record=True) as ws:
        coupling = lot_core.fit_lineage_coupling(
            adata,
            sample_times["early"],
            sample_times["late"],
            lineage_tree_t2=tree,
            epsilon=epsilon,
            normalize_cost=normalize_cost,
            **kwargs,
        )
    coupling = coupling.X.astype(np.float64)
    conv = not [w for w in ws if "did not converge" in str(w.message)]

    return coupling, conv and np.all(np.isfinite(coupling))


def process_data(
        rna_arrays: Mapping[Literal["early", "late"], np.ndim],
        *,
        n_pcs: int = 30,
        lognorm: bool = True,
        pca: bool = True,
) -> np.ndarray:
    adata_early = AnnData(rna_arrays["early"], dtype=float)
    adata_late = AnnData(rna_arrays["late"], dtype=float)
    adata = sc.concat([adata_early, adata_late], label="time", keys=["0", "1"])
    if lognorm:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if pca:
        sc.pp.highly_variable_genes(adata)
        sc.tl.pca(adata, use_highly_variable=False)

        return adata.obsm["X_pca"][:, :n_pcs].astype(float, copy=True)
    else:
        return adata.X.astype(float, copy=True)


def compute_dists(
        tree_type: Literal["gt", "bc"],
        trees: Mapping[Literal["early", "late"], nx.DiGraph],
        rna_arrays: Mapping[Literal["early", "late"], np.ndarray],
        barcode_arrays: Mapping[Literal["early", "late"], np.ndarray],
        dist_cache: Optional[Path] = None,
        scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distances for time_1, time_2 and joint.

    Parameters
    ----------
    trees:
        The lineage tree fitted to cells at time_1 and time_2. Nodes should already be annotated with times.
        Annotations related to cell state will be added.
    rna_arrays: RNA arrays of later and early cells.
    barcode_arrays: barcode arrays of late and early cells.
    tree_type: the type of distance to evaluate.
    dist_cache: Path to a cache file where to read/write the distance matrices.
    scale: Whether to scale the cost by maximum.

    Returns
    -------
    Early and late distances.
    """

    def maybe_scale(arr: np.ndarray) -> np.ndarray:
        if np.any(~np.isfinite(edist)):
            raise ValueError("Non-finite values found in distances.")
        return arr / np.max(arr) if scale else arr

    if dist_cache is not None and dist_cache.is_file():
        logger.info(f"Loading distances from `{dist_cache}`")
        with open(dist_cache, "rb") as fin:
            return pickle.load(fin)

    if tree_type == "gt":
        edist = lot_inf.compute_tree_distances(trees["early"])
        ldist = lot_inf.compute_tree_distances(trees["late"])
    elif tree_type == "bc":
        edist = lot_inf.barcode_distances(barcode_arrays["early"])
        ldist = lot_inf.barcode_distances(barcode_arrays["late"])
        edist[np.isnan(edist)] = np.nanmax(edist)
        ldist[np.isnan(ldist)] = np.nanmax(ldist)
    else:
        raise NotImplementedError(f"Tree type `{tree_type}` not yet implemented.")

    rna_dist = euclidean_distances(rna_arrays["early"], rna_arrays["late"])
    edist, ldist, rna_dist = (
        maybe_scale(edist),
        maybe_scale(ldist),
        maybe_scale(rna_dist),
    )

    if dist_cache is not None:
        logger.info(f"Saving distances to `{dist_cache}`")
        with open(dist_cache, "wb") as fout:
            pickle.dump((edist, ldist, rna_dist), fout)

    return edist, ldist, rna_dist


def is_leaf(G: nx.DiGraph, n: Any) -> bool:
    return not list(nx.descendants(G, n))


def newick2digraph(tree: str) -> nx.DiGraph:
    def trav(clade, prev: Any, depth: int) -> None:
        nonlocal cnt
        if depth == 0:
            name = "root"
        else:
            name = clade.name
            if name is None:
                name = cnt
                cnt -= 1
            else:
                name = int(name[1:]) - 1

        G.add_node(name, node_depth=depth)
        if prev is not None:
            G.add_edge(prev, name)

        for c in clade.clades:
            trav(c, name, depth + 1)

    G = nx.DiGraph()
    cnt = -1
    tree = Phylo.read(io.StringIO(tree), "newick")
    trav(tree.clade, None, 0)

    start = max([n for n in G.nodes if n != "root"]) + 1
    for n in list(nx.dfs_preorder_nodes(G)):
        if n == "root":
            pass
        if is_leaf(G, n):
            continue

        assert start not in G.nodes
        G = nx.relabel_nodes(G, {n: start}, copy=False)
        start += 1

    return G


def annotate(
        G: nx.DiGraph,
        cell_arr_data: List[lot_inf.sim.Cell],
        meta: List[Dict[str, Any]],
        ttp: int = 100,
) -> nx.DiGraph:
    G = G.copy()
    n_leaves = len([n for n in G.nodes if not len(list(G.successors(n)))])
    assert (n_leaves & (n_leaves - 1)) == 0, f"{n_leaves} is not power of 2"
    max_depth = int(np.log2(n_leaves))

    n_expected_nodes = 2 ** (max_depth + 1) - 1
    assert len(G) == n_expected_nodes, "graph is not a full binary tree"

    if len(cell_arr_data) != n_expected_nodes:  # missing root, add after observed nodes
        dummy_cell = lot_inf.sim.Cell([], [])
        cell_arr_data += [dummy_cell]
        meta += [{}]

    for nid in G.nodes:
        depth = G.nodes[nid]["node_depth"]
        metadata = {
            **meta[nid],  # contains `depth`, which is different from `node_depth`
            "cell": cell_arr_data[nid],
            "nid": nid,
            "time": depth * ttp,
            "time_to_parent": ttp,
        }
        G.nodes[nid].update(metadata)

    for eid in G.edges:
        G.edges[eid].update({"time": ttp})

    return nx.relabel_nodes(G, {n_leaves: "root"}, copy=False)


def cut_at_depth(G: nx.DiGraph, *, max_depth: Optional[int] = None) -> nx.DiGraph:
    if max_depth is None:
        return deepcopy(G)
    selected_nodes = [n for n in G.nodes if G.nodes[n]["node_depth"] <= max_depth]
    G = deepcopy(G.subgraph(selected_nodes).copy())

    # relabel because of LOT
    leaves = sorted(n for n in G.nodes if not list(G.successors(n)))
    for new_name, n in enumerate(leaves):
        G = nx.relabel_nodes(G, {n: new_name}, copy=False)
    return G


def is_valid_edge(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
    r"""Assumes the following state tree:

       /-4
      7
     / \-3
    5
     \ /-1
      6
       \-2
    """
    state_tree = nx.from_edgelist([(5, 6), (5, 7), (6, 1), (6, 2), (7, 3), (7, 4)])
    try:
        # parent, cluster, depth
        p1, c1, d1 = n1["parent"], n1["cluster"], n1["depth"]
        p2, c2, d2 = n2["parent"], n2["cluster"], n2["depth"]
    except KeyError:
        # no metadata, assume true
        return True

    # root, anything is permitted
    if (p1, c1, d1) == (5, 6, 0):
        return True

    # sanity checks
    assert p1 in [5, 6, 7], p1
    assert p2 in [5, 6, 7], p2
    assert c1 in [1, 2, 3, 4, 6, 7], c1
    assert c2 in [1, 2, 3, 4, 6, 7], c2

    if p1 == p2:
        if c1 == c2:
            # check if depth of a parent is <=
            return d1 <= d2
        # sanity check that clusters are valid siblings
        return (c1, c2) in state_tree.edges

    # parent-cluster relationship
    assert c1 == p2, (c1, p2)
    # valid transition
    assert (c1, c2) in state_tree.edges, (c1, c2)
    return True


def build_true_trees(
        rna: np.ndarray,
        barcodes: np.ndarray,
        meta: pd.DataFrame,
        *,
        tree: str,
        depth: int,
        max_depth: Optional[int] = None,
        n_pcs: int = 30,
        pca: bool = True,
        lognorm: bool = True,
        ttp: float = 100.0):
    cell_arr_adata = [
        lot_inf.sim.Cell(rna[nid], barcodes[nid]) for nid in range(rna.shape[0])
    ]
    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = newick2digraph(tree)
    G = annotate(G, cell_arr_adata, metadata, ttp=ttp)
    for s, t in G.edges:
        sn, tn = G.nodes[s], G.nodes[t]
        assert is_valid_edge(sn, tn), (s, t)

    trees = {
        "early": cut_at_depth(G, max_depth=depth),
        "late": cut_at_depth(G, max_depth=max_depth),
    }
    rna_arrays = {
        kind: np.asarray(
            [
                trees[kind].nodes[n]["cell"].x
                for n in trees[kind].nodes
                if is_leaf(trees[kind], n)
            ]
        )
        for kind in ["early", "late"]
    }

    data = process_data(rna_arrays, n_pcs=n_pcs, pca=pca, lognorm=lognorm)

    n_early_leaves = len([n for n in trees["early"] if is_leaf(trees["early"], n)])
    data_early, data_late = data[:n_early_leaves], data[n_early_leaves:]

    for kind, data in zip(["early", "late"], [data_early, data_late]):
        i, G = 0, trees[kind]
        for n in G.nodes:
            if is_leaf(G, n):
                G.nodes[n]["cell"].x = data[i]
                i += 1
            else:
                G.nodes[n]["cell"].x = np.full((n_pcs,), np.nan)

    return trees, rna_arrays


def prepare_data(
        fpath: Path,
        *,
        depth: int,
        max_depth: Optional[int] = None,
        ssr: Optional[float] = None,
        n_pcs: int = 30,
        pca: bool = True,
        lognorm: bool = True,
        ttp: float = 100.0,
) -> Tuple[
    Dict[Literal["early", "late"], nx.DiGraph],
    Dict[Literal["early", "late"], np.ndarray],
    Dict[Literal["early", "late"], np.ndarray],
    Dict[Literal["early", "late"], np.ndarray],
]:
    adata = sc.read(fpath)
    tree = adata.uns["tree"]
    rna = adata.X.A.copy() if issparse(adata.X) else adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    if ssr is not None:
        barcodes = stochastic_silencing(barcodes, stochastic_silencing_rate=ssr)
    true_trees, rna_arrays = build_true_trees(
        rna,
        barcodes,
        meta=adata.obs,
        tree=tree,
        depth=depth,
        max_depth=max_depth,
        n_pcs=n_pcs,
        pca=pca,
        lognorm=lognorm,
        ttp=ttp
    )
    data_arrays = {
        "late": lot_inf.extract_data_arrays(true_trees["late"]),
        "early": lot_inf.extract_data_arrays(true_trees["early"]),
    }

    processed_rna_arrays = {
        "early": data_arrays["early"][0],
        "late": data_arrays["late"][0],
    }
    barcode_arrays = {
        "early": data_arrays["early"][1],
        "late": data_arrays["late"][1],
    }

    return true_trees, rna_arrays, processed_rna_arrays, barcode_arrays


def evaluate_coupling(
        pred_coupling: Union[np.ndarray, jnp.ndarray],
        couplings: Dict,
        dists: Dict,
        gt_ecost: float,
        gt_lcost: float
) -> Tuple[float, float]:
    edist, ldist = dists["early"], dists["late"]

    pred_coupling = np.asarray(pred_coupling)
    gt_coupling = np.asarray(couplings["true"])

    try:
        ecost = lot_inf.OT_cost(
            lot_eval.expand_coupling(pred_coupling.T, gt_coupling.T, edist).T,
            edist,
        )
    except Exception as e:
        logger.error(f"Unable to compute the early cost, reason: `{e}`")
        ecost = np.inf

    try:
        lcost = lot_inf.OT_cost(
            lot_eval.expand_coupling(pred_coupling, gt_coupling, ldist).T,
            ldist,
        )
    except Exception as e:
        logger.error(f"Unable to compute the late cost, reason: `{e}`")
        lcost = np.inf

    return ecost / gt_ecost, lcost / gt_lcost


def ground_truth(
        true_trees: Mapping[Literal["early", "late"], nx.DiGraph],
        rna_arrays: Mapping[Literal["early", "late"], np.ndarray],
) -> Tuple[
    Dict[Literal["true", "independent"], np.ndarray],
    Dict[Literal["early", "late"], np.ndarray],
    float,
    float,
]:
    couplings = {
        "true": lot_inf.get_true_coupling(true_trees["early"], true_trees["late"])
    }

    a = jnp.asarray(couplings["true"].sum(1))
    b = jnp.asarray(couplings["true"].sum(0))
    couplings["independent"] = np.outer(a, b)

    edist = euclidean_distances(rna_arrays["early"], rna_arrays["early"])
    ldist = euclidean_distances(rna_arrays["late"], rna_arrays["late"])

    ecost = lot_inf.OT_cost(
        lot_eval.expand_coupling(
            np.asarray(couplings["independent"].T),
            np.asarray(couplings["true"].T),
            edist,
        ).T,
        edist,
    )
    lcost = lot_inf.OT_cost(
        lot_eval.expand_coupling(
            np.asarray(couplings["independent"]), np.asarray(couplings["true"]), ldist
        ).T,
        ldist,
    )
    dists = {"early": edist, "late": ldist}

    return couplings, dists, ecost, lcost


def fitted_couplings(
        fitted_tree
):
    cells_early = [n for n in fitted_tree.nodes if type(n) == tuple]
    cells_late = lot_inf.get_leaves(fitted_tree, include_root=False)

    fitted_coupling = np.zeros([len(cells_early), len(cells_late)])

    for i, cell in enumerate(cells_early):
        for child in nx.descendants(fitted_tree, cell):
            if child in cells_late:
                fitted_coupling[i, child] = 1

    return fitted_coupling


def set_adata_cospar(true_trees, rna_arrays, pca_arrays):
    obs_names_early = [f"early_{leaf}" for leaf in lot_inf.get_leaves(true_trees["early"], include_root=False)]
    obs_names_late = [f"late_{leaf}" for leaf in lot_inf.get_leaves(true_trees["late"], include_root=False)]
    obs_names = obs_names_early + obs_names_late

    X = np.concatenate(list(rna_arrays.values()))
    adata = AnnData(X=X, dtype=X.dtype)
    adata.obs_names = obs_names

    adata.obs["time_info"] = 1
    adata.obs["time_info"][:rna_arrays["early"].shape[0]] = 0
    adata.obs["time"] = adata.obs["time_info"].copy().astype("category")

    adata.obsm["X_pca"] = np.concatenate(list(pca_arrays.values()))

    return adata

