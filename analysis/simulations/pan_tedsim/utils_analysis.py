from copy import deepcopy
from typing import Any, Dict, List, Literal, Mapping, Optional
import io

from Bio import Phylo
import lineageot.inference as lot_inf

from anndata import AnnData
import scanpy as sc

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx

node_colors = [
    '#bd6a47',
    '#bd6a47',
    '#779538',
    '#dbd4c0',
    '#b18393',
    '#8a6a4b',
    '#3e70ab',
    '#9d3d58',
    '#525566',
    '#3f5346',
    '#dbd4c0'
]


def _process_data(rna_arrays: Mapping[Literal["early", "late"], np.ndim], *, n_pcs: int = 30, pca=True) -> np.ndarray:
    adata = AnnData(rna_arrays["early"], dtype=np.float32).concatenate(
        AnnData(rna_arrays["late"], dtype=np.float32),
        batch_key="time",
        batch_categories=["0", "1"],
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    if pca:
        sc.tl.pca(adata, use_highly_variable=False)
        return adata.obsm["X_pca"][:, :n_pcs].astype(float, copy=True)
    return adata.X.astype(float, copy=True)


def _annotate(
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


def _cut_at_depth(G: nx.DiGraph, *, max_depth: Optional[int] = None) -> nx.DiGraph:
    if max_depth is None:
        return deepcopy(G)
    selected_nodes = [n for n in G.nodes if G.nodes[n]["node_depth"] <= max_depth]
    G = deepcopy(G.subgraph(selected_nodes).copy())

    # relabel because of LOT
    leaves = sorted(n for n in G.nodes if not list(G.successors(n)))
    for new_name, n in enumerate(leaves):
        G = nx.relabel_nodes(G, {n: new_name}, copy=False)
    return G


def _build_true_trees(
    rna: np.ndarray,
    barcodes: np.ndarray,
    meta: pd.DataFrame,
    *,
    tree: str,
    depth: int,
    n_pcs: int = 30,
    pca: bool = True,
    ttp: float = 100.0,
) -> Dict[Literal["early", "late"], nx.DiGraph]:
    cell_arr_adata = [lot_inf.sim.Cell(rna[nid], barcodes[nid]) for nid in range(rna.shape[0])]
    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = _newick2digraph(tree)
    G = _annotate(G, cell_arr_adata, metadata, ttp=ttp)

    trees = {"early": _cut_at_depth(G, max_depth=depth), "late": _cut_at_depth(G)}
    rna_arrays = {
        kind: np.asarray([trees[kind].nodes[n]["cell"].x for n in trees[kind].nodes if _is_leaf(trees[kind], n)])
        for kind in ["early", "late"]
    }
    data = _process_data(rna_arrays, n_pcs=n_pcs, pca=pca)

    n_early_leaves = len([n for n in trees["early"] if _is_leaf(trees["early"], n)])
    data_early, data_late = data[:n_early_leaves], data[n_early_leaves:]

    for kind, data in zip(["early", "late"], [data_early, data_late]):
        i, G = 0, trees[kind]
        for n in G.nodes:
            if _is_leaf(G, n):
                G.nodes[n]["cell"].x = data[i]
                i += 1
            else:
                G.nodes[n]["cell"].x = np.full((n_pcs,), np.nan)

    return trees

def _build_true_trees_draw(
    rna: np.ndarray,
    barcodes: np.ndarray,
    meta: pd.DataFrame,
    *,
    tree: str,
    depth: int,
    n_pcs: int = 30,
    pca: bool = True,
    ttp: float = 100.0,
) -> Dict[Literal["early", "late"], nx.DiGraph]:
    cell_arr_adata = [lot_inf.sim.Cell(rna[nid], barcodes[nid]) for nid in range(rna.shape[0])]
    metadata = [meta.iloc[nid].to_dict() for nid in range(rna.shape[0])]

    G = _newick2digraph(tree)
    G = _annotate(G, cell_arr_adata, metadata, ttp=ttp)

    trees = {"early": _cut_at_depth(G, max_depth=depth), "late": _cut_at_depth(G)}
    rna_arrays = {
        kind: np.asarray([trees[kind].nodes[n]["cell"].x for n in trees[kind].nodes if _is_leaf(trees[kind], n)])
        for kind in ["early", "late"]
    }
    data = _process_data(rna_arrays, n_pcs=n_pcs, pca=pca)

    n_early_leaves = len([n for n in trees["early"] if _is_leaf(trees["early"], n)])
    data_early, data_late = data[:n_early_leaves], data[n_early_leaves:]

    for kind, data in zip(["early", "late"], [data_early, data_late]):
        i, G = 0, trees[kind]
        for n in G.nodes:
            if _is_leaf(G, n):
                G.nodes[n]["cell"].x = data[i]
                i += 1
            else:
                G.nodes[n]["cell"].x = np.full((n_pcs,), np.nan)

    return trees


def _is_leaf(G: nx.DiGraph, n: Any) -> bool:
    return not list(nx.descendants(G, n))


def _newick2digraph(tree: str) -> nx.DiGraph:
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
        if _is_leaf(G, n):
            continue

        assert start not in G.nodes
        G = nx.relabel_nodes(G, {n: start}, copy=False)
        start += 1

    return G


def state_tree_draw(state_tree="((t1:2, t2:2):1, (t3:2, t4:2):1):2;", path=None):
    """Draw the state tree"""
    fig, axs = plt.subplots(figsize=(4, 4))
    state_tree_ = _newick2digraph(state_tree)
    pos = nx.drawing.nx_agraph.graphviz_layout(state_tree_, prog="dot")

    node_list = [3, 4, 1, 2, 5, 7, 6]
    labels = {node: node_list[node] for node in state_tree_.nodes}
    node_color = [node_colors[node_list[node]] for node in state_tree_.nodes]

    nx.draw(state_tree_, 
            pos, #labels=labels, 
            node_color=node_color, 
            node_size=400, 
            arrowsize=20, 
            arrows=True, 
            ax=axs, 
           )
    axs.set_title("cell state tree", fontsize=22)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path + "/cell_state_tree.png", bbox_inches="tight", transparent=True, dpi=300)
    plt.show()


def tree_draw(adata, depth=8, path=None):
    """Draw tree"""
    tree = adata.uns["tree"]
    rna = adata.X.copy()
    barcodes = adata.obsm["barcodes"].copy()

    true_trees = _build_true_trees(rna, barcodes, meta=adata.obs, tree=tree, depth=depth, pca=False)

    g_ = true_trees["early"]
    cols = [node_colors[int(g_.nodes[node]["cluster"])] for node in g_.nodes]
    pos = nx.drawing.nx_agraph.graphviz_layout(g_, prog="dot")

    fig, axs = plt.subplots(figsize=(8, 5))
    nx.draw(g_, pos, node_color=cols, node_size=50, arrowsize=10, arrows=True, ax=axs)
    axs.set_title(f"The early cell division tree (up to depth {depth})", fontsize=22)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path + "/tree.png", bbox_inches="tight", transparent=True, dpi=300)
    plt.show()


def plot_cost(lp, depth_early=8, depth_late=12):
    """Plot cost matrix"""
    for problem in lp.problems:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        sns.heatmap(lp.problems[problem].x.data, ax=axs[0], cbar=False, xticklabels=False, yticklabels=False)
        axs[0].set_title(f"Barcode distances\n(cells at depth {depth_early})", fontsize=16)
        axs[0].set_xlabel("cells", fontsize=14)
        axs[0].set_ylabel("cells", fontsize=14)

        sns.heatmap(lp.problems[problem].y.data, ax=axs[1], cbar=False, xticklabels=False, yticklabels=False)
        axs[1].set_title(f"Barcode distances\n(cells at depth {depth_late})", fontsize=16)
        axs[1].set_xlabel("cells", fontsize=14)
        axs[1].set_ylabel("cells", fontsize=14)

        plt.tight_layout()
        plt.show()
