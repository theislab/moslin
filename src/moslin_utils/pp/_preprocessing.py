from typing import Literal, Optional

import networkx as nx
import numpy as np

import scanpy as sc
from anndata import AnnData

from moslin_utils.constants import TIME_KEY
from moslin_utils.ul import assign_time_to_bin, create_lineage_tree


def preprocess(
    adata: AnnData,
    ref_tree: nx.DiGraph,
    *,
    lineage_info: Optional[Literal["precise", "abpxp"]],
    seed: int = 0,
) -> AnnData:
    """Preprocess single-cell data."""
    # Removing partially lineage-labeled cells
    if lineage_info.lower() == "precise":
        print("Restricting to cells with complete lineage annotations")
        keep_mask = adata.obs["lineage"].to_numpy() == adata.obs["random_precise_lineage"].to_numpy()
        adata = adata[keep_mask].copy()
    elif lineage_info.lower() == "abpxp":
        print("Restricting to cells with lineage annotations starting with 'ABpxp'")
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
    # adata.obs[TIME_KEY].cat.categories = adata.obs[TIME_KEY].cat.categories.astype('float64')
    adata.obs = adata.obs.set_index("cell")

    rng = np.random.RandomState(seed)
    trees = {}
    for batch_time in adata.obs[TIME_KEY].cat.categories:
        print(batch_time)
        seed = rng.randint(0, 2**16 - 1)
        trees[batch_time] = create_lineage_tree(adata, batch_time=batch_time, reference_tree=ref_tree, seed=seed)

    adata.uns["trees"] = trees

    return adata
