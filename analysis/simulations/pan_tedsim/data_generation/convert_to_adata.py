import argparse
import pathlib

import newick
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

P_AS = (0.2, 0.4, 0.6, 0.8, 1)
STEP_SIZES = (0.2, 0.4, 0.6, 0.8, 1)
SEEDS = (23860, 50349, 36489, 59707, 38128, 25295, 49142, 12102, 30139, 4698)


def create_anndata(*, p_a: float, step_size: float, seed: int) -> AnnData:
    adata = AnnData(
        X=pd.read_csv(
            f"./{root}/counts_tedsim_{p_a}_{step_size}_{seed}.csv"
        ).T.values.astype(float),
        obs=pd.read_csv(
            f"./{root}/cell_meta_tedsim_{p_a}_{step_size}_{seed}.csv", index_col=0
        ),
        obsm={
            "barcodes": pd.read_csv(
                f"./{root}/character_matrix_{p_a}_{step_size}_{seed}.txt", sep=" "
            ).values
        },
        dtype=float,
    )
    bcs = adata.obsm["barcodes"]
    if not np.issubdtype(bcs.dtype, int):
        bcs[bcs == "-"] = -1
        adata.obsm["barcodes"] = bcs.astype(int)

    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    adata.obs["depth"] = adata.obs["depth"].astype(int)
    adata.obs["parent"] = adata.obs["parent"].astype("category")
    with open(root / f"tree_gt_bin_tedsim_{p_a}_{step_size}_{seed}.newick", "r") as fin:
        tree = newick.load(fin)[0].newick

    p_a, step_size = float(p_a), float(step_size)
    adata.uns["tree"] = tree

    adata.uns["metadata"] = {"p_a": p_a, "step_size": step_size, "seed": seed}
    assert (adata.obsm["barcodes"] == -1).sum() == 0, "Some barcodes have been deleted."

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Path where the TedSim generated data is stored.")
    parser.add_argument(
        "--out",
        default="./adatas",
        help="Output path where to save the AnnData objects.",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True, parents=True)

    for p_a in P_AS:
        for ss in [0.4]:
            for seed in SEEDS:
                adata = create_anndata(p_a=p_a, step_size=ss, seed=seed)
                sc.write(out / f"adata_{p_a}_{ss}_{seed}.h5ad", adata)

    for p_a in [0.4]:
        for ss in STEP_SIZES:
            for seed in SEEDS:
                adata = create_anndata(p_a=p_a, step_size=ss, seed=seed)
                sc.write(out / f"adata_{p_a}_{ss}_{seed}.h5ad", adata)
