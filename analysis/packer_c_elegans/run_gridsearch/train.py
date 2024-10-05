import scanpy as sc
import sys
import networkx as nx
import wandb
import numpy as np

sys.path.insert(
    0, "../"
) 

try:
    import utils
except RuntimeError as e:
    if "jaxlib" not in str(e):
        raise

sys.path.insert(0, "../../../") 

from paths import DATA_DIR
ROOT = DATA_DIR / "packer_c_elegans"


def main():

    # register this run, get a set of parameters
    run = wandb.init()

    # extract parameters 
    config = wandb.config

    # prepare empty dict for results, and load data
    adata = sc.read(ROOT / "c_elegans.h5ad")
    full_reference_tree = nx.read_gml(ROOT / "ML_2023-11-06_packer_lineage_tree.gml")

    # pre-proces the data
    adata = utils.preprocess(adata, full_reference_tree, lineage_info=config.lineage_info)

    # run the corresponding method on this dataset
    early_time, late_time = config.tp
    if config.kind == "lot":
        tmat, conv, gt_coupling = utils.benchmark_lot(
            adata,
            early_time,
            late_time,
            epsilon=config.epsilon,
            scale_cost=config.scale_cost,
            tau_a=config.tau_a,
        )
    elif config.kind == "moslin":
        _, tmat, conv, gt_coupling = utils.benchmark_moscot(
            adata,
            early_time,
            late_time,
            alpha=config.alpha,
            epsilon=config.epsilon,
            scale_cost=config.scale_cost,
            tau_a=config.tau_a,
            max_inner_iterations=config.max_inner_iterations,
        )
    else:
        raise NotImplementedError(config.kind)

    # log metrics to wandb
    wandb.log(
        {
            "early_cost": gt_coupling.cost(tmat, late=False), 
            "late_cost": gt_coupling.cost(tmat, late=True),
            "mean_error": (gt_coupling.cost(tmat, late=False) + gt_coupling.cost(tmat, late=True)) / 2,
            "converged": conv,
            "deviation_from_balanced": np.abs(tmat.sum() - 1),
        }
    )


if __name__ == '__main__':
    main()