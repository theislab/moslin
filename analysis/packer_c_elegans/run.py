import pickle
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import scanpy as sc
import seml
from sacred import Experiment
from sacred.run import Run

try:
    import utils
except RuntimeError as e:
    if "jaxlib" not in str(e):
        raise


ROOT = Path("./data")


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run: Run) -> None:
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def benchmark(
    kind: Literal["lot", "moslin"],
    tp: Tuple[int, int],
    epsilon: float,
    alpha: float,
    lineage_info: Optional[Literal["precise", "abpxp"]],
    scale_cost: Literal["mean", "max_cost", 1.0],
    tau_a: float = 1.0,
) -> Dict[str, Any]:
    res = {}
    adata = sc.read(ROOT / "c_elegans.h5ad")
    with open(ROOT / "packer_lineage_tree.pkl", "rb") as file:
        full_reference_tree = pickle.load(file)
    adata = utils.preprocess(adata, full_reference_tree, lineage_info=lineage_info)

    early_time, late_time = tp
    if kind == "lot":
        tmat, conv, gt_coupling = utils.benchmark_lot(
            adata,
            early_time,
            late_time,
            epsilon=epsilon,
            scale_cost=scale_cost,
            tau_a=tau_a,
        )
    elif kind == "moslin":
        _, tmat, conv, gt_coupling = utils.benchmark_moscot(
            adata,
            early_time,
            late_time,
            alpha=alpha,
            epsilon=epsilon,
            scale_cost=scale_cost,
            tau_a=tau_a,
        )
    else:
        raise NotImplementedError(kind)

    res["early_cost"] = gt_coupling.cost(tmat, late=False)
    res["late_cost"] = gt_coupling.cost(tmat, late=True)
    res["converged"] = conv

    return res
