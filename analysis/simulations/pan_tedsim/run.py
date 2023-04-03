from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import seml
from sacred import Experiment
from sacred.run import Run

try:
    import utils_run
except RuntimeError as e:
    if "jaxlib" not in str(e):
        raise

ROOT = Path("./data_generation/adatas")
MAX_DEPTH = 13
TTP = 100  # time to parent

logger = getLogger()


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
    p_a: Literal[0.2, 0.4, 0.6, 0.8, 1],
    ss: Literal[0.2, 0.4, 0.6, 0.8, 1],
    data_seed: Literal[
        23860, 50349, 36489, 59707, 38128, 25295, 49142, 12102, 30139, 4698
    ],
    alpha: float,
    epsilon: float,
    ssr: Optional[float],
    tree_type: Literal["gt", "bc", "cas_dists", "cas"],
    kind: Literal["moscot", "lot"] = "moscot",
    scale_cost: Optional[Literal["mean", "max_cost"]] = "max_cost",
    depth: int = 8,
    rank: int = -1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    result = {
        "p_a": p_a,
        "ss": ss,
        "data_seed": data_seed,
        "alpha": alpha,
        "epsilon": epsilon,
        "ssr": ssr,
        "tree_type": tree_type,
        "kind": kind,
        "scale_cost": scale_cost,
        "depth": depth,
        "rank": rank,
        "seed": seed,
        "converged": None,
        "early_cost": np.nan,
        "late_cost": np.nan,
    }
    if seed is None:
        seed = int(abs(hash(f"pa{p_a}_ss{ss}") % (2**32)))
    np.random.seed(seed)

    true_trees, rna_arrays, barcode_arrays = utils_run.prepare_data(
        ROOT / f"adata_{p_a}_{ss}_{data_seed}.h5ad",
        depth=depth,
        ssr=ssr,
        n_pcs=30,
        ttp=TTP,
    )

    if kind == "moscot":
        dist_cache = Path(
            f"costs/pa={p_a}_ss={ss}_ds={data_seed}_tt={tree_type}_ssr={ssr}_depth={depth}.pkl"
        )
        edist, ldist, rna_dist = utils_run.compute_dists(
            tree_type,
            trees=true_trees,
            rna_arrays=rna_arrays,
            barcode_arrays=barcode_arrays,
            dist_cache=None,
        )
        tmat, converged = utils_run.run_moscot(
            edist=edist,
            ldist=ldist,
            rna_dist=rna_dist,
            alpha=alpha,
            epsilon=epsilon,
            rank=rank,
            scale_cost=scale_cost,
        )
    elif kind == "lot":
        assert tree_type in ("gt", "bc"), tree_type
        sample_times = {"early": depth * TTP, "late": MAX_DEPTH * TTP}
        tree = true_trees["late"] if tree_type == "gt" else None

        tmat, converged = utils_run.run_lot(
            barcode_arrays=barcode_arrays,
            rna_arrays=rna_arrays,
            sample_times=sample_times,
            tree=tree,
            epsilon=epsilon,
            normalize_cost=True,
            ot_method="sinkhorn_stabilized",
        )
    else:
        raise NotImplementedError(f"Benchmarking {kind} is not implemented.")

    early_cost, late_cost = utils_run.evaluate_coupling(
        tmat, true_trees=true_trees, rna_arrays=rna_arrays
    )
    result["converged"] = converged
    result["early_cost"] = early_cost
    result["late_cost"] = late_cost

    return result
