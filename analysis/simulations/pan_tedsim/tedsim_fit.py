import argparse
from typing import Literal, Optional, List

import numpy as np
import pandas as pd
import lineageot.evaluation as lot_eval
import lineageot.inference as lot_inf


try:
    import utils_run
except RuntimeError as e:
    if "jaxlib" not in str(e):
        raise

ROOT = ""
OUTPUT = ""
MAX_DEPTH = 13
TTP = 100  # time to parent


def fit_cospar(adata, info, barcode_arrays, sample_times, result_dict, couplings, cluster_arrays, dists_ref, gt_ecost, gt_lcost):

    tmat = utils_run.run_cospar(
        adata=adata,
        info=info,
        barcode_arrays=barcode_arrays,
        sample_times=sample_times,
        couplings=couplings,
        seed=result_dict["data_seed"][0],
        data_path="all"
    )

    edist, ldist = dists_ref["early"], dists_ref["late"]
    gt_coupling = np.asarray(couplings["true"]).copy()

    c_early = tmat.T
    x_marginal_early = np.sum(gt_coupling.T, 1)
    c_early = (c_early / np.sum(c_early, 1)[:, np.newaxis]) * x_marginal_early[:, np.newaxis]

    c_late = tmat
    x_marginal_late = np.sum(gt_coupling, 1)
    c_late = (c_late / np.sum(c_late, 1)[:, np.newaxis]) * x_marginal_late[:, np.newaxis]

    pred_coupling = np.asarray(c_early)
    ecost = lot_inf.OT_cost(
        lot_eval.expand_coupling(pred_coupling, gt_coupling.T, edist).T,
        edist,
    )
    pred_coupling = np.asarray(c_late)
    lcost = lot_inf.OT_cost(
        lot_eval.expand_coupling(pred_coupling, gt_coupling, ldist).T,
        ldist,
    )

    result_dict.update({
        "kind": ["CoSpar"],
        "converged": [None],
        "early_cost": [ecost / gt_ecost],
        "late_cost": [lcost / gt_lcost],
        "tree_type": [info],
        "epsilon": [0.01],
    })

    kind = "CoSpar" if info == "barcodes-distance" else f"CoSpar ({info})"
    cluster_accuracy = utils_run.evaluate_cluster_accuracy(
        pred_coupling=pred_coupling,
        couplings=couplings,
        cluster_arrays=cluster_arrays,
        result_dict=result_dict,
        kind=kind
    )

    return pd.DataFrame(result_dict), pd.DataFrame(cluster_accuracy)


def fit_lot(true_trees, barcode_arrays, rna_arrays, sample_times, tree_type, result_dict, couplings, cluster_arrays, dists_ref, gt_ecost,
            gt_lcost):
    tree = true_trees["late"] if tree_type == "gt" else None

    tmat, converged = utils_run.run_lot(
        barcode_arrays=barcode_arrays,
        rna_arrays=rna_arrays,
        sample_times=sample_times,
        tree=tree,
        epsilon=result_dict["epsilon"][0],
        normalize_cost=True,
        ot_method="sinkhorn_stabilized",
    )

    early_cost, late_cost = utils_run.evaluate_coupling(
        pred_coupling=tmat, couplings=couplings, dists=dists_ref, gt_ecost=gt_ecost, gt_lcost=gt_lcost,
    )

    result_dict.update({
        "kind": ["LineageOT"],
        "converged": [converged],
        "early_cost": [early_cost],
        "late_cost": [late_cost],
        "tree_type": [tree_type],
    })

    cluster_accuracy = utils_run.evaluate_cluster_accuracy(
        pred_coupling=tmat,
        couplings=couplings,
        cluster_arrays=cluster_arrays,
        result_dict=result_dict,
        kind="LineageOT"
    )

    return pd.DataFrame(result_dict), pd.DataFrame(cluster_accuracy)


def fit_moslin(dists, alpha, tree_type, result_dict, couplings, cluster_arrays, dists_ref, gt_ecost, gt_lcost):
    edist, ldist, rna_dist = dists

    tmat, converged = utils_run.run_moslin(
        edist=edist,
        ldist=ldist,
        rna_dist=rna_dist,
        epsilon=result_dict["epsilon"][0],
        alpha=alpha,
        scale_cost=result_dict["scale_cost"][0]
    )

    early_cost, late_cost = utils_run.evaluate_coupling(
        pred_coupling=tmat, couplings=couplings, dists=dists_ref, gt_ecost=gt_ecost, gt_lcost=gt_lcost,
    )

    result_dict.update({
        "kind": ["moslin"],
        "converged": [converged],
        "early_cost": [early_cost],
        "late_cost": [late_cost],
        "alpha": [alpha],
        "tree_type": [tree_type],
    })

    cluster_accuracy = utils_run.evaluate_cluster_accuracy(
        pred_coupling=tmat,
        couplings=couplings,
        cluster_arrays=cluster_arrays,
        result_dict=result_dict,
        kind="moslin"
    )

    return pd.DataFrame(result_dict), pd.DataFrame(cluster_accuracy)


def benchmark(
        data_seed: Literal[
            23860, 50349, 36489, 59707, 38128, 25295, 49142, 12102, 30139, 4698
        ],
        scale_cost: Optional[Literal["mean", "max_cost"]] = "mean",
        depth: int = 10,
        seed: Optional[int] = None,
        ssr: Optional[float] = 0.0,
        alphas: Optional[List] = None,
        epsilons: Optional[List] = None,
        subsample: Optional[float] = None
):
    result = []
    cluster_accuracy = []
    cluster_accuracy_gt = []

    p_a = 0.4
    ss = 0.4
    ssr = ssr
    sample_times = {"early": depth * TTP, "late": MAX_DEPTH * TTP}

    epsilons_lot = [1e-1, 1.0]

    if seed is None:
        seed = int(abs(hash(f"pa{p_a}_ss{ss}") % (2 ** 32)))
    np.random.seed(seed)

    true_trees, rna_arrays, pca_arrays, barcode_arrays, cluster_arrays, remove_late_cells = utils_run.prepare_data(
        ROOT + f"adata_{p_a}_{ss}_{data_seed}.h5ad",
        depth=depth,
        ssr=ssr,
        pca=True,
        lognorm=True,
        n_pcs=30,
        ttp=TTP,
        subsample=subsample
    )

    couplings, dists, gt_ecost, gt_lcost = utils_run.ground_truth(true_trees, pca_arrays, remove_late_cells)
    gt_res = {"couplings": couplings, "dists_ref": dists, "gt_ecost": gt_ecost, "gt_lcost": gt_lcost}

    result_dict = {
        "p_a": [p_a],
        "ss": [ss],
        "ssr": [ssr],
        "data_seed": [data_seed],
        "alpha": [None],
        "epsilon": [None],
        "scale_cost": [scale_cost],
        "depth": [depth],
        "seed": [seed],
    }

    if alphas is not None:
        bc_dists = utils_run.compute_dists(
            tree_type="bc",
            trees=true_trees,
            rna_arrays=rna_arrays,
            barcode_arrays=barcode_arrays,
            dist_cache=None,
            remove_late_cells=remove_late_cells
        )

        if ssr == 0:
            gt_dists = utils_run.compute_dists(
                tree_type="gt",
                trees=true_trees,
                rna_arrays=rna_arrays,
                barcode_arrays=barcode_arrays,
                dist_cache=None,
                remove_late_cells=remove_late_cells
            )

        for alpha in alphas:
            for epsilon in epsilons:
                res_dict = result_dict.copy()
                res_dict["epsilon"] = [epsilon]

                res_mos, cluster_accuracy_mos = fit_moslin(
                    dists=bc_dists,
                    alpha=alpha,
                    tree_type="bc",
                    result_dict=res_dict,
                    cluster_arrays=cluster_arrays,
                    **gt_res
                )
                result.append(res_mos)
                cluster_accuracy.append(cluster_accuracy_mos)

                if ssr == 0.0:
                    res_mos_gt, cluster_accuracy_mos_gt = fit_moslin(
                        dists=gt_dists,
                        alpha=alpha,
                        tree_type="gt",
                        result_dict=result_dict.copy(),
                        cluster_arrays=cluster_arrays,
                        **gt_res
                    )
                    result.append(res_mos_gt)
                    cluster_accuracy_gt.append(cluster_accuracy_mos_gt)
    # barcodes
    adata_cospar = utils_run.set_adata_cospar(
        true_trees=true_trees,
        rna_arrays=rna_arrays,
        pca_arrays=pca_arrays,
        remove_late_cells=remove_late_cells
    )

    res_cospar_bcs, cluster_accuracy_cospar_bcs = fit_cospar(
        adata=adata_cospar,
        info="barcodes-distance",
        barcode_arrays=barcode_arrays,
        sample_times=sample_times,
        result_dict=result_dict.copy(),
        cluster_arrays=cluster_arrays,
        **gt_res
    )
    result.append(res_cospar_bcs)
    cluster_accuracy.append(cluster_accuracy_cospar_bcs)

    if ssr < 0.3:
        # lot recon
        res_cospar, cluster_accuracy_cospar = fit_cospar(
            adata=adata_cospar,
            info="fitted-tree",
            barcode_arrays=barcode_arrays,
            sample_times=sample_times,
            result_dict=result_dict.copy(),
            cluster_arrays=cluster_arrays,
            **gt_res
        )
        result.append(res_cospar)
        cluster_accuracy.append(cluster_accuracy_cospar)

        for epsilon in epsilons_lot:
            res_dict = result_dict.copy()
            res_dict["epsilon"] = [epsilon]
            res_lot, cluster_accuracy_lot = fit_lot(
                true_trees=true_trees,
                barcode_arrays=barcode_arrays,
                rna_arrays=rna_arrays,
                sample_times=sample_times,
                tree_type="bc",
                result_dict=res_dict,
                cluster_arrays=cluster_arrays,
                **gt_res
            )
            result.append(res_lot)
            cluster_accuracy.append(cluster_accuracy_lot)

    pd.concat(result).to_csv(OUTPUT + f"{data_seed}_{ssr}_{scale_cost}_{depth}_{subsample}.csv")
    pd.concat(cluster_accuracy).to_csv(OUTPUT + f"cluster_accuracy_{data_seed}_{ssr}_{scale_cost}_{depth}_{subsample}.csv")

    if ssr == 0.0:
        # gt
        adata_cospar = utils_run.set_adata_cospar(
            true_trees=true_trees,
            rna_arrays=rna_arrays,
            pca_arrays=pca_arrays
        )

        res_cospar_state, cluster_accuracy_cospar_state = fit_cospar(
            adata=adata_cospar,
            info="state-only",
            barcode_arrays=barcode_arrays,
            sample_times=sample_times,
            result_dict=result_dict.copy(),
            cluster_arrays=cluster_arrays,
            **gt_res
        )
        result.append(res_cospar_state)
        cluster_accuracy_gt.append(cluster_accuracy_cospar_state)

        res_cospar_gt, cluster_accuracy_cospar_gt = fit_cospar(
            adata=adata_cospar,
            info="ground-truth",
            barcode_arrays=barcode_arrays,
            sample_times=sample_times,
            result_dict=result_dict.copy(),
            cluster_arrays=cluster_arrays,
            **gt_res
        )
        result.append(res_cospar_gt)
        cluster_accuracy_gt.append(cluster_accuracy_cospar_gt)

        for epsilon in epsilons_lot:
            res_dict = result_dict.copy()
            res_dict["epsilon"] = [epsilon]
            res_lot_gt, cluster_accuracy_lot_gt = fit_lot(
                true_trees=true_trees,
                barcode_arrays=barcode_arrays,
                rna_arrays=rna_arrays,
                sample_times=sample_times,
                tree_type="gt",
                result_dict=res_dict,
                cluster_arrays=cluster_arrays,
                 **gt_res
            )
            result.append(res_lot_gt)
            cluster_accuracy_gt.append(cluster_accuracy_lot_gt)

        pd.concat(result).to_csv(OUTPUT + f"{data_seed}_{ssr}_{scale_cost}_{depth}_{subsample}_gt.csv")
        pd.concat(cluster_accuracy_gt).to_csv(OUTPUT + f"cluster_accuracy_{data_seed}_{ssr}_{scale_cost}_{depth}_{subsample}_gt.csv")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=False, default=23860)
    parser.add_argument("--ssr", type=float, required=False, default=0.0)
    parser.add_argument("--depth", type=int, required=False, default=8)
    parser.add_argument("--subsample", type=float, required=False, default=0.0)
    args = parser.parse_args()

    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1]
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
    scale_cost = "mean"

    benchmark(
        depth=args.depth,
        data_seed=args.seed,
        ssr=args.ssr,
        alphas=alphas,
        epsilons=epsilons,
        scale_cost=scale_cost,
        subsample=args.subsample
    )
