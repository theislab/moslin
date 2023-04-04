import copy
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ot
from typing import Literal, Dict, List, Optional

import anndata
import networkx as nx

import lineageot.simulation as sim
import lineageot.evaluation as sim_eval
import lineageot.inference as sim_inf

import os
import sys
from jax.config import config

config.update("jax_enable_x64", True)

from ott.geometry import pointcloud, geometry
from ott.solvers.quadratic import gromov_wasserstein
from ott.problems.quadratic import quadratic_problem

sys.path.insert(0, "/cs/labs/mornitzan/zoe.piran/research/projects/moscot/src")
sys.path.insert(0, "../../../") 

from moscot.problems.time import LineageProblem, TemporalProblem
from paths import DATA_DIR, FIG_DIR

DATA_DIR = DATA_DIR / "simulations/forrow_2gene"
FIG_DIR = FIG_DIR / "simulations/forrow_2gene"

import mplscience

mplscience.set_style()
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["figure.dpi"] = 300

method_colors = {
    "moslin": "#E1BE6A", 
    "GW": "#117733",
    "OT": "#88CCEE",
    "LineageOT": "#40B0A6",
} 

def run_seeds(
        flow_type: Literal["bifurcation", "convergent", "partial_convergent", "mismatched_clusters"],
        seeds: Optional[List] = None,
        epsilons: Optional[List] = None,
        alphas: Optional[List] = None,
        cost_keys: Optional[List] = None,
        save: bool = True,
        return_res: bool = True
):
    sim_dicts, adatas = simulate_seeds(flow_type=flow_type, seeds=seeds, save=save)

    return fit_seeds(flow_type=flow_type,
                     sim_dicts=sim_dicts,
                     adatas=adatas,
                     epsilons=epsilons,
                     alphas=alphas,
                     cost_keys=cost_keys,
                     return_res=return_res)


def simulate_seeds(
        flow_type: Literal["bifurcation", "convergent", "partial_convergent", "mismatched_clusters"],
        seeds: Optional[List] = None,
        save: bool = True,
):
    if seeds is None:
        seeds = [
            4698.,
            # 12102.,
            # 23860.,
            # 25295.,
            # 30139.,
            # 36489.,
            # 38128.,
            # 48022.,
            # 49142.,
            # 59706.
        ]
    sim_dicts = {}
    for seed in seeds:
        sim_dicts[seed] = simulate_data(flow_type, seed=int(seed), save=save)

    adatas = {}
    for key, sim_dict in sim_dicts.items():
        adatas[key] = anndata_from_sim(sim_dict, flow_type=flow_type)

    return sim_dicts, adatas


def fit_seeds(
    flow_type: Literal["bifurcation", "convergent", "partial_convergent", "mismatched_clusters"],
    sim_dicts: Dict,
    adatas: Dict,
    epsilons: Optional[List] = None,
    alphas: Optional[List] = None,
    cost_keys: Optional[List] = None,
    return_res: bool = True,
    verbose: bool = True,
    
):
    if epsilons is None:
        epsilons = np.logspace(-4, 1, 15)
    if alphas is None:
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.999]
    if cost_keys is None:
        cost_keys = ["true", "fitted"]

    # create couplings
    if verbose:
        print(f"evaluating {flow_type} over {len(epsilons)} epsilons and {len(alphas)} alphas.")
        
    lot_couplings = {}
    for key, sim_dict in sim_dicts.items():
        lot_couplings[key] = {}
        lot_couplings[key]["true"], lot_couplings[key]["fitted"] = fit_lineageOT(epsilons=epsilons, **sim_dict)
    moslin_couplings = {}
    for seed, sim_dict in sim_dicts.items():
        moslin_couplings[seed] = {}
        for key in cost_keys:
            moslin_couplings[seed][key] = fit_moslin(
                adatas[seed],
                epsilons,
                alphas,
                joint_full=True,  # if key=="true" else True,
                cost_matrices_key=key,
            )
    ot_couplings = {}
    for seed, sim_dict in sim_dicts.items():
        ot_couplings[seed] = fit_moslin_ot(
            adatas[seed],
            epsilons,
            joint_full=True
        )
    gw_couplings = {}
    for seed, sim_dict in sim_dicts.items():
        gw_couplings[seed] = {}
        for key in cost_keys:
            gw_couplings[seed][key] = fit_moslin_gw(
                adatas[seed],
                epsilons,
                alphas,
                cost_matrices_key=key,
            )

    # evaluate couplings
    # independent couplings
    for seed, sim_dict in sim_dicts.items():
        sim_dicts[seed] = independent_coupling(sim_dict)

    # lot
    ancestor_errors_lot = {}
    for seed, couplings_seed in lot_couplings.items():
        ancestor_errors_lot[seed] = {}
        for key, couplings in couplings_seed.items():
            ancestor_errors_lot[seed][key] = get_metrics(
                couplings,
                lambda x: sim_inf.OT_cost(x, sim_dicts[seed]["rna_cost"]["early"]),
                scale=sim_dicts[seed]["independent_ancestor_error"],
                return_dict=False
            )

    descendant_errors_lot = {}
    for seed, couplings_seed in lot_couplings.items():
        descendant_errors_lot[seed] = {}
        for key, couplings in couplings_seed.items():
            descendant_errors_lot[seed][key] = get_metrics(
                couplings,
                lambda x: sim_inf.OT_cost(sim_eval.expand_coupling(x, sim_dicts[seed]["couplings"]["true"],
                                                                   sim_dicts[seed]["rna_cost"]["late"]),
                                          sim_dicts[seed]["rna_cost"]["late"]),
                scale=sim_dicts[seed]["independent_descendant_error"],
                return_dict=False
            )

    # moslin-ot
    ancestor_errors_ot = {}

    for seed, couplings_seed in ot_couplings.items():
        ancestor_errors_ot[seed] = get_metrics(
            couplings_seed,
            lambda x: sim_inf.OT_cost(x, sim_dicts[seed]["rna_cost"]["early"]),
            scale=sim_dicts[seed]["independent_ancestor_error"],
            return_dict=False
        )
    descendant_errors_ot = {}

    for seed, couplings_seed in ot_couplings.items():
        descendant_errors_ot[seed] = get_metrics(
            couplings_seed,
            lambda x: sim_inf.OT_cost(sim_eval.expand_coupling(x, sim_dicts[seed]["couplings"]["true"],
                                                               sim_dicts[seed]["rna_cost"]["late"]),
                                      sim_dicts[seed]["rna_cost"]["late"]),
            scale=sim_dicts[seed]["independent_descendant_error"],
            return_dict=False
        )

    # moslin-GW
    ancestor_errors_gw = {}
    for seed, couplings_seed in gw_couplings.items():
        ancestor_errors_gw[seed] = {}
        for key, couplings in couplings_seed.items():
            ancestor_errors_gw[seed][key] = get_metrics(
                couplings,
                lambda x: sim_inf.OT_cost(x, sim_dicts[seed]["rna_cost"]["early"]),
                scale=sim_dicts[seed]["independent_ancestor_error"],
            )

    descendant_errors_gw = {}
    for seed, couplings_seed in gw_couplings.items():
        descendant_errors_gw[seed] = {}
        for key, couplings in couplings_seed.items():
            descendant_errors_gw[seed][key] = get_metrics(
                couplings,
                lambda x: sim_inf.OT_cost(sim_eval.expand_coupling(x, sim_dicts[seed]["couplings"]["true"],
                                                                   sim_dicts[seed]["rna_cost"]["late"]),
                                          sim_dicts[seed]["rna_cost"]["late"]),
                scale=sim_dicts[seed]["independent_descendant_error"],
                return_dict=False
            )

    # moslin
    ancestor_errors_moslin = {}

    for seed, couplings_seed in moslin_couplings.items():
        ancestor_errors_moslin[seed] = {}
        for key, moslin_couplings_type in couplings_seed.items():
            ancestor_errors_moslin[seed][key] = {}
            for a, coupling in moslin_couplings_type.items():
                ancestor_errors_moslin[seed][key][a] = get_metrics(
                    coupling,
                    lambda x: sim_inf.OT_cost(x, sim_dicts[seed]["rna_cost"]["early"]),
                    scale=sim_dicts[seed]["independent_ancestor_error"],
                    return_dict=False
                )
    descendant_errors_moslin = {}

    for seed, couplings_seed in moslin_couplings.items():
        descendant_errors_moslin[seed] = {}
        for key, moslin_couplings_type in couplings_seed.items():
            descendant_errors_moslin[seed][key] = {}
            for a, coupling in moslin_couplings_type.items():
                descendant_errors_moslin[seed][key][a] = get_metrics(
                    coupling,
                    lambda x: sim_inf.OT_cost(
                        sim_eval.expand_coupling(x, sim_dicts[seed]["couplings"]["true"],
                                                 sim_dicts[seed]["rna_cost"]["late"]),
                        sim_dicts[seed]["rna_cost"]["late"]),
                    scale=sim_dicts[seed]["independent_descendant_error"],
                    return_dict=False
                )

    # optimal \alpha
    alpha_min = {}
    for seed in ancestor_errors_moslin.keys():
        alpha_min[seed] = {}
        for key in cost_keys:
            cost_min = 1
            for a, costs in ancestor_errors_moslin[seed][key].items():
                if np.min(costs[1]) < cost_min:
                    cost_min = np.min(costs[1])
                    alpha_min[seed][key] = a

    df = pd.DataFrame(columns=[
        "method",
        "seed",
        "flow_type",
        "tree_type",
        "epsilon",
        "ancestor_error",
        "descendant_error",
        "mean_error"
    ])

    ancestor_errors = {}
    descendant_errors = {}
    for seed in ancestor_errors_lot.keys():
        ancestor_errors[seed] = {}
        descendant_errors[seed] = {}
        for key in cost_keys:
            ancestor_errors[seed][key] = {
                "LineageOT": ancestor_errors_lot[seed][key],
                "OT": ancestor_errors_ot[seed],
                "GW": ancestor_errors_gw[seed][key],
                "moslin": ancestor_errors_moslin[seed][key][alpha_min[seed][key]]
            }

            descendant_errors[seed][key] = {
                "LineageOT": descendant_errors_lot[seed][key],
                "OT": descendant_errors_ot[seed],
                "GW": descendant_errors_gw[seed][key],
                "moslin": descendant_errors_moslin[seed][key][alpha_min[seed][key]]
            }
    methods = ["LineageOT", "OT", "GW", "moslin"]

    for seed in ancestor_errors.keys():
        for method in methods:
            for key in cost_keys:
                mean_err = [sum(x) / 2 for x in zip(ancestor_errors[seed][key][method][1],
                                                    descendant_errors[seed][key][method][1])]
                idx_min = np.argmin(mean_err)
                dict_ = pd.DataFrame({
                    "method": method,
                    "seed": seed,
                    "flow_type": flow_type,
                    "tree_type": key,
                    "epsilon": ancestor_errors[seed][key][method][0][idx_min],
                    "ancestor_error": ancestor_errors[seed][key][method][1][idx_min],
                    "descendant_error": descendant_errors[seed][key][method][1][idx_min],
                    "mean_error": mean_err[idx_min]
                }, index=[1])
                df = pd.concat((df, dict_), ignore_index=True)

    df.to_csv(DATA_DIR / f"{flow_type}_res_seeds.csv")

    with open(DATA_DIR / f"{flow_type}_ancestor_errors_moslin.pkl", "wb") as fout:
        pickle.dump(ancestor_errors_moslin, fout)
    with open(DATA_DIR / f"{flow_type}_descendant_errors_moslin.pkl", "wb") as fout:
        pickle.dump(descendant_errors_moslin, fout)
    if verbose:
        print("    Done!")
    if return_res:
        return df, ancestor_errors_moslin, descendant_errors_moslin
    return


def simulate_data(
        flow_type: Literal["bifurcation", "convergent", "partial_convergent", "mismatched_clusters"],
        seed: int = 257,
        save: bool = True,
):
    # Check if simulation exists
    
    fpath = DATA_DIR / f"{flow_type}_{seed}_sim.pickle"
    if os.path.isfile(fpath):
        with open(fpath, "rb") as fin:
            res = pickle.load(fin)
        return res
    
    # Set simulation parameters
    np.random.seed(seed)

    if flow_type == "bifurcation":
        timescale = 1
    else:
        timescale = 100

    x0_speed = 1 / timescale

    sim_params = sim.SimulationParameters(division_time_std=0.01 * timescale,
                                          flow_type=flow_type,
                                          x0_speed=x0_speed,
                                          mutation_rate=1 / timescale,
                                          mean_division_time=1.1 * timescale,
                                          timestep=0.001 * timescale
                                          )

    # These parameters can be adjusted freely.
    # As is, they replicate the plots in the paper for the fully convergent simulation.
    mean_x0_early = 2
    time_early = 7.4 * timescale  # Time when early cells are sampled
    time_late = time_early + 4 * timescale  # Time when late cells are sampled
    x0_initial = mean_x0_early - time_early * x0_speed
    initial_cell = sim.Cell(np.array([x0_initial, 0, 0]), np.zeros(sim_params.barcode_length))
    sample_times = {"early": time_early, "late": time_late}

    ## Running the simulation
    sample = sim.sample_descendants(initial_cell.deepcopy(), time_late, sim_params)

    # Extracting trees and barcode matrices
    true_trees = {"late": sim_inf.list_tree_to_digraph(sample)}
    true_trees["late"].nodes["root"]["cell"] = initial_cell

    true_trees["early"] = sim_inf.truncate_tree(true_trees["late"], sample_times["early"], sim_params)

    # Computing the ground-truth coupling
    couplings = {"true": sim_inf.get_true_coupling(true_trees["early"], true_trees["late"])}

    data_arrays = {}
    rna_arrays = {}
    barcode_arrays = {}
    num_cells = {}
    for tree_kind in ["early", "late"]:
        data_arrays[tree_kind] = sim_inf.extract_data_arrays(true_trees[tree_kind])
        rna_arrays[tree_kind] = data_arrays[tree_kind][0]
        barcode_arrays[tree_kind] = data_arrays[tree_kind][1]
        num_cells[tree_kind] = rna_arrays[tree_kind].shape[0]

    # print("Times    : ", sample_times)
    # print("Number of cells: ", num_cells)

    # Creating a copy of the true tree for use in LineageOT
    true_trees["late, annotated"] = copy.deepcopy(true_trees["late"])
    sim_inf.add_node_times_from_division_times(true_trees["late, annotated"])

    sim_inf.add_nodes_at_time(true_trees["late, annotated"], sample_times["early"]);

    # Infer ancestor locations for the late cells based on the true lineage tree

    observed_nodes = [n for n in sim_inf.get_leaves(true_trees["late, annotated"], include_root=False)]
    sim_inf.add_conditional_means_and_variances(true_trees["late, annotated"], observed_nodes)

    ancestor_info = {
        "true tree": sim_inf.get_ancestor_data(true_trees["late, annotated"], sample_times["early"])}

    # True distances
    true_distances = {key: sim_inf.compute_tree_distances(true_trees[key]) for key in true_trees}

    # Fit lineaeg tree with LineageOT
    # Estimate mutation rate from fraction of unmutated barcodes

    rate_estimate = sim_inf.rate_estimator(barcode_arrays["late"], sample_times["late"])

    # (1) Compute Hamming distance matrices for neighbor joining
    # (2) Compute neighbor-joining tree
    # (3) Annotate fitted tree with internal node times
    hamming_distances_with_roots = {}
    hamming_distances = {}
    fitted_trees = {}
    for tree_kind in ["early", "late"]:
        hamming_distances_with_roots[tree_kind] = sim_inf.barcode_distances(
            np.concatenate([barcode_arrays[tree_kind], np.zeros([1, sim_params.barcode_length])]))
        hamming_distances[tree_kind] = sim_inf.barcode_distances(barcode_arrays[tree_kind])
        fitted_trees[tree_kind] = sim_inf.neighbor_join(hamming_distances_with_roots[tree_kind])

        sim_inf.add_leaf_barcodes(fitted_trees[tree_kind], barcode_arrays[tree_kind])
        sim_inf.add_leaf_x(fitted_trees[tree_kind], rna_arrays[tree_kind])
        sim_inf.add_leaf_times(fitted_trees[tree_kind], sample_times[tree_kind])
        sim_inf.annotate_tree(
            fitted_trees[tree_kind],
            rate_estimate * np.ones(sim_params.barcode_length),
            time_inference_method="least_squares")

    # True distances
    fitted_distances = {key: sim_inf.compute_tree_distances(fitted_trees[key]) for key in fitted_trees}

    sim_inf.add_node_times_from_division_times(fitted_trees["late"])
    sim_inf.add_nodes_at_time(fitted_trees["late"], sample_times["early"])

    observed_nodes = [n for n in sim_inf.get_leaves(fitted_trees["late"], include_root=False)]
    sim_inf.add_conditional_means_and_variances(fitted_trees["late"], observed_nodes)
    ancestor_info["fitted tree"] = sim_inf.get_ancestor_data(fitted_trees["late"], sample_times["early"])

    # Compute cost matrices for each method

    rna_cost = {}
    rna_cost["early"] = ot.utils.dist(rna_arrays["early"],
                                      sim_inf.extract_ancestor_data_arrays(true_trees["late"],
                                                                           sample_times["early"], sim_params)[
                                          0])
    rna_cost["late"] = ot.utils.dist(rna_arrays["late"], rna_arrays["late"])

    res = {
        "rna_arrays": rna_arrays,
        "ancestor_info": ancestor_info,
        "sample_times": sample_times,
        "sim_params": sim_params,
        "true_trees": true_trees,
        "fitted_trees": fitted_trees,
        "true_distances": true_distances,
        "fitted_distances": fitted_distances,
        "hamming_distances": hamming_distances,
        "couplings": couplings,
        "rna_cost": rna_cost
    }

    if save:
        with open(fpath, "wb") as fout:
            pickle.dump(res, fout)

    return res


def fit_lineageOT(epsilons, rna_arrays, ancestor_info, sample_times, sim_params, true_trees, **kwargs):
    # Compute cost matrices for each setting
    coupling_costs = {}
    coupling_costs["lineageOT, true tree"] = ot.utils.dist(rna_arrays["early"],
                                                           ancestor_info["true tree"][0]) @ np.diag(
        ancestor_info["true tree"][1] ** (-1))
    coupling_costs["lineageOT, fitted tree"] = ot.utils.dist(rna_arrays["early"],
                                                             ancestor_info["fitted tree"][0]) @ np.diag(
        ancestor_info["fitted tree"][1] ** (-1))

    early_time_rna_cost = ot.utils.dist(rna_arrays["early"],
                                        sim_inf.extract_ancestor_data_arrays(true_trees["late"],
                                                                             sample_times["early"],
                                                                             sim_params)[0])
    late_time_rna_cost = ot.utils.dist(rna_arrays["late"], rna_arrays["late"])

    couplings_lot = {}
    couplings_true = {}
    couplings_fitted = {}
    couplings_lot["lineageOT"] = ot.emd([], [], coupling_costs["lineageOT, true tree"])
    couplings_lot["lineageOT, fitted"] = ot.emd([], [], coupling_costs["lineageOT, fitted tree"])
    for e in epsilons:
        if e >= 0.1:
            f = ot.sinkhorn
        else:
            # Epsilon scaling is more robust at smaller epsilon, but slower than simple sinkhorn
            f = ot.bregman.sinkhorn_epsilon_scaling
        print("Working on couplings for epsilon = " + str(e) + " .")
        couplings_true["true " + str(e)] = f([], [], coupling_costs["lineageOT, true tree"],
                                             e * np.mean(ancestor_info["true tree"][1] ** (-1)))
        couplings_fitted["fitted " + str(e)] = f([], [], coupling_costs["lineageOT, fitted tree"],
                                                 e * np.mean(ancestor_info["fitted tree"][1] ** (-1)))

    return couplings_true, couplings_fitted


def fit_moslin(
        adata: anndata.AnnData,
        epsilons: List[float],
        alphas: List[float],
        cost_matrices_key: Literal["true", "hamming", "fitted"],
        joint_full=True,
):
    # Compute cost matrices for `cost_matrices_key`
    couplings_alphas = {}
    joint_attr = {"attr": "X"} if joint_full else {"attr": "obsm", "key": "X_dims"}
    for a in alphas:
        couplings_alphas[a] = {}
        for e in epsilons:
            lp = LineageProblem(adata=adata).prepare(
                time_key="time",
                joint_attr=joint_attr,
                lineage_attr={"attr": "obsp", "key": f"cost_matrices_{cost_matrices_key}"},
                policy="sequential",
                scale_cost="max_cost",
            )
            lp = lp.solve(alpha=a, epsilon=e)

            if lp.solutions[(0, 1)].converged:
                couplings_alphas[a][f"{cost_matrices_key} " + str(e)] = np.array(
                    lp.solutions[(0, 1)].transport_matrix)

    return couplings_alphas


def fit_moslin_ot(
        adata: anndata.AnnData,
        epsilons: List[float],
        joint_full=True,
):
    # Compute cost matrices for each setting
    couplings = {}
    joint_attr = {"attr": "X"} if joint_full else {"attr": "obsm", "key": "X_dims"}
    for e in epsilons:
        tp = TemporalProblem(adata=adata).prepare(
            time_key="time",
            joint_attr=joint_attr,
            scale_cost="max_cost",
        )
        tp = tp.solve(epsilon=e)
        if tp.solutions[(0, 1)].converged:
            couplings[str(e)] = np.array(tp.solutions[(0, 1)].transport_matrix)

    return couplings


def fit_moslin_gw(
        adata: anndata.AnnData,
        epsilons: List[float],
        alphas: List[float],
        cost_matrices_key: Literal["true", "hamming", "fitted"]
):
    # Compute cost matrices for each setting
    couplings = {}

    for e in epsilons:
        geom_xx = geometry.Geometry(
            cost_matrix=adata[adata.obs["time"].isin([0])].obsp[f"cost_matrices_{cost_matrices_key}"],
            scale_cost="max_cost",
            epsilon=e,
        )
        geom_yy = geometry.Geometry(
            cost_matrix=adata[adata.obs["time"].isin([1])].obsp[f"cost_matrices_{cost_matrices_key}"],
            scale_cost="max_cost",
            epsilon=e,
        )
        prob = quadratic_problem.QuadraticProblem(
            geom_xx,
            geom_yy,
        )
        solver = gromov_wasserstein.GromovWasserstein(epsilon=e)
        lp = solver(prob)

        if lp.converged:
            couplings[f"{cost_matrices_key} " + str(e)] = np.array(lp.matrix)

    return couplings


def anndata_from_sim(sim_dict: Dict,
                     flow_type: Literal[
                         "bifurcation", "convergent", "partial_convergent", "mismatched_clusters"]):
    """
    Create an anndata from simulation ouput, 
    """

    X = np.concatenate((sim_dict["rna_arrays"]["early"], sim_dict["rna_arrays"]["late"]))
    cost_matrices_true = np.zeros((X.shape[0], X.shape[0]))
    cost_matrices_hamming = np.zeros((X.shape[0], X.shape[0]))
    cost_matrices_fitted = np.zeros((X.shape[0], X.shape[0]))

    cost_matrices_true[:sim_dict["true_distances"]["early"].shape[0],
    :sim_dict["true_distances"]["early"].shape[0]] = sim_dict["true_distances"]["early"]
    cost_matrices_true[sim_dict["true_distances"]["early"].shape[0]:,
    sim_dict["true_distances"]["early"].shape[0]:] = sim_dict["true_distances"]["late"]

    cost_matrices_hamming[:sim_dict["hamming_distances"]["early"].shape[0],
    :sim_dict["hamming_distances"]["early"].shape[0]] = sim_dict["hamming_distances"]["early"]
    cost_matrices_hamming[sim_dict["hamming_distances"]["early"].shape[0]:,
    sim_dict["hamming_distances"]["early"].shape[0]:] = sim_dict["hamming_distances"]["late"]

    cost_matrices_fitted[:sim_dict["fitted_distances"]["early"].shape[0],
    :sim_dict["fitted_distances"]["early"].shape[0]] = sim_dict["fitted_distances"]["early"]
    cost_matrices_fitted[sim_dict["fitted_distances"]["early"].shape[0]:,
    sim_dict["fitted_distances"]["early"].shape[0]:] = sim_dict["fitted_distances"]["late"]

    adata = anndata.AnnData(X=X, dtype=X.dtype)
    adata.obsp["cost_matrices_true"] = cost_matrices_true
    adata.obsp["cost_matrices_hamming"] = cost_matrices_hamming
    adata.obsp["cost_matrices_fitted"] = cost_matrices_fitted

    adata.obs["time"] = 1
    obs_early = [str(x) for x in range(sim_dict["rna_arrays"]["early"].shape[0])]
    adata.obs.loc[obs_early, "time"] = 0

    adata.uns["true_trees"] = {}
    adata.uns["fitted_trees"] = {}
    for tree in ["true_trees", "fitted_trees"]:
        for i, kind in enumerate(["early", "late"]):
            H = nx.relabel_nodes(sim_dict[tree][kind], lambda x: str(x))
            adata.uns[tree][i] = H

    # Choosing which of the three dimensions to use
    if flow_type == "mismatched_clusters":
        dimensions_to_use = [1, 2]
    else:
        dimensions_to_use = [0, 1]

    adata.obsm["X_dims"] = X[:, dimensions_to_use]

    return adata


def independent_coupling(sim_dict):
    sim_dict["couplings"]["independent"] = np.ones(sim_dict["couplings"]["true"].shape) / \
                                           sim_dict["couplings"]["true"].size

    sim_dict["independent_ancestor_error"] = sim_inf.OT_cost(
        sim_dict["couplings"]["independent"],
        sim_dict["rna_cost"]["early"]
    )

    sim_dict["independent_descendant_error"] = sim_inf.OT_cost(
        sim_eval.expand_coupling(sim_dict["couplings"]["independent"],
                                 sim_dict["couplings"]["true"],
                                 sim_dict["rna_cost"]["late"]),
        sim_dict["rna_cost"]["late"]
    )

    return sim_dict


def get_metrics(
        couplings,
        cost_func,
        scale=1.0,
        method_key=None,
        return_dict=True,
):
    """
    Returnd cost_func values
    """

    eps = []
    ys = []

    for key, val in couplings.items():
        eps.append(float(key.split(" ")[-1]))
        ys.append(cost_func(val) / scale)

    return [eps, ys]


def plot_metrics(
        ys_dict,
        cost_func_name="coupling error",
        label_font_size=18,
        tick_font_size=12,
        pal_dict=None
):
    """
    Plots cost_func evaluated as a function of epsilon
    """
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    pal = sns.diverging_palette(220, 20, n=len(ys_dict))
    # pal = ["#e1be6a","#e4c478","#e7cb87","#ead196","#edd8a5","#f0deb4","#f3e5c3","#f6ebd2","#f9f2e1","#fcf8f0","#ffffff"]
    pal_dict = pal_dict if pal_dict is not None else {val: pal[i] for i, val in enumerate(ys_dict)}
    for label, ys in ys_dict.items():
        c = pal_dict[label]
        sns.lineplot(x=ys[0], y=ys[1], label=r"$\alpha=$" + f"{label}", color=c, ax=axs)

    plt.ylabel(cost_func_name, fontsize=label_font_size)
    plt.xlabel(r"$\epsilon$" + " (entropy parameter)", fontsize=label_font_size)
    plt.xscale("log")

    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    plt.xlim([ys[0][0], ys[0][-1]])

    ylims = plt.ylim([0, None])
    # upper limit should be at least 1
    plt.ylim([0, max(ylims[1], 1)])
    ncols = 2 if len(ys_dict) > 6 else 1
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=tick_font_size,
               ncols=ncols)
    plt.tight_layout()
    plt.show()
    return


def plot_trajectory(
    sim_dict,
    flow_type,
    thr=1e-8,
    alpha_scale=1,
    ax=None,
    savefig=False,
    subsample=False,
    legend_off=True,
    title=True,
    **kwargs
):
    """  
    Plot matrix M in 2D with  lines using alpha values
    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples.
    Notes
    -----
    This function is inspired from LineageOT implementation.
    """

    # Choosing which of the three dimensions to plot
    if flow_type == "mismatched_clusters":
        dimensions_to_use = [1, 2]
    else:
        dimensions_to_use = [0, 1]

    xs = sim_dict["rna_arrays"]["early"][:, dimensions_to_use]
    xt = sim_dict["rna_arrays"]["late"][:, dimensions_to_use]
    G = sim_dict["couplings"]["true"]

    if subsample:
        sidx = np.random.choice(xs.shape[0], int(xs.shape[0] / 2))
        tidx = np.random.choice(xt.shape[0], int(xt.shape[0] / 2))
        xs = xs[sidx]
        xt = xt[tidx]
        G = G[sidx, :][:, tidx]
    colors = {
        "early": "#005AB5",
        "late": "#DC3220",
        "line": "#808080"
    }
    size = 4
    if ("color" not in kwargs) and ("c" not in kwargs):
        kwargs["color"] = colors["line"]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    mx = G.max()
    sources = []
    targets = []
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                sns.lineplot(x=[xs[i, 0], xt[j, 0]], y=[xs[i, 1], xt[j, 1]],
                             alpha=alpha_scale * G[i, j] / mx, zorder=0, **kwargs)
                sources.append(i)
                targets.append(j)

    for xy, label in zip([xs, xt], ['early', 'late']):
        sns.scatterplot(x=xy[:, 0], y=xy[:, 1], alpha=0.4, label=f"{label} cells", color=colors[label])

    plt.axis("off")
    if legend_off:
        plt.legend().remove()
    if title:
        plt.title(flow_type)
    plt.tight_layout()
    if savefig:
        plt.savefig(FIG_DIR / f"{flow_type}_trajectory.png", bbox_inches="tight", transparent=True, dpi=300)
    plt.show()

    return sources, targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_type',
                        type=str,
                        required=False,
                        default="bifurcation")
    args = parser.parse_args()

    run_seeds(args.flow_type)