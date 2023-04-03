import os
import sys
import numpy as np
import pickle as pkl
import argparse
import networkx as nx

import scanpy as sc
import pandas as pd

import ot

from jax.config import config
config.update("jax_enable_x64", True)


sys.path.insert(0, "/cs/labs/mornitzan/zoe.piran/research/projects/moscot/src")
sys.path.insert(0, "/cs/labs/mornitzan/zoe.piran/research/projects/moslin_reproducibility")

from moscot.problems.time import LineageProblem, TemporalProblem

from paths import DATA_DIR, CACHE_DIR, FIG_DIR


DATA_DIR = DATA_DIR / "hu_zebrafish_linnaeus"


def beta_lineage_cost(adata, Ci, dpi, beta):
    cells = (adata.obs['ident'].isin([dpi]))
    cells_idx = np.where(adata.obs['ident'].isin([dpi]))[0]
    C = np.zeros((len(cells_idx), len(cells_idx)))
    for i, ci in enumerate(adata[cells].obs_names):
        C[i, :] = Ci[dpi][adata.obs.loc[[ci], 'core'].values, 
                               adata.obs.loc[adata[cells].obs_names,'core'].values]
    C = C / C.max()
    M = ot.utils.dist(adata[cells].obsm['latent'], adata[cells].obsm['latent'])
    M = M / M.max()
    
    return pd.DataFrame(beta*C + (1-beta)*M, 
                        index=adata.obs_names[cells],                            
                        columns=adata.obs_names[cells])


def estimate_marginals(adata, dpi_source, dpi_target, time_diff):
    source_growth = adata[(adata.obs["ident"].isin([dpi_source]))].obs["cell_growth_rate"].to_numpy()
    target_growth = adata[(adata.obs["ident"].isin([dpi_target]))].obs["cell_growth_rate"].to_numpy()
    
    growth = np.power(source_growth, time_diff)
    a = growth 
    a = a / a.sum()
    
    
    b = np.full(target_growth.shape[0], np.average(growth))
    b = b / b.sum()
    
    adata.obs["a_marginal"] = 0
    adata.obs["b_marginal"] = 0
    
    adata.obs.loc[adata.obs["ident"].isin([dpi_source]), "a_marginal"]  = a
    adata.obs.loc[adata.obs["ident"].isin([dpi_target]), "b_marginal"]  = b

    return


def fit_couplings_all(args):
    adata = sc.read(DATA_DIR / 'adata.h5ad')
    
    with open(DATA_DIR / 'trees.pkl', 'rb') as file:
        trees = pkl.load(file)

    trees_dpis = {}
    trees_dpis_core = {}
    core_nodes = 100
    for i, dpi in enumerate(adata.obs["ident"].cat.categories):
        trees_dpis[dpi] = trees[dpi]
        trees_dpis_core[dpi] = trees_dpis[dpi].subgraph([str(n) for n in range(core_nodes)])
    
    Ci_core = {}
    for dpi in trees_dpis_core:
        C_core = np.zeros((len(trees_dpis_core[dpi].nodes), len(trees_dpis_core[dpi].nodes)))
        for nodes, an in nx.algorithms.lowest_common_ancestors.all_pairs_lowest_common_ancestor(trees_dpis_core[dpi]):
            C_core[int(nodes[0]), int(nodes[1])] = nx.dijkstra_path_length(trees_dpis_core[dpi], an, nodes[0]) + nx.dijkstra_path_length(trees_dpis_core[dpi], an, nodes[1])
            C_core[int(nodes[1]), int(nodes[0])] = C_core[int(nodes[0]), int(nodes[1])] 
        Ci_core[dpi] = C_core

    hrts_tps = {}
    for time in adata.obs["tps"].cat.categories:
        hrts_tps[time] = adata[adata.obs['tps'] == time].obs['ident'].cat.categories

    tps = adata.obs["tps"].cat.categories

    tau_b = 1.0
    
    dir_path = f"tmats_moslin_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}"

    if not os.path.exists(DATA_DIR / f"output/{dir_path}"):
        os.makedirs(DATA_DIR / f"output/{dir_path}")

    for dpi0 in hrts_tps[tps[0]]:
        C_early = beta_lineage_cost(adata, Ci_core, dpi0, args.beta)
        for dpi3 in hrts_tps[tps[1]]:
            C_mid = beta_lineage_cost(adata, Ci_core, dpi3, args.beta)
            if not os.path.isfile(DATA_DIR / f"output/{dir_path}/tmat-{dpi0}_{dpi3}_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}"):
                adata_c = adata[adata.obs["ident"].isin([dpi0, dpi3])].copy()
                df = pd.DataFrame(index=adata_c.obs_names, columns=adata_c.obs_names)
                df = df.combine_first(C_early)
                df = df.combine_first(C_mid)

                adata_c.obsp["cost_matrices"] = df
                adata_c.obsp["cost_matrices"] = adata_c.obsp["cost_matrices"].astype(float)
                estimate_marginals(adata_c, dpi0, dpi3, (tps[1]-tps[0]))
                
                if args.alpha == 0:
                    prob = TemporalProblem(adata=adata_c).prepare(
                        time_key="tps",
                        joint_attr="latent",
                        a="a_marginal", 
                        b="b_marginal")

                    prob = prob.solve(epsilon=args.epsilon, tau_a=args.tau_a, tau_b=tau_b)
                else:
                    prob = LineageProblem(adata=adata_c).prepare(
                        time_key="tps",
                        lineage_attr={"attr": "obsp", "key": "cost_matrices", "tag": "cost", "cost": "custom"},
                        joint_attr="latent",
                        a="a_marginal", 
                        b="b_marginal",
                        policy="sequential", 
                    )
                    prob = prob.solve(alpha=args.alpha, epsilon=args.epsilon, tau_a=args.tau_a, tau_b=tau_b)
                        
                if prob.solutions[list(prob.solutions.keys())[0]].converged:
                    print(f"ctrl - {dpi0}, dpi3 - {dpi3}")
                    tmat = pd.DataFrame(
                        prob.solutions[list(prob.solutions.keys())[0]].transport_matrix, 
                        index=adata.obs_names[adata.obs["ident"].isin([dpi0])],
                        columns=adata.obs_names[adata.obs["ident"].isin([dpi3])])
                    tmat.to_csv(DATA_DIR / f"output/{dir_path}/tmat-{dpi0}_{dpi3}_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}")
                else:
                    print(f"tmat for alpha {args.alpha} and {dpi0}-{dpi3} is invalid")

            for dpi7 in hrts_tps[tps[2]]:
                if not os.path.isfile(DATA_DIR / f"output/{dir_path}/tmat-{dpi3}_{dpi7}_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}"):
                    C_late = beta_lineage_cost(adata, Ci_core, dpi7, args.beta)
                 
                    adata_c = adata[adata.obs["ident"].isin([dpi3, dpi7])].copy()
                    df = pd.DataFrame(index=adata_c.obs_names, columns=adata_c.obs_names)
                    df = df.combine_first(C_mid)
                    df = df.combine_first(C_late)

                    adata_c.obsp["cost_matrices"] = df
                    adata_c.obsp["cost_matrices"] = adata_c.obsp["cost_matrices"].astype(float)
                    
                    estimate_marginals(adata_c, dpi3, dpi7, (tps[2]-tps[1]))
                    
                    if args.alpha == 0:
                        prob = TemporalProblem(adata=adata_c).prepare(
                            time_key="tps",
                            joint_attr="latent",                           
                            a="a_marginal", 
                            b="b_marginal",)

                        prob = prob.solve(epsilon=args.epsilon, tau_a=args.tau_a, tau_b=tau_b)
                    else:
                        prob = LineageProblem(adata=adata_c).prepare(
                            time_key="tps",
                            lineage_attr={"attr": "obsp", "key": "cost_matrices", "tag": "cost", "cost": "custom"},
                            joint_attr="latent",
                            a="a_marginal", 
                            b="b_marginal",
                            policy="sequential", 
                        )
                        prob = prob.solve(alpha=args.alpha, epsilon=args.epsilon, tau_a=args.tau_a, tau_b=tau_b)

                    if prob.solutions[list(prob.solutions.keys())[0]].converged:
                        print(f"dpi3 - {dpi3}, dpi7 - {dpi7}")
                        tmat = pd.DataFrame(
                            prob.solutions[list(prob.solutions.keys())[0]].transport_matrix, 
                            index=adata.obs_names[adata.obs["ident"].isin([dpi3])],
                            columns=adata.obs_names[adata.obs["ident"].isin([dpi7])]
                        ) 
                        tmat.to_csv(DATA_DIR / f"output/{dir_path}/tmat-{dpi3}_{dpi7}_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}")

                    else:
                        print(f"tmat for alpha {args.alpha} and {dpi3}-{dpi7} is invalid")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',
                        type=float,
                        required=False,
                        default=0.5)
    parser.add_argument('--epsilon',
                        type=float,
                        required=False,
                        default=0.05)
    parser.add_argument('--beta',
                        type=float,
                        required=False,
                        default=0.2)
    parser.add_argument('--tau_a',
                        type=float,
                        required=False,
                        default=0.9)
    parser.add_argument('--n_iters',
                        type=int,
                        required=False,
                        default=1)
    args = parser.parse_args()

    fit_couplings_all(args)
    