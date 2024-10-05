import jax

jax.config.update("jax_enable_x64", True)

import argparse
import os
import pickle as pkl

import networkx as nx
import numpy as np
import ot
import pandas as pd
import re
from scipy import stats
from sklearn.metrics import roc_auc_score

import scanpy as sc
from moscot.problems.time import LineageProblem, TemporalProblem

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

sys.path.insert(0, "../../../")
from paths import DATA_DIR

from pathlib import Path

DATA_DIR = DATA_DIR / "hu_zebrafish_linnaeus"
OUTPUT_DIR = DATA_DIR / "hu_zebrafish_linnaeus"


# set cell type annotation cutoff
cell_frequency_cutoff = 10
ct_annotations = pd.read_csv(DATA_DIR / "Tree_cells.csv")
ct_freqs = pd.DataFrame(data=ct_annotations.groupby(["Tree", "Cell_type"]).size())
ct_freqs.rename(mapper={0: "Frequency"}, axis=1, inplace=True)
ct_freqs_cutoff = ct_freqs[ct_freqs["Frequency"] > cell_frequency_cutoff]
ct_annotations_cutoff = ct_annotations.join(
    ct_freqs_cutoff, how="right", on=["Tree", "Cell_type"]
)
whitelist_col12_origin = ["Epicardium (Ventricle)", "Epicardium (Atrium)", "Fibroblasts (const.)", "Fibroblasts (cfd)", "Fibroblasts (cxcl12a)", "Fibroblasts (proliferating)"]
whitelist_nppc_origin = ["Endocardium (Ventricle)", "Endocardium (Atrium)", "Fibroblasts (spock3)"]


def calculate_AUROC(ct_persistency_result):
    ## Calculate AUROCs for persistency test
    AUCs = pd.DataFrame({'AUC': -1,
                         'Time': ['Ctrl', '3dpi'] })
    for AUC_index, AUC_row in AUCs.iterrows():
        run_time = AUC_row['Time']
        ROC_df = ct_persistency_result.copy()
        ROC_df['Expected_out'] = (~(ROC_df['Cell_type_from'] == ROC_df['Cell_type_to'])).astype(int)
        ROC_df = ROC_df[ROC_df['t1'] == run_time]

        # Since we are testing persistency, we only include cell types that are present at both timepoints. To do this
        # we select cell types in the intersection of t1 and t2 cell types. Note that this means we do not require
        # cell types to be present at both timepoints for every individual t1-t2 combination; the absence of cell
        # types in some datasets is an interesting challenge since the algorithm should be able to recognize this.
        this_shared_ct = np.intersect1d(ROC_df['Cell_type_from'].unique(), ROC_df['Cell_type_to'].unique())
        ROC_df = ROC_df[(ROC_df['Cell_type_from'].isin(this_shared_ct)) & (ROC_df['Cell_type_to'].isin(this_shared_ct))]

        AUCs.at[AUC_index, 'AUC'] = roc_auc_score(ROC_df['Expected_out'], ROC_df['Welch_p'])

    return AUCs


def convert_ratio_to_freq(transfer_ratios):
    ## Convert transfer ratios to percentages
    transfer_percentages = []

    for run_time in ['Ctrl', '3dpi']:

        ### Join transfer ratios with all possible combinations and set non-detected ratios to 0
        these_transfer_ratios = transfer_ratios.copy()
        these_transfer_ratios = these_transfer_ratios[these_transfer_ratios['t1'] == run_time]
        t1t2_combinations = these_transfer_ratios[['Tree_from', 'Tree_to']].drop_duplicates()
        t1t2_combinations['Join_string'] = t1t2_combinations['Tree_from'] + '_' + t1t2_combinations['Tree_to']
        all_ct_combinations = pd.DataFrame(
            [(x, y, z) for x in these_transfer_ratios['Cell_type_from'].unique() for y in
             these_transfer_ratios['Cell_type_to'].unique()
             for z in t1t2_combinations['Join_string']],
            columns=['Cell_type_from', 'Cell_type_to', 'Join_string'])
        all_ct_combinations[['Tree_from', 'Tree_to']] = all_ct_combinations['Join_string'].str.split('_',
                                                                                                     expand=True)

        all_ct_combinations = all_ct_combinations.merge(
            these_transfer_ratios[['Tree_from', 'Cell_type_from', 'ct_from_freq']].drop_duplicates(), how='left',
            on=['Tree_from', 'Cell_type_from'])
        all_ct_combinations = all_ct_combinations.merge(
            these_transfer_ratios[['Tree_to', 'Cell_type_to', 'ct_to_freq']].drop_duplicates(), how='left',
            on=['Tree_to', 'Cell_type_to'])
        all_ct_combinations.fillna(0, inplace=True)
        these_transfer_ratios = these_transfer_ratios[
            ['Cell_type_from', 'Cell_type_to', 'Tree_from', 'Tree_to', 'Transfer_ratio']].merge(all_ct_combinations,
                                                                                                on=[
                                                                                                    'Cell_type_from',
                                                                                                    'Cell_type_to',
                                                                                                    'Tree_from',
                                                                                                    'Tree_to'],
                                                                                                how='right')
        these_transfer_ratios.fillna(0, inplace=True)

        ### Calculate t1*t2 size products and join
        for combi_index, combi_row in t1t2_combinations.iterrows():
            t1_this_combi = these_transfer_ratios[these_transfer_ratios['Join_string'] == combi_row['Join_string']][
                ['Join_string', 'ct_from_freq']].drop_duplicates()
            t2_this_combi = these_transfer_ratios[these_transfer_ratios['Join_string'] == combi_row['Join_string']][
                ['Join_string', 'ct_to_freq']].drop_duplicates()
            t1t2_combinations.loc[combi_index, 't1_size'] = t1_this_combi['ct_from_freq'].sum()
            t1t2_combinations.loc[combi_index, 't2_size'] = t2_this_combi['ct_to_freq'].sum()

        t1t2_combinations['size_product'] = t1t2_combinations['t1_size'] * t1t2_combinations['t2_size']
        these_transfer_ratios = these_transfer_ratios.merge(
            t1t2_combinations[['Tree_from', 'Tree_to', 'Join_string', 'size_product']],
            on=['Tree_from', 'Tree_to', 'Join_string'])
        ### Calculate weighted average of transfer ratios and calculate percentages from
        average_transfer_ratio = these_transfer_ratios.groupby(['Cell_type_from', 'Cell_type_to']).apply(
            lambda x: np.average(x.Transfer_ratio, weights=x.size_product)).to_frame().rename({0: 'Transfer_ratio'},
                                                                                              axis=1)
        average_transfer_ratio['Percentage_from'] = average_transfer_ratio.groupby(['Cell_type_to'])[
            'Transfer_ratio'].transform(lambda x: 100 * x / x.sum())
        average_transfer_ratio['t1'] = run_time

        transfer_percentages.append(average_transfer_ratio.reset_index())

    transfer_percentages_df = pd.concat(transfer_percentages, axis=0, ignore_index=True)
    return transfer_percentages_df


def calculate_col12_nppc(transfer_percentages):
    ## Calculate percentages transferred to col12 cells (3dpi) and nppc cells (7dpi)

    transient_test = dict()

    col12_test = transfer_percentages.copy()
    col12_test = col12_test[(col12_test['t1'] == 'Ctrl') & (col12_test['Cell_type_to'] == 'Fibroblasts (col12a1a)')]
    transient_test['Percentage_from_col12_correct'] = [col12_test[col12_test['Cell_type_from'].isin(whitelist_col12_origin)]['Percentage_from'].sum()]

    nppc_test = transfer_percentages.copy()
    nppc_test = nppc_test[(nppc_test['t1'] == '3dpi') & (nppc_test['Cell_type_to'] == 'Fibroblasts (nppc)')]
    transient_test['Percentage_from_nppc_correct'] = [nppc_test[nppc_test['Cell_type_from'].isin(whitelist_nppc_origin)]['Percentage_from'].sum()]

    print(transient_test)
    return pd.DataFrame(transient_test)

def evaluate_transient_fibro_prob(tmat, dpi1, dpi2):
    t1_name = "Ctrl" if re.match("H[0-9]", dpi1) else "3dpi"
    transitions_wide = tmat.copy().reset_index()
    transitions_wide.rename(columns={transitions_wide.columns[0]: "From"}, inplace=True)
    transitions_long = pd.melt(
        transitions_wide, id_vars="From", var_name="To", value_name="Probability"
    )
    transitions_long = transitions_long.set_index("From").join(
        ct_annotations_cutoff.set_index("Cell").drop(
            labels=["Tree", "Frequency"], axis=1
        ),
        how="inner",
    )
    transitions_long.rename(
        mapper={"Cell_type": "Cell_type_from"}, axis=1, inplace=True
    )
    transitions_long = (
        transitions_long.reset_index().rename({"index": "From"}, axis=1).set_index("To")
    )
    transitions_long = transitions_long.join(
        ct_annotations_cutoff.set_index("Cell").drop(
            labels=["Tree", "Frequency"], axis=1
        ),
        how="inner",
    )
    transitions_long.rename(mapper={"Cell_type": "Cell_type_to"}, axis=1, inplace=True)

    ### Remove cell types that are under cutoff frequency
    ct_t1_freqs = (
        transitions_long[["From", "Cell_type_from"]]
        .drop_duplicates()
        .value_counts("Cell_type_from")
        .to_frame("counts")
    )
    ct_t1_cutoff = ct_t1_freqs[
        ct_t1_freqs["counts"] > cell_frequency_cutoff
    ].index.tolist()
    ct_t2_freqs = (
        transitions_long[["Cell_type_to"]]
        .reset_index()
        .drop_duplicates()
        .value_counts("Cell_type_to")
        .to_frame("counts")
    )
    ct_t2_cutoff = ct_t2_freqs[
        ct_t2_freqs["counts"] > cell_frequency_cutoff
    ].index.tolist()
    transitions_long = transitions_long[
        (transitions_long["Cell_type_from"].isin(ct_t1_cutoff))
        & (transitions_long["Cell_type_to"].isin(ct_t2_cutoff))
    ]
    ### Determine most likely precursors for all cell types at t2 in persistent cell types.
    transitions_long_for_argmax = transitions_long.reset_index(names = 'To')
    most_likely_transitions = transitions_long_for_argmax.loc[transitions_long_for_argmax.groupby('To')['Probability'].idxmax()]
    ### Remove type_to that is not in type_from
    most_likely_transitions = most_likely_transitions[most_likely_transitions['Cell_type_to'].isin(most_likely_transitions['Cell_type_from'].unique())]
    most_likely_transitions['t1'] = t1_name
    most_likely_transitions['Tree_from'] = dpi1
    most_likely_transitions['Tree_to'] = dpi2

    ### Calculate transfer rates between cell types
    inout_this = (
        transitions_long.groupby(["Cell_type_from", "Cell_type_to"])
        .sum("Probability")
        .rename({"Probability": "Transfer_ratio"}, axis=1)
    )
    inout_this["t1"] = t1_name
    inout_this["Tree_from"] = dpi1
    inout_this["Tree_to"] = dpi2
    inout_this = inout_this.join(
        ct_t1_freqs[ct_t1_freqs["counts"] > cell_frequency_cutoff],
        on=["Cell_type_from"],
    ).rename({"counts": "ct_from_freq"}, axis=1)
    inout_this = inout_this.join(
        ct_t2_freqs[ct_t2_freqs["counts"] > cell_frequency_cutoff], on=["Cell_type_to"]
    ).rename({"counts": "ct_to_freq"}, axis=1)
    inout_this = inout_this.reset_index()

    ### Loop over t2 cell types, test distribution of probabilities of same t1 cell type vs other t1 cell types
    test_grid = pd.DataFrame(
        [(x, y) for x in ct_t1_cutoff for y in ct_t2_cutoff],
        columns=["Cell_type_from", "Cell_type_to"],
    )
    test_grid["t1"] = t1_name
    test_grid["Tree_from"] = dpi1
    test_grid["Tree_to"] = dpi2
    test_grid["Welch_p"] = -1

    #### Outer loop over t2 types
    for type_t2 in ct_freqs_cutoff.loc[dpi2, :].index:

        ##### Slice data t2_type
        transitions_type_t2 = transitions_long[
            transitions_long["Cell_type_to"] == type_t2
        ]

        ##### Inner loop over t1 types
        for type_t1 in ct_freqs_cutoff.loc[dpi1, :].index:
            # print(type_t1)

            ###### Find index in test_grid
            j = test_grid.index[
                (test_grid["Cell_type_from"] == type_t1)
                & (test_grid["Cell_type_to"] == type_t2)
            ].tolist()
            if len(j) > 1:
                print("Multiple indices found")
                continue

            ###### Slice data t1_type and !t1_type
            x = transitions_type_t2[transitions_type_t2["Cell_type_from"] == type_t1][
                "Probability"
            ]
            y = transitions_type_t2[transitions_type_t2["Cell_type_from"] != type_t1][
                "Probability"
            ]
            if (np.std(x) == 0) & (np.std(y) == 0):
                continue
            ###### Perform welch test and output p-value
            test_grid.loc[j, "Welch_p"] = stats.ttest_ind(
                a=x, b=y, equal_var=False, alternative="greater"
            ).pvalue

    return inout_this, test_grid, most_likely_transitions


def beta_lineage_cost(adata, Ci, dpi, beta):
    cells = adata.obs["ident"].isin([dpi])
    cells_idx = np.where(adata.obs["ident"].isin([dpi]))[0]
    C = np.zeros((len(cells_idx), len(cells_idx)))
    for i, ci in enumerate(adata[cells].obs_names):
        C[i, :] = Ci[dpi][
            adata.obs.loc[[ci], "core"].values,
            adata.obs.loc[adata[cells].obs_names, "core"].values,
        ]
    C = C / C.max()
    M = ot.utils.dist(adata[cells].obsm["latent"], adata[cells].obsm["latent"])
    M = M / M.max()

    return pd.DataFrame(
        beta * C + (1 - beta) * M,
        index=adata.obs_names[cells],
        columns=adata.obs_names[cells],
    )


def estimate_marginals(adata, dpi_source, dpi_target, time_diff):
    source_growth = (
        adata[(adata.obs["ident"].isin([dpi_source]))]
        .obs["cell_growth_rate"]
        .to_numpy()
    )
    target_growth = (
        adata[(adata.obs["ident"].isin([dpi_target]))]
        .obs["cell_growth_rate"]
        .to_numpy()
    )

    growth = np.power(source_growth, time_diff)
    a = growth
    a = a / a.sum()

    b = np.full(target_growth.shape[0], np.average(growth))
    b = b / b.sum()

    adata.obs["a_marginal"] = 0
    adata.obs["b_marginal"] = 0

    adata.obs.loc[adata.obs["ident"].isin([dpi_source]), "a_marginal"] = a
    adata.obs.loc[adata.obs["ident"].isin([dpi_target]), "b_marginal"] = b


def fit_couplings_all(args):

    ct_persistency_results = None
    transfer_ratios = None
    ct_persistency_argmax = None
    

    ct_persistency_results_dict = dict()
    transfer_ratios_dict = dict()
    ct_persistency_argmax_dict = dict()
    
    adata = sc.read_h5ad(DATA_DIR / "adata.h5ad")

    with open(DATA_DIR / "trees.pkl", "rb") as file:
        trees = pkl.load(file)

    trees_dpis = dict()
    trees_dpis_core = dict()
    core_nodes = 100
    for i, dpi in enumerate(adata.obs["ident"].cat.categories):
        trees_dpis[dpi] = trees[dpi]
        trees_dpis_core[dpi] = trees_dpis[dpi].subgraph(
            [str(n) for n in range(core_nodes)]
        )

    Ci_core = {}
    for dpi in trees_dpis_core:
        C_core = np.zeros(
            (len(trees_dpis_core[dpi].nodes), len(trees_dpis_core[dpi].nodes))
        )
        for (
            nodes,
            an,
        ) in nx.algorithms.lowest_common_ancestors.all_pairs_lowest_common_ancestor(
            trees_dpis_core[dpi]
        ):
            C_core[int(nodes[0]), int(nodes[1])] = nx.dijkstra_path_length(
                trees_dpis_core[dpi], an, nodes[0]
            ) + nx.dijkstra_path_length(trees_dpis_core[dpi], an, nodes[1])
            C_core[int(nodes[1]), int(nodes[0])] = C_core[int(nodes[0]), int(nodes[1])]
        Ci_core[dpi] = C_core

    hrts_tps = {}
    for time in adata.obs["tps"].cat.categories:
        hrts_tps[time] = adata[adata.obs["tps"] == time].obs["ident"].cat.categories

    tps = adata.obs["tps"].cat.categories

    tau_b = args.tau_b

    dir_path = f"moslin_max_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}_taub-{tau_b}_save-{args.save}"

    if not os.path.exists(OUTPUT_DIR / f"output/{dir_path}"):
        os.makedirs(OUTPUT_DIR / f"output/{dir_path}")

    if args.save:
        os.makedirs(OUTPUT_DIR / f"output/{dir_path}/tmats")

    converged_sum = 0
    tot_tmats = 0
    for i0, dpi0 in enumerate(hrts_tps[tps[0]]):
        C_early = beta_lineage_cost(adata, Ci_core, dpi0, args.beta)
        for dpi3 in hrts_tps[tps[1]]:
            C_mid = beta_lineage_cost(adata, Ci_core, dpi3, args.beta)
            adata_c = adata[adata.obs["ident"].isin([dpi0, dpi3])].copy()
            df = pd.DataFrame(index=adata_c.obs_names, columns=adata_c.obs_names)
            df = df.combine_first(C_early)
            df = df.combine_first(C_mid)

            adata_c.obsp["cost_matrices"] = df
            adata_c.obsp["cost_matrices"] = adata_c.obsp["cost_matrices"].astype(
                float
            )
            estimate_marginals(adata_c, dpi0, dpi3, (tps[1] - tps[0]))

            if args.alpha == 0:
                prob = TemporalProblem(adata=adata_c).prepare(
                    time_key="tps",
                    joint_attr="latent",
                    a="a_marginal",
                    b="b_marginal",
                )

                prob = prob.solve(
                    epsilon=args.epsilon,
                    tau_a=args.tau_a,
                    tau_b=tau_b,
                    max_iterations=1000,
                )
            else:
                prob = LineageProblem(adata=adata_c).prepare(
                    time_key="tps",
                    lineage_attr={
                        "attr": "obsp",
                        "key": "cost_matrices",
                        "tag": "cost_matrix",
                        "cost": "custom",
                    },
                    joint_attr="latent",
                    a="a_marginal",
                    b="b_marginal",
                    policy="sequential",
                )
                prob = prob.solve(
                    alpha=args.alpha,
                    epsilon=args.epsilon,
                    tau_a=args.tau_a,
                    tau_b=tau_b,
                    max_iterations=1000,
                    linear_solver_kwargs={"max_iterations": 1000},
                )

            converged_c = prob.solutions[list(prob.solutions.keys())[0]].converged
            converged_sum += converged_c
            tot_tmats += 1
            print(f"ctrl - {dpi0}, dpi3 - {dpi3} - converged={converged_c}")
            if converged_c:
                tmat = pd.DataFrame(
                        prob.solutions[list(prob.solutions.keys())[0]].transport_matrix,
                        index=adata.obs_names[adata.obs["ident"].isin([dpi0])],
                        columns=adata.obs_names[adata.obs["ident"].isin([dpi3])],
                    )

                # evaluate transient fibro probability
                inout_this, test_grid, most_likely_transitions = evaluate_transient_fibro_prob(
                        tmat=tmat, dpi1=dpi0, dpi2=dpi3
                    )

                transfer_ratios_dict[f"{dpi0}_{dpi3}"] = inout_this
                ct_persistency_results_dict[f"{dpi0}_{dpi3}"] = test_grid
                ct_persistency_argmax_dict[f"{dpi0}_{dpi3}"] = most_likely_transitions

                if transfer_ratios is not None:
                    transfer_ratios = pd.concat(
                            [transfer_ratios, inout_this], axis=0, ignore_index=True
                        )
                else:
                    transfer_ratios = inout_this

                if ct_persistency_results is not None:
                    ct_persistency_results = pd.concat(
                            [ct_persistency_results, test_grid],
                            axis=0,
                            ignore_index=True,
                        )
                else:
                    ct_persistency_results = test_grid

                if ct_persistency_argmax is not None:
                    ct_persistency_argmax = pd.concat(
                            [ct_persistency_argmax, most_likely_transitions],
                            axis=0,
                            ignore_index=True,
                        )
                else:
                        ct_persistency_argmax = most_likely_transitions

                if args.save:
                    tmat.to_parquet(
                            OUTPUT_DIR / f"output/{dir_path}/tmats/moslintmat-{dpi0}_{dpi3}_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}_taub-{tau_b}"
                        )
            if i0 == 0: # iterate once over all dpi3-dpi7 hearts
                for dpi7 in hrts_tps[tps[2]]:
                    C_late = beta_lineage_cost(adata, Ci_core, dpi7, args.beta)
                    adata_c = adata[adata.obs["ident"].isin([dpi3, dpi7])].copy()
                    df = pd.DataFrame(
                        index=adata_c.obs_names, columns=adata_c.obs_names
                    )
                    df = df.combine_first(C_mid)
                    df = df.combine_first(C_late)

                    adata_c.obsp["cost_matrices"] = df
                    adata_c.obsp["cost_matrices"] = adata_c.obsp[
                        "cost_matrices"
                    ].astype(float)

                    estimate_marginals(adata_c, dpi3, dpi7, (tps[2] - tps[1]))

                    if args.alpha == 0:
                        prob = TemporalProblem(adata=adata_c).prepare(
                            time_key="tps",
                            joint_attr="latent",
                            a="a_marginal",
                            b="b_marginal",
                        )

                        prob = prob.solve(
                            epsilon=args.epsilon, tau_a=args.tau_a, tau_b=tau_b,
                            max_iterations=1000,
                        )
                    else:
                        prob = LineageProblem(adata=adata_c).prepare(
                            time_key="tps",
                            lineage_attr={
                                "attr": "obsp",
                                "key": "cost_matrices",
                                "tag": "cost_matrix",
                                "cost": "custom",
                            },
                            joint_attr="latent",
                            a="a_marginal",
                            b="b_marginal",
                            policy="sequential",
                        )
                        prob = prob.solve(
                            alpha=args.alpha,
                            epsilon=args.epsilon,
                            tau_a=args.tau_a,
                            tau_b=tau_b,
                            max_iterations=1000,
                            linear_solver_kwargs={"max_iterations": 1000},
                        )

                    converged_c = prob.solutions[list(prob.solutions.keys())[0]].converged
                    converged_sum += converged_c
                    tot_tmats += 1
                    print(f"dpi3 - {dpi3}, dpi7 - {dpi7} - converged={converged_c}")
                    if converged_c:
                        tmat = pd.DataFrame(
                            prob.solutions[
                                list(prob.solutions.keys())[0]
                            ].transport_matrix,
                            index=adata.obs_names[adata.obs["ident"].isin([dpi3])],
                            columns=adata.obs_names[adata.obs["ident"].isin([dpi7])],
                        )

                        # evaluate transient fibro probability
                        inout_this, test_grid, most_likely_transitions = evaluate_transient_fibro_prob(
                            tmat=tmat,
                            dpi1=dpi3,
                            dpi2=dpi7
                        )

                        transfer_ratios_dict[f"{dpi3}_{dpi7}"] = inout_this
                        ct_persistency_results_dict[f"{dpi3}_{dpi7}"] = test_grid
                        ct_persistency_argmax_dict[f"{dpi3}_{dpi7}"] = most_likely_transitions

                        if transfer_ratios is not None:
                            transfer_ratios = pd.concat(
                                [transfer_ratios, inout_this], axis=0, ignore_index=True
                            )
                        else:
                            transfer_ratios = inout_this

                        if ct_persistency_results is not None:
                            ct_persistency_results = pd.concat(
                                [ct_persistency_results, test_grid],
                                axis=0,
                                ignore_index=True,
                            )
                        else:
                            ct_persistency_results = test_grid

                        if ct_persistency_argmax is not None:
                            ct_persistency_argmax = pd.concat(
                                [ct_persistency_argmax, most_likely_transitions],
                                axis=0,
                                ignore_index=True,
                            )
                        else:
                            ct_persistency_argmax = most_likely_transitions

                        if args.save:
                            tmat.to_parquet(
                                OUTPUT_DIR / f"output/{dir_path}/tmats/moslintmat-{dpi3}_{dpi7}_alpha-{args.alpha}_epsilon-{args.epsilon}_beta-{args.beta}_taua-{args.tau_a}_taub-{tau_b}"
                            )

    converged_all = converged_sum/tot_tmats
    print(f"dir_path - converged={converged_all}")
    ct_persistency_results.to_csv(
        OUTPUT_DIR / f"output/{dir_path}/ct_persistency_results_{converged_all}.csv"
    )
    transfer_ratios.to_csv(OUTPUT_DIR / f"output/{dir_path}/transfer_ratios_{converged_all}.csv")
    ct_persistency_argmax.to_csv(OUTPUT_DIR / f"output/{dir_path}/ct_persistency_argmax_{converged_all}.csv")

    with open(OUTPUT_DIR / f"output/{dir_path}/transfer_ratios_dict_{converged_all}.pkl", "wb") as file:
        pkl.dump(transfer_ratios_dict, file)

    with open(
        OUTPUT_DIR / f"output/{dir_path}/ct_persistency_results_dict_{converged_all}.pkl", "wb"
    ) as file:
        pkl.dump(ct_persistency_results_dict, file)

    with open(
        OUTPUT_DIR / f"output/{dir_path}/ct_persistency_argmax_dict_{converged_all}.pkl", "wb"
    ) as file:
        pkl.dump(ct_persistency_argmax_dict, file)

    AUCs = calculate_AUROC(ct_persistency_results)
    transfer_freqs = convert_ratio_to_freq(transfer_ratios)

    transient_test = calculate_col12_nppc(transfer_freqs)
    transient_test["converged_all"] = converged_all

    AUCs.to_csv(
        OUTPUT_DIR / f"output/{dir_path}/AUCs_{converged_all}.csv"
    )
    transfer_freqs.to_csv(OUTPUT_DIR / f"output/{dir_path}/transfer_freqs_{converged_all}.csv")
    transient_test.to_csv(OUTPUT_DIR / f"output/{dir_path}/transient_test_{converged_all}.csv")

    return AUCs, transfer_freqs, transient_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=False, default=0.5)
    parser.add_argument("--epsilon", type=float, required=False, default=0.05)
    parser.add_argument("--beta", type=float, required=False, default=0.2)
    parser.add_argument("--tau_a", type=float, required=False, default=0.9)
    parser.add_argument("--tau_b", type=float, required=False, default=1.0)
    parser.add_argument("--save", type=int, required=False, default=0)
    args = parser.parse_args()

    fit_couplings_all(args)
