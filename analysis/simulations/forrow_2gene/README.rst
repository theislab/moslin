2-gene analysis
===============

File description
----------------
.. csv-table::
    :header: "File", "What it does"
    :delim: |

    `run.sh` | Helper script used by `run_sbatch.py`. Calls the relevant functions from `utils.py` to run the simulations and performance analysis.
    `run_sbatch.py` | Main SLURM script to simulate the trajectories, compute the couplings (`CoSpar <https://www.nature.com/articles/s41587-022-01209-1>`_, `LineageOT <https://www.nature.com/articles/s41467-021-25133-1)>`_, OT, GW, and moslin) and evaluate the accuracy. The output of this analysis is saved under `DATA_DIR`.
    `utils_ot.py` | Utilities used by `run.sh` and the analysis notebooks. Includes the main function, `run_seeds()`, which is used to simulate a trajectory of the given `flow_type` over 10 seeds and evaluate coupling performance of the different methods (CoSpar, LineageOT, OT, GW, and moslin). Running `run_seeds()` creates an output `.csv` file `"{flow_type}_res_seeds.csv"` holding the performance evaluation.
    `ZP_2023-11-25_2gene_analysis.ipynb` | Analysis of `bifurcation`, `convergent`, `partial_convergent`, and `mismatched_clusters` trajectories. In addition, it presents a visualization of the `bifurcation` trajectory. This notebook uses `"bifurcation_res_seeds.csv", "bifurcation_ancestor_errors_moslin.pkl" and "bifurcation_descendant_errors_moslin.pkl"` files and `.csv` files `"{flow_type}_res_seeds.csv"`.
