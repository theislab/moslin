# Analysis pipeline

We provide scripts which can be run on a SLURM based cluster scheduling system and output.

1. Run simulation and couplings: modify paths within `run.sh` to activate local environment and call: `python3 run_sbatch.py`.
This will:
    1. Produce simulated trajectories of types: "bifurcation", "convergent", "partial_convergent", and "mismatched_clusters"  using LineageOT by ([[FS-21]](https://www.nature.com/articles/s41467-021-25133-1)).
    2. Compute the couplings using:
        1. LineageOT
        2. OT
        3. GW
        4. moslin
    3. Evaluate coupling accuracy. The output of this analysis is saved under `DATA_DIR`.
2. Analyze results over all trajectories: run [`ZP_2023-04-02_2gene_analysis.ipynb`](https://github.com/theislab/moslin/tree/main/analysis/simulations/forrow_2gene/ZP_2023-04-02_2gene_analysis.ipynb) (this notebook  uses the `.csv` files `"{flow_type}_res_seeds.csv"`).
3. Visualize simulation properties of `bifurcation` trajectory: run [`ZP_2023-04-02_2gene_bifurcation_example.ipynb`](https://github.com/theislab/moslin/tree/main/analysis/simulations/forrow_2gene/ZP_2023-04-02_2gene_bifurcation_example.ipynb) (this notebook uses `"bifurcation_res_seeds.csv", "bifurcation_ancestor_errors_moslin.pkl" and  "bifurcation_descendant_errors_moslin.pkl"` files)