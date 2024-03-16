TedSim simulation analysis
==========================

File description
----------------
.. csv-table::
    :header: "File", "What it does"
    :delim: |

    `run_tedsim.sh` | Helper script used by `sbatch_tedsim_all.py`. Calls the relevant functions from `tedsim_fit.py` to run the performance analysis.
    `sbatch_tedsim_all.py` | Main SLURM script to compute the couplings (`CoSpar <https://www.nature.com/articles/s41587-022-01209-1>`_, `LineageOT <https://www.nature.com/articles/s41467-021-25133-1)>`_, and moslin) and evaluate the accuracy.
    `tedsim_fit.py` | Main code to run the moslin, `LineageOT <https://www.nature.com/articles/s41467-021-25133-1>`_ , and `CoSpar <https://www.nature.com/articles/s41587-022-01209-1>`_ analysis.
    `utils_analysis.py` | Utilities for data visualization in `ZP_2023-11-02_tedsim_analysis.ipynb`.
    `utils_run.py` | Utilities for `tedsim_fit.py`.
    `ZP_2023-11-02_tedsim_analysis.ipynb` | Visualize the results - TedSim data and performance. Uses a pre-computed TedSim simulation to visualize the initial state tree, simulated tree and gene expression. Imports the grid search results and visualizes the cost as a function of method and stochastic silencing rate. 
