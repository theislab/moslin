TedSim simulation analysis
==========================

File description
----------------
.. csv-table::
    :header: "File", "What it does"
    :delim: |

    `grid_config.txt` (`grid_config_subsample.txt`) | Config files with tasks to launch for main (subsample) evaluation.
    `tedsim_array.sh` (`tedsim_array_subsample.sh`) | Will instantiate sbatch calls to calculate couplings according to the relevant grid. Evaluates all methods (`CoSpar <https://www.nature.com/articles/s41587-022-01209-1>`_, `LineageOT <https://www.nature.com/articles/s41467-021-25133-1)>`_, and moslin). Calls the relevant functions from `tedsim_fit.py` to run the performance analysis.
    `tedsim_fit.py` | Main code to run the moslin, `LineageOT <https://www.nature.com/articles/s41467-021-25133-1>`_ , and `CoSpar <https://www.nature.com/articles/s41587-022-01209-1>`_ analysis.
    `utils_analysis.py` | Utilities for data visualization in `ZP_2023-11-02_tedsim_analysis.ipynb`.
    `utils_run.py` | Utilities for `tedsim_fit.py`.
    `ZP_2024-06-14_tedsim_analysis-main.ipynb` | Visualize the main results - TedSim data and performance (Fig. 2e-f, Supp. Fig. 2d). Uses a pre-computed TedSim simulation to visualize the initial state tree, simulated tree and gene expression. Imports the grid search results and visualizes the cost as a function of method and stochastic silencing rate.
    `ZP_2024-06-14_tedsim_analysis-hamming-distance.ipynb` | Visualize the hamming distance (Supp. Fig. 2a)
    `ZP_2024-06-14_tedsim_analysis-subsample.ipynb` | Visualize the hamming distance (Supp. Fig. 2e)
