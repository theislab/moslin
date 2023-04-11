TedSim simulation analysis
==========================

File description
----------------
.. csv-table::
    :header: "File", "What it does"
    :delim: |

    `lot.yml` | Config file used by `run.py` to benchmark `LineageOT`.
    `moslin.yml` | Config file used by `run.py` to benchmark `moslin`.
    `run.py` | Main script to run the moslin/`LineageOT <https://www.nature.com/articles/s41467-021-25133-1>`_ analysis.
    `utils_analysis.py` | Utilities for data visualization in `ZP_2023-04-02_tedsim_analysis.ipynb`.
    `utils_run.py` | Utilities for `run.py`.
    `ZP_2023-04-02_tedsim_analysis.ipynb` | Visualize the results - TedSim data and performance. Uses a pre-computed TedSim simulation to visualize the initial state tree, simulated tree and gene expression. Imports the grid search results and visualizes the cost as a function of method and stochastic silencing rate. 
