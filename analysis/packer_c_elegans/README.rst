C. elegans analysis
===================
We showcase moslin on C. elegans embryogenesis in Fig. 3 and corresponding
supplementary Figures. The following table describes the files that need to be
run to reproduce our analysis. Scripts and
notebooks should be executed in the order given by the table. Data files for
this analysis are available from `figshare`_, details are given in the notebooks
themselves.

File description
----------------
.. csv-table::
    :header: "File/Folder", "What it does"
    :delim: |

    ``run_gridsearch`` | Benchmark moslin, pure Gromow-Wasserstein, pure OT and `LineageOT`_ across a grid of hyperparameters using `wandb`_. The gridsearch parameters are specified in ``moslin.yml`` and ``lot.yml`` for moslin and LineageOT, respectively. Utilities for the benchmark can be found in ``utils.py``.
    ``ML_2024-03-11_explore_c_elegans.ipynb`` | Visualize the two data subsets employed in the manuscript: all cells with precise lineage information and the ABpxp lineage.
    ``ML_2024-03-11_prepare_data.ipynb`` | Clean up provided metadata, add TF annotations, aggregate clusters and assign custom colors.
    ``ML_2023-03-11_celegans_bar.ipynb`` | Visualize the benchmark results in terms of the mean error.
    ``ML_2024-03-12_GW_performance.ipynb`` | Show that pure GW performs better on precise lineage information when initialized with the OT solution. 
    ``ML_2024-03-11_compute_couplings.ipynb`` | Compute couplings using hyperparmeters identified in the gridsearch. 
    ``ML_2024-03-12_palantir.ipynb`` | Run `Palantir`_ and `MAGIC`_ to obtain a pseudotime and imputed gene expression, respectively. Imputed data is only used to visualize gene expression trends.
    ``ML_2024-03-12_zoom_in_abpxp.ipynb`` | Zoom-in onto one pair of time points and explore method differences.
    ``ML_2024-03-12_cellrank2_{METHOD}.ipynb`` | Compare cellular trajectories by combining CellRank 2 with different methods (`METHOD` in {ot, gw, moslin, LineageOT, palantir, cytotrace})

.. _figshare: https://doi.org/10.6084/m9.figshare.c.6533377.v1
.. _wandb: https://wandb.ai
.. _LineageOT: https://doi.org/10.1038/s41467-021-25133-1
.. _Palantir: https://doi.org/10.1038/s41587-019-0068-4
.. _MAGIC: https://doi.org/10.1016/j.cell.2018.05.061
