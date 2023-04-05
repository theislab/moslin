C. elegans analysis
====================
We showcase moslin on C. elegans embryogenesis in Fig. 3 and corresponding
supplementary Figures. The following table describes the files that need to be
run to reproduce our analysis. Scripts and
notebooks should be executed in the order given by the table. Data files for
this analysis are available from `figshare`_, details are given in the notebooks
themselves.

File description
----------------

.. csv-table::
   :delim:|
   :header: "File"|"What it does"

    `run.py` | Benchmark moslin and `LineageOT`_ across a grid of hyperparameters using `SEML`_. The gridsearch parameters are specific in `moslin.yml` and `lot.yml` for moslin and LineageOT, respectively. Utilities for the benchmark can be found in `utils.py`.
    `ML_2023-03-31_explore_c_elegans.ipynb` | Visualize the two data subsets using in the manuscript, all cells with precise lineage information and the ABpxp lineage.
    `ML_2023-03-31_prepare_data.ipynb` | Clean up provided metadata, add TF annotations, aggregate clusters and assign custom colors.
    `ML_2023-03-31_celegans_bar.ipynb` | Visualize the benchmark results in terms of the mean error.
    `ML_2023-03-31_moslin_abpxp.ipynb` | Run moslin on the ABpxp lineage using the identified hyperparameters.
    `ML_2023-03-31_zoom_in_abpxp.ipynb` | Zoom-in onto one pair of time points and explore method differences.
    `ML_2023-03-31_palantir.ipynb` | Run `Palantir`_ and `MAGIC`_ to obtain a pseudotime and imputed gene expression, respectively. Imputed data is only used to visualize gene expression trends.
    `ML_2023-03-31_cellrank_abpxp.ipynb` | Run CellRank on the ABpxp lineage, based on moslin's couplings. Compute and visualize terminal states, fate probabilities, driver genes, and expression trends.

.. _figshare: TODO
.. _SEML: https://github.com/TUM-DAML/seml
.. _LineageOT: https://doi.org/10.1038/s41467-021-25133-1
.. _Palantir: https://doi.org/10.1038/s41587-019-0068-4
.. _MAGIC: https://doi.org/10.1016/j.cell.2018.05.061
