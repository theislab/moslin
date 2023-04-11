zebrafish analysis
===============

The analysis of the zebrafish data is a two-step process:

1. Calculation of transition matrices using moslin.
2. Downstream analysis of the transition matrices.

The input data files can be found on figshare: `scRNA-seq <https://figshare.com/account/projects/163357/articles/22494529>`_
and `lineage data <https://figshare.com/account/projects/163357/articles/22494541>`_.

File description
----------------
.. csv-table::
    :header: "File", "What it does"
    :delim: |

    `run_hu_moslin.sh` | Helper script used by ``run_sbatch_hu_moslin.py``. Calls the relevant functions from `utils.py` to run the simulations and performance analysis.
    `run_sbatch_hu_moslin.py` | Main SLURM script to calculate the couplings between the hearts at consecutive time points. Calling the command ``python3 run_sbatch_hu_moslin.py`` will instantiate sbatch calls to calculate all couplings. The output of each coupling is saved as a csv files under ``DATA_DIR/output/``.
    `hu_moslin_fit_couplings.py` | Main function called by ``run_hu_moslin.sh``. ``fit_couplings_all()`` is used to compute the couplings between all hearts at consecutive time points for the given input arguments (`alpha`, `epsilon`, `beta`, and `tau_a`). The couplings are saved as ``.csv`` files under ``DATA_DIR/output/``
    `Zebrafish_coupling_analysis.R` | Downstream analysis script. Apart from the couplings generated in the previous step, the downstream analysis additionally uses data files for single-cell annotation, cell type colors, lineage information and sample timepoints found `here <https://figshare.com/account/projects/163357/articles/22502974>`_.
