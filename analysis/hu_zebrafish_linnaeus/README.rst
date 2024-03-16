Zebrafish analysis
===============

The analysis of the zebrafish data is a two-step process:

1. Calculation of transition matrices using moslin.
2. Downstream analysis of the transition matrices.

The input data files can be found on  `figshare <https://doi.org/10.6084/m9.figshare.c.6533377.v1>`_: ``adata`` file with the scRNA-seq data under ``Zebrafish heart regeneration`` and the lineage data as a ``.pkl`` file at ``Zebrafish reconstructed trees``.

File description
----------------
.. csv-table::
    :header: "File", "What it does"
    :delim: |

    run_hu_moslin.sh (run_hu_lot.sh) | Helper script used by ``run_sbatch_hu_moslin.py`` (``run_sbatch_hu_lot.py`` ). Calls the relevant functions from `utils.py` to run the simulations and performance analysis using moslin (LineageOT) mapping.
    run_sbatch_hu_moslin.py (run_sbatch_hu_lot.py) | Main SLURM script to calculate the couplings between the hearts at consecutive time points. Calling the command ``python3 run_sbatch_hu_moslin.py`` will instantiate sbatch calls to calculate all couplings using moslin (LineageOT). The output of each coupling is saved as a ``.csv`` file under ``data/hu_zabrafish_linnaeus/output/``.
    hu_moslin_fit_couplings.py (hu_lot_fit_couplings.py) | Main function called by ``run_hu_moslin.sh`` (``run_hu_lot.sh``). ``fit_couplings_all()`` is used to compute the couplings between all hearts at consecutive time points for the given input moslin arguments, `alpha`, `epsilon`, `beta`, and `tau_a` (given LineageOT argument `epsilon`). The couplings are saved as ``.csv`` files under ``DATA_DIR/output/``.
    Zebrafish_coupling_analysis.R | Downstream analysis script. Apart from the couplings generated in the previous step, the downstream analysis additionally uses data files for single-cell annotation, cell type colors, lineage information and sample timepoints found `here <https://figshare.com/account/projects/163357/articles/22502974>`_.
