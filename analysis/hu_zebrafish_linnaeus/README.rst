Reproducing the zebrafish analysis
==================================
The analysis of the zebrafish data is a two-step process; the first step is the calculation of transition matrices using
moslin, and the second is the downstream analysis of these transition matrices.

We provide `code <https://github.com/theislab/moslin/blob/analysis/simulations/analysis/hu_zebrafish_linnaeus/run_sbatch_hu_moslin.py>`_
to obtain the couplings between hearts at consecutive time points on a SLURM based cluster scheduling system.
The input data files can be found on figshare: `scRNA-seq <https://figshare.com/account/projects/163357/articles/22494529>`_
and `lineage data <https://figshare.com/account/projects/163357/articles/22494541>`_.
Calling the command ``python3 run_sbatch_hu_moslin.py`` will instantiate sbatch calls to calculate all couplings.
The output of each coupling is saved as a csv files under ``DATA_DIR/output/``.

Downstream analysis can now be done through
`this script <https://github.com/theislab/moslin/blob/main/analysis/hu_zebrafish_linnaeus/Zebrafish_coupling_analysis.R>`_.
Apart from the couplings generated in the previous step, the downstream analysis additionally uses data files for
single-cell annotation, cell type colors, lineage information and sample timepoints found
`here <https://figshare.com/account/projects/163357/articles/22502974>`_.
