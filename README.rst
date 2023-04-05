moslin's reproducibility repository
=====================================
Please open an `issue <https://github.com/theislab/moslin/issues/new>`_ should you experience difficulties reproducing any result.

Manuscript, code, tutorials and data
-------------------------
The moslin manuscript is available as a preprint at `bioRxiv`_. Under the hood,
moslin is based on `moscot`_ to solve the optimal transport problem of mapping
lineage-traced cells across time points. Specifically, we implement moslin via the
`LineageClass`_ , we demonstrate a use case in our `tutorial`_ and we showcase
how to work with `tree distances`_ in an example.

Raw published data is available from the Gene Expression Omnibus (GEO) under accession codes:

- `c elegans`_: `GSE126954 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954>`_.
- `zebrafish`_: `GSE159032  <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159032>`_.

Additionally, we simulated data using `LineageOT`_ and `TedSim`_. Processed data
is available on `figshare`_. To ease reproducibility, our data examples can
also be accessed through moscot's `dataset interface <https://moscot.readthedocs.io/en/latest/user.html#module-moscot.datasets>`_.

Navigating this repository
--------------------------
We've organized this repository along the categories below. Each folder contains
notebooks and scripts necessary to reproduce the results. We read data from `data <data/>`_
and write figures to `figures <figures/>`_.

Results
-------

.. csv-table::
   :header: "Application", "Folder path"

    Simulated data (Fig. 2), `analysis/simulations/ <analysis/simulations/>`__
    C elegans embryogenesis (Fig. 3), `analysis/packer_c_elegans/ <analysis/packer_c_elegans/>`__
    Zebrafish heart regeneration (Fig. 4), `analysis/hu_zebrafish_linnaeus/ <analysis/hu_zebrafish_linnaeus/>`__


.. _bioRxiv: TODO
.. _moscot: https://moscot-tools.org/
.. _LineageClass: https://moscot.readthedocs.io/en/latest/genapi/moscot.problems.time.LineageProblem.html
.. _tree distances: https://moscot.readthedocs.io/en/latest/notebooks/examples/problems/600_leaf_distance.html
.. _tutorial: https://moscot.readthedocs.io/en/latest/notebooks/tutorials/100_lineage.html
.. _downstream analysis:
.. _LineageOT: https://doi.org/10.1038/s41467-021-25133-1
.. _TedSim: https://doi.org/10.1093/nar/gkac235
.. _c elegans: https://doi.org/10.1126/science.aax1971
.. _zebrafish: https://doi.org/10.1038/s41588-022-01129-5
.. _figshare: TODO
