moslin's reproducibility repository
=====================================
Please open an `issue <https://github.com/theislab/moslin/issues/new>`_ should you experience difficulties reproducing any result.

Manuscript, code, tutorial and data
-------------------------
The moslin manuscript is available as a preprint at `bioRxiv`_. Under the hood,
moslin is based on `moscot`_ to solve the optimal transport problem of mapping
lineage-traced cells across time points. Specifically, we implement moslin via the
`LineageClass`_ , and we demonstrate a use case in our `tutorial`_.  

Raw published data is available from the Gene Expression Omnibus (GEO) under accession codes:

- `c elegans`_: `GSE126954 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954>`_.
- `zebrafish`_: `GSE159032  <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159032>`_.

Additionally, we simulated data using `LineageOT`_ and `TedSim`_. Processed data
 is available on `figshare`_. To ease reproducibility, our data examples can
 also be accessed through moscot's `dataset interface <https://moscot.readthedocs.io/en/latest/user.html#module-moscot.datasets>`_.

Navigating this repository
--------------------------
We've organized this repository along the categories below. Each folder contains
notebooks and scripts necessary to reproduce the results. We read data from `data`
and write figures to `figures`. 

Results
-------

.. csv-table:: Main Figures
   :header: "Figure", "Folder path"

    Fig. 2 (Simulations), `path <analysis/simulations/>`__
    Fig. 3 (C. elegans), `path <analysis/packer_c_elegans/>`__
    Fig. 4 (Zebrafish), `path <analysis/hu_zebrafish_linnaeus/>`__


.. _bioRxiv: TODO
.. _moscot: https://moscot-tools.org/
.. _tutorial: https://moscot.readthedocs.io/en/latest/notebooks/tutorials/100_lineage.html
.. _LineageOT: https://doi.org/10.1038/s41467-021-25133-1
.. _TedSim: https://doi.org/10.1093/nar/gkac235
.. _c elegans: https://doi.org/10.1126/science.aax1971
.. _zebrafish: https://doi.org/10.1038/s41588-022-01129-5
.. _figshare: TODO
