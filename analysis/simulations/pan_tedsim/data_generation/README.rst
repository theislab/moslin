TedSim data generation
======================
This directory contains all the necessary files to generate `AnnData <https://anndata.readthedocs.io/en/latest/>`_
objects for the `TedSim <https://academic.oup.com/nar/article/50/8/4272/6567477>`_ simulation analysis.

File description
----------------
.. csv-table::
    :header: "File", "What it does"
    :delim: |

    `config.yml` | `seml <https://github.com/TUM-DAML/seml>`_ configuration file.
    `convert_to_adata.py` | Helper script to convert the output of `generate_data.R` to `AnnData <https://anndata.readthedocs.io/en/latest/>`_.
    `generate.py` | Main `seml <https://github.com/TUM-DAML/seml>`_ script for data generation.
    `generate_data.R` | R script to generate `TedSim <https://academic.oup.com/nar/article/50/8/4272/6567477>`_ data.
