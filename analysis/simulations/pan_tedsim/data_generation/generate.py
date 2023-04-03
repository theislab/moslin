import pathlib
import subprocess

import seml
from sacred import Experiment
from sacred.run import Run

ENV_NAME = "dorc2"

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run: Run) -> None:
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def generate(p_a: float, ss: float, seed: int) -> None:
    pathlib.Path("./output").mkdir(exist_ok=True, parents=True)
    res = subprocess.run(
        f"""conda init bash
        conda activate {ENV_NAME}
        Rscript generate_data.R {p_a} {ss} {seed}
        """,
        shell=True,
        executable="/bin/bash",
        check=True,
    )
    res.check_returncode()
