import sys
from subprocess import Popen

sys.path.insert(0, "../../../")

from paths import DATA_DIR

DATA_DIR = DATA_DIR / "simulations/forrow_2gene"


for flow_type in [
    "bifurcation",
    "convergent",
    "partial_convergent",
    "mismatched_clusters",
]:
    cmdline0 = [
        "sbatch",
        "--gres=gpu:a5000:1",
        "--time=1-0",
        "--mem=100gb",
        f"--output={DATA_DIR}/logs/sim-2gene-{flow_type}.log",
        f"--job-name=sim-2gene-{flow_type}",
        "run.sh",
        str(flow_type),
    ]

    print(" ".join(cmdline0))
    Popen(cmdline0)
