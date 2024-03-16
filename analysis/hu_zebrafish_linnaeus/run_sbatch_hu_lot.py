import sys
import os
from subprocess import Popen

LCL_DIR = ""
sys.path.insert(0, LCL_DIR)

from paths import DATA_DIR

DATA_DIR = DATA_DIR / "hu_zebrafish_linnaeus"


epsilons = [0.001, 0.01] 
tau_as = [0.6, 0.8, 0.9]
save = 0

for tau_ac in tau_as:
    for epsilon_c in epsilons:
        cmdline0 = [
            "sbatch",
            "--gres=gpu:a5000:1",
            "--time=1-0",
            "--mem=100gb",
            "--account=mornitzan",
            f"--output={DATA_DIR}/logs/hu-lot-{tau_ac}-{epsilon_c}-{save}.log",
            f"--job-name=hu-lot-{tau_ac}-{epsilon_c}-{save}",
            "run_hu_lot.sh",
            str(epsilon_c),
            str(tau_ac),
            str(save)
        ]
        print(" ".join(cmdline0))
        Popen(cmdline0)


