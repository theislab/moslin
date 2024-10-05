import sys
import os
from subprocess import Popen

LCL_DIR = ""
sys.path.insert(0, LCL_DIR)

from paths import DATA_DIR

DATA_DIR = DATA_DIR / "hu_zebrafish_linnaeus"


epsilons = [0.05, 0.01, 0.1, 1.0]
alphas = [0.01, 0.1, 0.15, 0.5]
betas = [0.0, 0.2, 0.4]
tau_as = [0.4, 0.5, 0.6, 0.9, 1.0]
tau_bs = [0.95, 1.0]
save = 0

# run alpha + taus + epsilon_c + beta 
for alpha_c in alphas:
    for tau_ac in tau_as:
        for tau_bc in tau_bs:
            for epsilon_c in epsilons:
                for beta_c in betas:
                    cmdline0 = [
                                "sbatch",
                                "--gres=gpu:a5000:1",
                                "--time=1-0",
                                "--mem=100gb",
                                f"--output={DATA_DIR}/logs/hu-moslin-{tau_ac}-{tau_bc}-{epsilon_c}-{alpha_c}-{beta_c}-{save}.log",
                                f"--job-name=hu-moslin-{tau_ac}-{tau_bc}-{epsilon_c}-{alpha_c}-{beta_c}-{save}",
                                "run_hu_moslin.sh",
                                str(alpha_c),
                                str(epsilon_c),
                                str(beta_c),
                                str(tau_ac),
                                str(tau_bc),
                                str(save)
                            ]

                    print(" ".join(cmdline0))
                    Popen(cmdline0)