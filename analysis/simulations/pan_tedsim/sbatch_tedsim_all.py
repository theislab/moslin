import sys
from subprocess import Popen


DATA_DIR = "path/to/data"

seeds = [36489, 59707, 38128, 25295, 49142, 12102, 30139, 4698, 50349, 23860]
ssrs = [0.0, 0.1, 0.2, 0.3, 0.4]

for seed in seeds:
    for ssr in ssrs:
        cmdline0 = [
                    "sbatch",
                    "--gres=gpu:a5000:1,vmem:10g",
                    "--time=7-0",
                    "--mem=10gb",
                    f"--output={DATA_DIR}/logs/tedsim-all-{seed}-{ssr}.log",
                    f"--job-name=tedsim-all-{seed}-{ssr}",
                    "run_tedsim.sh",
                    str(seed),
                    str(ssr)
            ]
                
        print(" ".join(cmdline0))
        Popen(cmdline0)
