from subprocess import Popen
import os
import sys


sys.path.insert(0, "/cs/labs/mornitzan/zoe.piran/research/projects/moslin_reproducibility")

from paths import DATA_DIR

DATA_DIR = DATA_DIR / "hu_zebrafish_linnaeus"


alpha = 0.5
epsilon = 1e-2
beta = 0.2
tau_a = 0.9

alphas = [0, 0.4, 0.6]
epsilons = [5e-3, 1e-2, 1e-1]
betas = [0.1, 0.3]
tau_as = [0.85, 0.95]


for alpha_c in alphas:
    cmdline0 = ['sbatch', '--gres=gpu:a5000:1', '--time=1-0', '--mem=300gb', 
                    f'--output={DATA_DIR}/logs/hu-moslin-{tau_a}-{epsilon}-{alpha_c}-{beta}.log',
                    f'--job-name=hu-moslin-{tau_a}-{epsilon}-{alpha_c}-{beta}',
                    'run_hu_moslin.sh', str(alpha_c), str(epsilon), str(beta), str(tau_a)
                    ]

    print(' '.join(cmdline0))
    Popen(cmdline0)


for eps_c in epsilons:
    cmdline0 = ['sbatch', '--gres=gpu:a5000:1', '-c1', '--time=1-0', '--mem=300gb',
                f'--output={DATA_DIR}/logs/hu-unb-{tau_a}-{eps_c}-{alpha}-{beta}.log',
                f'--job-name=hu-moslin-{tau_a}-{eps_c}-{alpha}-{beta}',
                'run_hu_moslin.sh', str(alpha), str(eps_c), str(beta), str(tau_a)
                ]

    print(' '.join(cmdline0))
    Popen(cmdline0)

for beta_c in betas:
    cmdline0 = ['sbatch', '--gres=gpu:a5000:1', '-c1', '--time=1-0', '--mem=300gb',
                f'--output={DATA_DIR}/logs/hu-unb-{tau_a}-{epsilon}-{alpha}-{beta_c}.log',
                f'--job-name=hu-moslin-{tau_a}-{epsilon}-{alpha}-{beta_c}',
                'run_hu_moslin.sh', str(alpha), str(epsilon), str(beta_c), str(tau_a)
                ]

    print(' '.join(cmdline0))
    Popen(cmdline0)


for tau_c in tau_as:
    cmdline0 = ['sbatch', '--gres=gpu:a5000:1', '-c1', '--time=1-0', '--mem=300gb',
                f'--output={DATA_DIR}/logs/hu-unb-{tau_c}-{epsilon}-{alpha}-{beta}.log',
                f'--job-name=hu-moslin-{tau_c}-{epsilon}-{alpha}-{beta}',
                'run_hu_moslin.sh', str(alpha), str(epsilon), str(beta), str(tau_c)
                ]

    print(' '.join(cmdline0))
    Popen(cmdline0)
