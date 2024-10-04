from pathlib import Path

ROOT = Path(__file__).parents[2].resolve()


DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

TIME_KEY = "assigned_batch_time"
REP_KEY = "X_pca"
TIMEPOINTS = [170, 210, 270, 330, 390, 450, 510]
KEY_STATE_COLORS = {
    "Ciliated neuron": "#B3424C",
    "Non-ciliated neuron": "#4C8EB4",
    "Glia and excretory": "#95AD4D",
}
