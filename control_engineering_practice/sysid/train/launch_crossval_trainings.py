import itertools
import subprocess
import json
import os
from datetime import datetime

# === Config ===
TRAIN_SCRIPT = "train_lstm_model_multistep.py"  # your main training file
LOG_DIR = "./logs_crossval"
os.makedirs(LOG_DIR, exist_ok=True)

# All available trajectories
ALL_TRAJS = ["random1", "melon", "square", "multisine"]

# Generate all unique 3-element combinations
combinations = [
    ["random1", "melon", "square"],
    ["melon", "square", "multisine"],
    ["random1", "melon", "multisine"],
    ["square", "multisine", "random1"],  # example 5th
]

# You can also use this instead if you want *all unique 3-combinations*:
# combinations = list(itertools.combinations(ALL_TRAJS, 3))

# === Launcher ===
for i, train_trajs in enumerate(combinations, start=1):
    model_name = f"lstm_crossval_run{i}_" + "_".join(train_trajs)
    log_file = os.path.join(LOG_DIR, f"{model_name}.log")

    print(f"🚀 Launching run {i}: {train_trajs} → {model_name}")

    cmd = [
        "python", TRAIN_SCRIPT,
        "--train_trajs", json.dumps(train_trajs),
        "--model_name", model_name
    ]

    with open(log_file, "w") as log:
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

print("✅ All runs launched. Logs are being written to:", LOG_DIR)
