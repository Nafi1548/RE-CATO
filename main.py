import os
import sys
import json
import torch
from morbo.run_one_replication import run_one_replication

def run():
    # 1. Setup paths based on your folder structure
    exp_dir = "experiments/rover"
    config_path = os.path.join(exp_dir, "config.json")
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    # 2. Load your settings
    with open(config_path, "r") as f:
        config = json.load(f)

    # 3. Define a save callback to store results
    def save_callback(output):
        save_path = os.path.join(exp_dir, "results_seed_1.pt")
        torch.save(output, save_path)
        print(f"Results synced to {save_path}")

    # 4. Execute the replication
    # We unpack the config dictionary directly into the function
    run_one_replication(
        seed=1,
        label="morbo",
        save_callback=save_callback,
        **config
    )

if __name__ == "__main__":
    run()