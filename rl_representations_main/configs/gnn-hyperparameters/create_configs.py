import itertools
import yaml
import copy

# Define the options for the parameters you want to substitute
params = {
    "autoencoder_lr": [0.001, 0.0001],
    "encoder_hidden_size": [64, 128],
    "encoder_num_layers": [2, 3]
}

# Read the baseline config
with open("../config_sepsis_gnn.yaml", "r") as file:
    baseline_config = yaml.safe_load(file)

# Generate all combinations (2^3 = 8)
keys = list(params.keys())
combinations = list(itertools.product(*(params[key] for key in keys)))

# For each combination, substitute the parameters in a copy of the baseline and write out a new config file
for i, combo in enumerate(combinations):
    # Create a copy so the baseline remains unchanged
    new_config = copy.deepcopy(baseline_config)
    
    # Substitute the three parameters with the current combination values
    for key, value in zip(keys, combo):
        new_config[key] = value

    # Write out the new config to a file
    with open(f"config_sepsis_gnn_{i}.yaml", "w") as outfile:
        yaml.dump(new_config, outfile, default_flow_style=False)

print("8 new YAML config files have been generated.")
