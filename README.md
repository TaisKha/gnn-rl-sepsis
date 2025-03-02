## Run instruction

1. Install Pipenv and run `pipenv shell`.
2. Check correctness of configs for common and model specific in `rl_representations_main/configs`.
3. Make sure that BC cloning model is there as it is crucial for the evaluation. If not train it. See instructions in `rl_representations_main/README.md`.
4. Go to rl-representations folder and run `python3 scripts/train_model.py --autoencoder GNN`

## Pipenv hints

IMPORTANT: after updating Pipfile manually, we need to run 
`pipenv update`, which is `pipenv lock`+`pipenv sync`.

It is useful to with with `--verbose` option to see error step by step.


## Config hints

1. Only one of the following parameters should be True: 
- log_autoencoder_training
- log_BCQ_training

By setting them to true, the code is running either in encoder-decoder training mode or in policy training mode.

2. In `config_sepsis_gnn.yaml` gnn_gentype parameter is responsible for the choice of GNN model. 1 or 2 for SAGE and GATv2, respectively.

