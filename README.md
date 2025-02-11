## Run instructions

1. pipenv shell
2. check configs common and model specific
3. make sure that BC cloning model is there. if not train it.
4. go to rl-representations folder and run `python3 scripts/train_model.py --autoencoder GNN`

## Useful stuff

IMPORTANT: after updating Pipfile manually, we need to run 
`pipenv update`, which is `pipenv lock`+`pipenv sync`.

It is useful to with with `--verbose` option to see error step by step.



