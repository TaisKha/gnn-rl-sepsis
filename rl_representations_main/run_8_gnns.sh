# Loop for each config file (0 to 7)
for i in {0..7}
do
    if [ $i -eq 0 ]; then
        # In the first pane, run the command
        tmux send-keys -t my_experiment "python3 scripts/train_model.py --autoencoder GNN --autoencoder_config_override configs/gnn-hyperparameters/config_sepsis_gnn_${i}.yaml" C-m
    else
        # For subsequent panes, split the window and run the command
        tmux split-window -h -t my_experiment "python3 scripts/train_model.py --autoencoder GNN --autoencoder_config_override configs/gnn-hyperparameters/config_sepsis_gnn_${i}.yaml"
        tmux select-layout -t my_experiment tiled
    fi
done
