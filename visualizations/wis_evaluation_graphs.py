import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

SMOOTH = True
ALPHA = 0.01

def get_averaged_runs_data(project_name, run_ids, group_name, alpha_param=0.10, smooth=False):
    api = wandb.Api()
    runs_data = []

    # First, collect and smooth each run separately
    for run_id in run_ids:
        run = api.run(f"{project_name}/{run_id}")
        history = run.scan_history()
        history = pd.DataFrame(run.scan_history(keys=['training_iters', 'wis']))
        
        if smooth:
            # Apply exponential smoothing to this individual run
            history['smoothed_wis'] = history['wis'].ewm(alpha=alpha_param, adjust=False).mean()
       

        runs_data.append(history)
       

    # Combine all smoothed runs into a single DataFrame
    combined_runs = pd.concat(runs_data, axis=0)

    # Define from which column to calculate mean and std
    if smooth:
        column_to_consider = "smoothed_wis"
    else:
        column_to_consider = "wis"
    
    # Calculate mean and std 
    grouped = combined_runs.groupby('training_iters').agg({
        column_to_consider: ['mean', 'std']
    }).reset_index()
    
    

    # Flatten column names
    grouped.columns = ['training_iters', 'wis_mean', 'wis_std']

    # # Fill NaN values in std column with 0
    # grouped['wis_std'] = grouped['wis_std'].fillna(0)

    # Add group name
    grouped['group'] = group_name

    return grouped

# Main code
project_name = "taiskhakharova-brandenburgische-technische-universit-t-c/sepsis-BCQ"

# # With no training of encoder, random init only
# gnn1_run_ids = ['43nsp7vr', 'us51u36n', '5u9bviii', 'ukj3k2qk', '405k564x']
# gnn2_run_ids = ['j603d09k', 'tfnxpkfw', 'wltsavsm', '8cg1ryti', '21gwe9o5']
# ae_run_ids = ['fw0c56a5', '4mervybm', '4v4fccny', 'yjqoovg4', 'dwpca2qv']

# With training of encoder. 5e5

# gnn1_run_ids = ['o1tjq0dn', 'lisanu6b', 'm0eh839i']
# gnn2_run_ids = ['mf7kpoci', 't7gs87ul', '3qcq4ugz']
# ae_run_ids = ['7nqlbmr2', 'te756qy6', '0zkxyj8r']


# # With training of encoder. 1e6
gnn1_run_ids = ['8cw6v521', 'n8gkude9', 'f55f3c7l']
gnn2_run_ids = ['i1j5o9pq', 'uak6eeri', 'rq695mvn']


# Get data for each group
group1_data = get_averaged_runs_data(project_name, gnn1_run_ids, "GNN-SAGE", alpha_param=ALPHA, smooth=SMOOTH)
group2_data = get_averaged_runs_data(project_name, gnn2_run_ids, "GNN-GATv2Conv ", alpha_param=ALPHA, smooth=SMOOTH)
# group3_data = get_averaged_runs_data(project_name, ae_run_ids, "AE", alpha_param=ALPHA, smooth=SMOOTH)

# Combine all data
combined_df = pd.concat([group1_data, group2_data])
# combined_df = pd.concat([group1_data, group2_data,  group3_data])

# The plotting function doesn't need smoothing anymore since data is already smoothed
def create_loss_plot(combined_df, colors=['#2ecc71', '#e74c3c', '#3498db']):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set color palette
    sns.set_palette(colors)
    
    # Plot for each group
    for (group_name, group_data), color in zip(combined_df.groupby('group', sort=False), colors):
        # Convert to numeric if needed
        group_data['training_iters'] = pd.to_numeric(group_data['training_iters'])
        group_data['wis_mean'] = pd.to_numeric(group_data['wis_mean'])
        group_data['wis_std'] = pd.to_numeric(group_data['wis_std'])
        
        # Remove any NaN values
        group_data = group_data.dropna(subset=['training_iters', 'wis_mean', 'wis_std'])
        
        if len(group_data) > 0:

            # Calculate and smooth the corridor bounds

            # upper_bound = group_data['wis_mean'] + group_data['wis_std']
            # lower_bound = group_data['wis_mean'] - group_data['wis_std']

            

            # Smooth the bounds using the same smoothing as we used for the mean

            # upper_bound_smooth = upper_bound.ewm(alpha=alpha_param, adjust=False).mean()
            # lower_bound_smooth = lower_bound.ewm(alpha=alpha_param, adjust=False).mean()

            # Plot smoothed corridor to show standart deviation

            # ax.fill_between(group_data['training_iters'],

            #               lower_bound_smooth,
            #               upper_bound_smooth,
            #               alpha=0.2, color=color,
            #               label=f'{group_name} ±σ')
            
            # Plot mean line (already smoothed)
            ax.plot(group_data['training_iters'], group_data['wis_mean'], 
                    label=group_name, color=color, linewidth=2.5)
            
            # Plot standard deviation
            ax.fill_between(group_data['training_iters'],
                           group_data['wis_mean'] - group_data['wis_std'],
                           group_data['wis_mean'] + group_data['wis_std'],
                           alpha=0.2, color=color)
    
    # Customize the plot
    ax.set_xlabel('Training iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel('WIS Score', fontsize=14, fontweight='bold')
    ax.set_title('Average WIS Score Comparison', 
                 fontsize=16, 
                 fontweight='bold', 
                 pad=20)

    ax.grid(True, linestyle='--', alpha=0.7)
    legend = ax.legend(title='', 
                      bbox_to_anchor=(1.05, 1), 
                      loc='upper left',
                      frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    return fig

# colors = ['#7FB3D5' ,'#F5B7B1', '#A2D9CE']
colors = ['#7FB3D5' ,'#F5B7B1']

fig = create_loss_plot(combined_df, colors=colors)
if SMOOTH:
    filename = f"wis_score_comparison_smoothed_alpha_{ALPHA}.png"
else:
    filename = f"wis_score_comparison_no_smooth.png"
fig.savefig(filename, 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.show()