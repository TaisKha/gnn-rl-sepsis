# Transfer functions from load_from_tuples_to_pyg.ipynb

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.patches as mpatches
import seaborn as sns
import random
import string

def create_dummy_graph(reference_graph):

    assert isinstance(reference_graph, HeteroData)
    # print(f"{reference_graph=}")

    data = HeteroData()

    node_types, edge_types = reference_graph.node_types, reference_graph.edge_types

    for node_type in node_types:
        data[node_type].x = torch.zeros(reference_graph[node_type].x.shape)

    for edge_type in edge_types:
        
        data[edge_type].edge_index = reference_graph[edge_type].edge_index
        
        if hasattr(reference_graph[edge_type], 'edge_attr'):
            shape_of_edge_attr = reference_graph[edge_type].edge_attr.shape
            # print(f"{edge_type=}")
            # print(f"{shape_of_edge_attr=}")
            
            data[edge_type].edge_attr = torch.zeros(reference_graph[edge_type].edge_attr.shape)
    # print(f"{data=}")   
    # print("Dummy graph created")
    return data


def split_trajectory_into_steps(batch_full_trajectory_graphs, batch_lengths):

    # DO WE NEED TERMINAL STATE AT ALL? NO

    assert len(batch_full_trajectory_graphs) == len(batch_lengths)
    batch = []
    max_length = max(batch_lengths)
    sequence_length = max_length
    # Works with one minibatch of graphs. Each graph is an instance of HeteroData(). batch_full_trajectory_graphs is a list().
    for full_trajectory_graph, n_timesteps in zip(batch_full_trajectory_graphs, batch_lengths):

        assert isinstance(full_trajectory_graph, HeteroData)
        
        time_sequence = []
        
        for t_idx in range(n_timesteps):

            graph = HeteroData()
            # Node features
            graph['patient'].x = full_trajectory_graph['patient'].x
            graph['timestep'].x = full_trajectory_graph['timestep'].x[:t_idx+1]
            

            # Edges
            graph["patient", "has_timestep", "timestep"].edge_index = full_trajectory_graph["patient", "has_timestep", "timestep"].edge_index[:, :t_idx+1]
            
            graph["timestep", "has_timestep", "patient"].edge_index = full_trajectory_graph["timestep", "has_timestep", "patient"].edge_index[:, :t_idx+1]
            
            graph["timestep", "action", "timestep"].edge_index = full_trajectory_graph["timestep", "action", "timestep"].edge_index[:, :t_idx] # We do not need the action that is outgoing from the t_idx node

            # Edge features
            graph["patient", "has_timestep", "timestep"].edge_attr = full_trajectory_graph["patient", "has_timestep", "timestep"].edge_attr[:t_idx+1]

            graph["timestep", "has_timestep", "patient"].edge_attr = full_trajectory_graph["timestep", "has_timestep", "patient"].edge_attr[:t_idx+1]
            
            graph["timestep", "action", "timestep"].edge_attr = full_trajectory_graph["timestep", "action", "timestep"].edge_attr[:t_idx]

            
            # print(graph)
            # draw_graph(graph, save_to_file=False, display=True)
            

            time_sequence.append(graph)
            
        # Padding with dummy graphs to reach sequence_length

        for k_ids in range(n_timesteps, sequence_length):

            dummy_graph = create_dummy_graph(time_sequence[-1])
            time_sequence.append(dummy_graph)
            
        batch.append(time_sequence)

    return batch
            

# Function to generate a random alphanumeric string
def generate_random_string(length=6):
    """Generate a random string of fixed length."""
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choices(letters_and_digits, k=length))
    
def draw_graph(data, save_to_file=True, display=False):
    # Vizualisation function of a HeteroData() graph using networkx libarary.
    # The method saves the graph as a PNG and/or prints it on the screen
    
    if not save_to_file and not display:
        raise Exception("Parameters `save_to_file` and `display` are both set to False. Set one of them to True.")

    
    # Convert PyG HeteroData to NetworkX graph
    graph = to_networkx(data, to_undirected=False)
    
    # Define an aesthetic color palette using ColorBrewer and Seaborn
    node_type_colors = {
        "patient": "#1b9e77",    # Teal
        "timestep": "#d95f02",   # Vermillion
        "terminal": "#7570b3",   # Purple
    }
    
    edge_type_colors = {
        ("patient", "has_timestep", "timestep"): "#e7298a",     # Pink
        ("timestep", "action", "timestep"): "#66a61e",        # Green
        ("timestep", "terminates_at", "terminal"): "#e6ab02", # Mustard
    }
    
    # Initialize lists for node colors and labels
    node_colors = []
    labels = {}
    
    # Assign colors and labels based on node types
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        color = node_type_colors.get(node_type, "#999999")  # Default to gray if type unknown
        node_colors.append(color)
        
        # Assign labels based on node type
        if node_type == "patient":
            labels[node] = f"P{node}"  # e.g., P1, P2
        elif node_type == "timestep":
            labels[node] = f"T{node}"  # e.g., T1, T2
        elif node_type == "terminal":
            labels[node] = "Terminal"
        else:
            labels[node] = f"U{node}"  # Unknown
    
    # Initialize list for edge colors
    edge_colors = []
    
    # Assign colors based on edge types
    for u, v, attrs in graph.edges(data=True):
        # Edge type is a tuple: (source_node_type, relation_type, target_node_type)
        edge_type = tuple(attrs.get("type", ("unknown_src", "unknown_rel", "unknown_tgt")))
        
        # Retrieve color for the edge type
        color = edge_type_colors.get(edge_type, "#999999")  # Default to gray if type unknown
        
        # Assign the color to the edge in the graph
        graph.edges[u, v]["color"] = color
        edge_colors.append(color)
    
    # Generate positions for nodes using a layout algorithm
    pos = nx.spring_layout(graph, k=0.5, seed=42)  # Adjust 'k' and 'seed' for better layout
    
    # Create the plot
    plt.figure(figsize=(14, 10))  # Increased figure size for better clarity
    
    # Draw the graph with specified node and edge colors
    nx.draw_networkx(
        graph,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=800,
        arrows=True,            # Show directionality
        arrowstyle='-|>',
        arrowsize=20,           # Arrow size
        linewidths=1,
        font_size=12,
        font_color='black',
        edge_cmap=plt.cm.Blues  # Optional: Add a colormap for edges
    )
    
    # Create legend patches for node types
    node_patches = [mpatches.Patch(color=color, label=node_type.capitalize()) 
                   for node_type, color in node_type_colors.items()]
    
    # Create legend patches for edge types
    edge_patches = [mpatches.Patch(color=color, label='_'.join(edge_type)) 
                   for edge_type, color in edge_type_colors.items()]
    
    # Add legends to the plot
    first_legend = plt.legend(handles=node_patches, title="Node Types", loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().add_artist(first_legend)  # Add first legend manually
    plt.legend(handles=edge_patches, title="Edge Types", loc='upper left', bbox_to_anchor=(1, 0.6))
    
    # Remove axis for better visualization
    plt.axis('off')
    plt.title("HeteroData Graph Visualization", fontsize=18, fontweight='bold')
    
    # Adjust layout to make room for legends
    plt.tight_layout()

    if save_to_file:
    
        # Generate a random string for the filename
        random_str = generate_random_string(length=6)  # Generates a string like 'irunvQ'
        
        # Create the filename with the random string
        filename = f"heterodata_graph_{random_str}.png"
        
        # Save the plot as a PNG file with the random string in the filename
        plt.savefig(filename, format="PNG", dpi=300, bbox_inches='tight')

        # Print the filename for confirmation
        print(f"Graph has been saved as {filename}")
    if display:
        # Display the plot
        plt.show()
    


def create_trajectory_graph(data):
    # Important! I will add the weight from demographics features to the timestep nodes, because it changes
    assert torch.is_tensor(data[0])==True
    assert data[0].dim() == 3
    
    
    demography, observations, actions, l, t, scores, rewards, idx = data
    minibatch_size = len(observations)

    
    # Remove the last column from dem
    demography_new = demography[:, :, :-1]  # Shape: (128, 20, 4)
    
    # Extract the last column from dem
    demography_last_column = demography[:, :, -1:]  # Shape: (128, 20, 1)

    # Append the last column to obs
    observations_new = torch.cat([observations, demography_last_column], dim=2)  # Shape: (128, 20, 33+1=34)
    
    batch_trajectory_graphs = []
    batch_lengths = []

    
    
    for traj_i in range(minibatch_size): 

        # Initialize a new graph
        graph = HeteroData()
        n_timesteps = l[traj_i].item()
        assert isinstance(n_timesteps, int)
        
        
        curr_demography = demography_new[traj_i][0] # We take only the first element, because demography info does not chage over time
        current_observations = observations_new[traj_i]
        current_actions = actions[traj_i]

        assert current_observations.dim() == 2
        assert current_actions.dim() == 2

        # Cut the padding that was applied before
        current_timesteps = current_observations[:n_timesteps, :]
        patient_data = curr_demography.unsqueeze(0) # Shape should be (num patients, num features). We have only one patient in the graph.
        current_actions = current_actions[:n_timesteps, :]

        # --Adding nodes info--
        graph['patient'].x = patient_data
        graph['timestep'].x = current_timesteps
        graph['terminal'].x = torch.zeros(1, current_timesteps.shape[-1])

        # --Adding edges info--
        
        # Add edges between patient and timestep
        patient_to_timestep_edge_index = torch.tensor([
            [0] * n_timesteps,  # Patient node repeated `num_timesteps` times (source nodes)
            list(range(n_timesteps))  # Timestep node indices (target nodes) - start from 0
        ])

        timestep_to_patient_edge_index = torch.stack(
            (patient_to_timestep_edge_index[1], patient_to_timestep_edge_index[0]),
            dim=0
        )

        assert patient_to_timestep_edge_index.shape == timestep_to_patient_edge_index.shape

        graph["patient", "has_timestep", "timestep"].edge_index = patient_to_timestep_edge_index
        graph["timestep", "has_timestep", "patient"].edge_index = timestep_to_patient_edge_index
        

        # Add edges between timesteps. 
        timestep_to_timestep_edge_index = torch.tensor([
            list(range(n_timesteps-1)),  # source timestep nodes
            list(range(1, n_timesteps))  # target timestep nodes
        ])
        
        graph["timestep", "action", "timestep"].edge_index = timestep_to_timestep_edge_index
        
        # Connect last timestep node with the terminal
        graph["timestep", "terminates_at", "terminal"].edge_index = torch.tensor([
            [n_timesteps-1], # Source - the very last timestep node
            [0] # Target - terminal node, it is only one, therefore index is 0
        ])

        # --Adding values to edges--

        # Add dummy weight to patient-to-timestep nodes. Without doing that it is not possible to use .to_hetero()

        graph["patient", "has_timestep", "timestep"].edge_attr = torch.tensor([1] * n_timesteps).view(n_timesteps, 1)

        graph["timestep", "has_timestep", "patient"].edge_attr = torch.tensor([1] * n_timesteps).view(n_timesteps, 1)

        # Add last action leading from the last timestep node to the ternimal node
        graph["timestep", "terminates_at", "terminal"].edge_attr = current_actions[-1]

        # Add property(action) to the edges between timesteps
        graph["timestep", "action", "timestep"].edge_attr = current_actions[:-1]
        
        
        # draw_graph(graph, save_to_file=False, display=False)

        batch_trajectory_graphs.append(graph)
        batch_lengths.append(n_timesteps)
        
    
        
    return batch_trajectory_graphs, batch_lengths



# train_demog, train_states, train_interventions, train_lengths, train_times, acuities, rewards = torch.load(train_data_file)
# train_idx = torch.arange(train_demog.shape[0])
# train_dataset = TensorDataset(train_demog, train_states, train_interventions,train_lengths,train_times, acuities, rewards, train_idx)
# train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# for ii, (dem, ob, ac, l, t, scores, rewards, idx) in enumerate(train_loader):
#                 if ii > 0:
#                     break
#                 # print("Batch {}".format(ii),end='')
    
#                 # we've got 128 different trajectories of different length
    
#                 dem = dem.to(device)  # 5 dimensional vector (Gender, Ventilation status, Re-admission status, Age, Weight)
#                 ob = ob.to(device)    # 33 dimensional vector (time varying measures)
#                 ac = ac.to(device) # actions
#                 l = l.to(device)
#                 t = t.to(device)
#                 scores = scores.to(device)
#                 idx = idx.to(device)
#                 loss_pred = 0

#                 # Cut tensors down to the batch's largest sequence length... Trying to speed things up a bit...
#                 max_length = int(l.max().item())
#                 min_length = int(l.min().item())
#                 # print(f"{max_length=}")
#                 # print(f"{min_length=}")
#                 # # The following losses are for DDM and will not be modified by any other approach
#                 # train_loss, dec_loss, inv_loss = 0, 0, 0
#                 # model_loss, recon_loss, forward_loss = 0, 0, 0                    
                    
#                 # # Set training mode (nn.Module.train()). It does not actually trains the model, but just sets the model to training mode.
#                 # self.gen.train()
#                 # self.pred.train()
                
#                 ob = ob[:,:max_length,:]
#                 dem = dem[:,:max_length,:]
#                 ac = ac[:,:max_length,:]
#                 scores = scores[:,:max_length,:]

#                 data = (dem, ob, ac, l, t, scores, rewards, idx)
#                 batch_full_trajectory_graphs, batch_lengths = create_trajectory_graph(data)
#                 final_batch = split_trajectory_into_steps(batch_full_trajectory_graphs, batch_lengths)
    
#                 # print(dem[0])
#                 # print(l[0])
#                 # print(ac[0][:-1].shape)
#                 # print(ob.shape)
#                 # print(ac.shape)
                
                

#                 # trajectory_graph = create_trajectory_graph(dem, ob, ac, l, t, scores, rewards, idx)
