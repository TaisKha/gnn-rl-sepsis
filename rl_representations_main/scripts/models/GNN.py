import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .AbstractContainer import AbstractContainer
from .common import weights_init
from .common import pearson_correlation
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import Batch, HeteroData
from torch.utils.data import TensorDataset, DataLoader

from .graph_utils import split_trajectory_into_steps, create_trajectory_graph

# It works always with demographics and observations

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device          

    def generate_metadata(self, domain):
        if domain == "sepsis":
            node_types = ['patient', 'timestep']
            edge_types = [("patient", "has_timestep", "timestep"), ("timestep", "has_timestep", "patient"), ("timestep", "action", "timestep")]
            metadata = (node_types, edge_types)
        else:
            raise NotImplementedError("Only sepsis domain is supported")
        return metadata
    
    def make_encoder(self, hidden_size, domain="sepsis", encoder_hidden_size=128, encoder_num_layers=2):
        
        # self.gen = baseRNN_generate(hidden_size, state_dim, num_actions, context_input, context_dim).to(self.device)
        metadata = self.generate_metadata(domain=domain)
        self.gen  = SequenceGNNEncoder(hidden_channels=encoder_hidden_size, out_channels=hidden_size, num_layers=encoder_num_layers, metadata=metadata, device=self.device).to(self.device)
        
        return self.gen

    def make_decoder(self, hidden_size, state_dim, num_actions, inject_action):
        self.inject_action = inject_action
        self.pred = GNN_predict(h_size=hidden_size, obs_dim=state_dim, num_actions=num_actions, inject_action=self.inject_action).to(self.device)
        return self.pred   
    
    def transform_into_graph(self, current_obs, current_dem, current_actions):
        # need to understand first how in the original code of HTGNN the graph is created
        pass


    def loop(self, obs, dem, actions, scores, l, max_length, context_input, corr_coeff_param, device='cpu', **kwargs):
        '''This loop through the training and validation data is the general template for AIS, RNN, etc'''
        # Split the observations 
        # autoencoder = kwargs['autoencoder']
        
        # Transfer weight property from dem to obs, because weight changes over time

        # Remove the last column from dem
        demography_new = dem[:, :, :-1]  # Shape: (128, 20, 4)
        
        # Extract the last column from dem
        demography_last_column = dem[:, :, -1:]  # Shape: (128, 20, 1)

        # Append the last column to obs
        obs_new = torch.cat([obs, demography_last_column], dim=2)  # Shape: (128, 20, 33+1=34)

        obs = obs_new
        dem = demography_new
        
        cur_obs, next_obs = obs[:,:-1,:], obs[:,1:,:]
        # print(f"{cur_obs.shape=}") # (128, 19, 34)
        # print(f"{next_obs.shape=}") # (128, 19, 34)
        cur_dem = dem[:,:-1,:] 
        # print(f"{cur_dem.shape=}")# (128, 19, 4)
        
        
        # We need to cut the actions, too
        curr_actions = actions[:,:-1,:] 
        # And we need to subtract 1 from every length, because we cut one element from every trajectory in the minibatch
        curr_l = l - 1
        
        # cur_scores, next_scores = scores[:,:-1,:], scores[:,1:,:] # I won't need the "next scores"
        mask = (cur_obs ==0).all(dim=2) # Compute mask for extra appended rows of observations (all zeros along dim 2)
 
        sequence_size = cur_obs.shape[1] # number of steps in a trajectory
  

        data = (cur_dem, cur_obs, curr_actions, curr_l)
        batch_full_trajectory_graphs, batch_lengths = create_trajectory_graph(data)
        graphs_batch = split_trajectory_into_steps(batch_full_trajectory_graphs, batch_lengths)

       
        # Optionally, move data to device
        # This requires iterating and moving each HeteroData to the device
        for i in range(len(graphs_batch)):
            for j in range(sequence_size):
                graphs_batch[i][j].to(device)
                # for key in graphs_batch[i][j].x_dict.keys():
                #     graphs_batch[i][j].x_dict[key] = graphs_batch[i][j].x_dict[key].to(device)
                # for key in graphs_batch[i][j].edge_index_dict.keys():
                #     graphs_batch[i][j].edge_index_dict[key] = graphs_batch[i][j].edge_index_dict[key].to(device)
        
        # Encode the batch
        
        latent_representations = self.gen(graphs_batch)  # Shape: (batch_size, sequence_size, 128)

        if self.inject_action:
            pred_obs = self.pred(torch.cat((latent_representations, curr_actions), dim=-1))
        else: 
            pred_obs = self.pred(latent_representations)


        assert pred_obs.shape == next_obs.shape
        
        

        # Calculate the correlation between the hidden parameters and the acuity score (For now we'll use SOFA--idx 0)
        # corr_loss = pearson_correlation(hidden_states[~mask], scores[:,:-1,:][~mask], device=device)
        temp_loss = -torch.distributions.MultivariateNormal(pred_obs, torch.eye(pred_obs.shape[-1]).to(device)).log_prob(next_obs)
        mse_loss = sum(temp_loss[~mask])
        # loss_pred = mse_loss - corr_coeff_param*corr_loss.sum() # We only want to keep the relevant rows of the loss, sum them up! We then add the scaled correlation coefficient
        # We do not care that the latent representations correlate with acuity scores
        loss_pred = mse_loss

        return loss_pred, mse_loss, latent_representations
    
    

class HeteroGNNEncoder(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=128, num_layers=2, metadata=None):
        """
        Initializes the Heterogeneous GNN Encoder.

        Args:
            hidden_channels (int): Number of hidden units in GNN layers.
            out_channels (int): Dimension of the output latent vector.
            num_layers (int): Number of GNN layers.
            metadata (tuple): Metadata for HeteroConv, typically (node_types, edge_types).
        """
        super(HeteroGNNEncoder, self).__init__()
        
        if metadata is None:
            raise ValueError("Metadata must be provided for HeteroConv.")
        
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Define HeteroConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src, rel, dst = edge_type
                conv_dict[edge_type] = SAGEConv(-1, hidden_channels, aggr='mean')
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)
        
        # Linear layer to project to latent space
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, data):
        """
        Forward pass of the encoder.

        Args:
            data (data): A data of HeteroData graphs.

        Returns:
            Tensor: Latent representations of shape (num_graphs, out_channels).
        """
        x_dict = data.x_dict  # Dict of node_type -> node_features

        # print("x_dict", x_dict["author"].shape)
        # print()
        # in the beginning author shape was (200,32),
        # because we have 10 author nodes per graph, and we data size 4 and sequence size 5, so 10*4*5=200
        edge_index_dict = data.edge_index_dict  # Dict of edge_type -> edge_index
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)  # Perform HeteroConv
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}  # Apply activation

        # after covilution for every node type shape should be (200,64)
        
       
       
        
        # for every node type we take a mean of all of the nodes from one graph
        x_dict = {key: global_mean_pool(x_dict[key], data[key].batch) for key in data.node_types}
        # print("shape after pooling for author")
        # print(x_dict["author"].shape) #(20, 64) so for every graph - 64 vector 
        # print("shape after pooling for paper")
        # print(x_dict["paper"].shape) #(20, 64) so for every graph - 64 vector
        # print("shape after pooling for institution")
        # print(x_dict["institution"].shape) #(20, 64) so for every graph - 64 vector
        # # 
        # print("AAAAAA")
        # print(x_dict.values())

        # here we connect along the 0 dimension, creating extra dimension
        # `stack` creates extra dimension, while `cat` connects along a certain dim
        # after stack we get [3, 20, 64], which is [node_types_num, num_of_graphs, hidden_size]
        # then we are summing up every element along the node_type dimension. So, sum up all the node types.
        # we should get again [20, 64] shape. For every graph, we get 64 size vector
        graph_emb = torch.stack(list(x_dict.values()), dim=0).sum(dim=0)
 
        
        # Project to desired latent dimension
        out = self.linear(graph_emb)  # Shape: (num_graphs, out_channels)
        
        return out

class SequenceGNNEncoder(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=128, num_layers=2, metadata=None, device=None):
        
        """
        Initializes the Sequence GNN Encoder.

        Args:
            hidden_channels (int): Number of hidden units in GNN layers.
            out_channels (int): Dimension of the output latent vector.
            num_layers (int): Number of GNN layers.
            metadata (tuple): Metadata for HeteroConv, typically (node_types, edge_types).
        """
        super(SequenceGNNEncoder, self).__init__()
        self.device = device
        self.encoder = HeteroGNNEncoder(hidden_channels, out_channels, num_layers, metadata).to(self.device)
        
    def forward(self, graphs_batch):
        """
        Forward pass for a batch of graph sequences.

        Args:
            graphs_batch (list of list of HeteroData): 
                Outer list has length batch_size.
                Each inner list has length sequence_size, containing HeteroData graphs.

        Returns:
            Tensor: Latent representations of shape (batch_size, sequence_size, out_channels).
        """
        batch_size = len(graphs_batch)
        sequence_size = len(graphs_batch[0])
        
        # Flatten the list of lists into a single list
        all_graphs = [graph for batch in graphs_batch for graph in batch]
        
        # Create a Batch object from the flattened list
        batch = Batch.from_data_list(all_graphs).to(self.device)
        
        # Encode all graphs
        encoded = self.encoder(batch)  # Shape: (batch_size * sequence_size, out_channels)
        
        # Reshape to (batch_size, sequence_size, out_channels)
        encoded = encoded.view(batch_size, sequence_size, -1)
        
        return encoded
    
class GNN_predict(nn.Module):
    def __init__(self, h_size, obs_dim, num_actions, inject_action=False):
        super(GNN_predict,self).__init__()
        if inject_action:
            self.l1 = nn.Linear(h_size+num_actions,64)
        else:
            self.l1 = nn.Linear(h_size, 64)

        self.l2 = nn.Linear(64,128)
        self.l3 = nn.Linear(128,obs_dim)
        self.apply(weights_init)
    def forward(self,h):
        h = torch.relu(self.l1(h))
        h = torch.relu(self.l2(h))
        obs = self.l3(h)
        return obs
    
