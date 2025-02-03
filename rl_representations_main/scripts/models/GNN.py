import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .AbstractContainer import AbstractContainer
from .common import weights_init
from .common import pearson_correlation

# It works always with demographics and observations

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device          
    
    def make_encoder(self, hidden_size, state_dim, num_actions, context_input, context_dim):
        pass
        # self.gen = baseRNN_generate(hidden_size, state_dim, num_actions, context_input, context_dim).to(self.device)
        # return self.gen

    def make_decoder(self, hidden_size, state_dim, num_actions):
        self.pred = GNN_predict(hidden_size, state_dim, num_actions).to(self.device)
        return self.pred   
    
    def transform_into_graph(self, current_obs, current_dem, current_actions):
        # need to understand first how in the original code of HTGNN the graph is created
        pass
    
    def loop(self, obs, dem, actions, scores, l, max_length, context_input, corr_coeff_param, device='cpu', **kwargs):

        assert context_input == True, "Context input must be True"


        '''This loop through the training and validation data is the general template for AIS, RNN, etc'''
        # Split the observations 
        autoencoder = kwargs['autoencoder'] 
        cur_obs, next_obs = obs[:,:-1,:], obs[:,1:,:]
        cur_dem = dem[:,:-1,:]
        # cur_scores, next_scores = scores[:,:-1,:], scores[:,1:,:] # I won't need the "next scores"
        mask = (cur_obs ==0).all(dim=2) # Compute mask for extra appended rows of observations (all zeros along dim 2)

        # Transform relational data into a graph
        graph_input = transform_into_graph(cur_obs, cur_dem, actions)

        # This concatenates an empty action with the first observation and shifts all actions 
        # to the next observation since we're interested in pairing obs with previous action
        if context_input:

            # hidden_states = self.gen(torch.cat((cur_obs, cur_dem, torch.cat((torch.zeros((obs.shape[0],1,actions.shape[-1])).to(device),actions[:,:-2,:]),dim=1)),dim=-1))
            hidden_states = self.gen(graph_input)
        else:
            raise NotImplementedError("Context input must be True")
            # hidden_states = self.gen(torch.cat((cur_obs, torch.cat((torch.zeros((obs.shape[0],1,actions.shape[-1])).to(device), actions[:,:-2,:]),dim=1)), dim=-1))

        if autoencoder == 'RNN':
            pred_obs = self.pred(hidden_states)
        else:
            pred_obs = self.pred(torch.cat((hidden_states,actions[:,:-1,:]),dim=-1))

        # Calculate the correlation between the hidden parameters and the acuity score (For now we'll use SOFA--idx 0)
        corr_loss = pearson_correlation(hidden_states[~mask], scores[:,:-1,:][~mask], device=device)
        temp_loss = -torch.distributions.MultivariateNormal(pred_obs, torch.eye(pred_obs.shape[-1]).to(device)).log_prob(next_obs)
        mse_loss = sum(temp_loss[~mask])
        loss_pred = mse_loss - corr_coeff_param*corr_loss.sum() # We only want to keep the relevant rows of the loss, sum them up! We then add the scaled correlation coefficient

        return loss_pred, mse_loss, hidden_states
    
    def transform_into_graph(self, current_obs, current_dem, current_actions):
        # need to understand first how in the original code of HTGNN the graph is created
        pass

# class baseRNN_generate(nn.Module):
#     def __init__(self, h_size, obs_dim, num_actions, context_input=False, context_dim=0):
#         super(baseRNN_generate,self).__init__()
#         if context_input:
#             self.l1 = nn.Linear(obs_dim + context_dim + num_actions, 64)
#         else:
#             self.l1 = nn.Linear(obs_dim + num_actions,64)
#         self.l2 = nn.Linear(64,128)
#         self.l3 = nn.GRU(128,h_size)
#         self.apply(weights_init)
#     def forward(self,x):
#         x = torch.relu(self.l1(x))
#         x = torch.relu(self.l2(x))
#         x = x.permute(1,0,2)
#         h, _ = self.l3(x)
#         return h.permute(1,0,2)

class GNN_predict(nn.Module):
    def __init__(self, h_size, obs_dim, num_actions, context_input=False, append_action=False):
        super(GNN_predict,self).__init__()
        if append_action:
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
    
    