import yaml
import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils import ReplayBuffer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

class Buffer(object):
  
  def __init__(self, hidden_size, state_dim, context_dim, train_data_file, validation_data_file, minibatch_size, device, dem_context, train_buffer_file, val_buffer_file):
        
        self.hidden_size = hidden_size
        self.minibatch_size = minibatch_size
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.train_data_file = train_data_file
        self.validation_data_file = validation_data_file
        self.context_input = dem_context
        self.train_buffer_file = train_buffer_file
        self.val_buffer_file = val_buffer_file

        if device == 'cuda':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif device == 'cpu':
            self.device = torch.device('cpu')
        else:
            print("Please set device to 'cuda' or 'cpu'")
            exit(1)
        print('Using device:', self.device)
            

  def create_buffers_and_save(self):
       

        replay_buffer_train = ReplayBuffer(self.hidden_size, self.minibatch_size, 200000, self.device, encoded_state=True, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))
        replay_buffer_val = ReplayBuffer(self.hidden_size, self.minibatch_size, 50000, self.device, encoded_state=True, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))
        
        self.train_demog, self.train_states, self.train_interventions, self.train_lengths, self.train_times, self.acuities, self.rewards = torch.load(self.train_data_file)
        train_idx = torch.arange(self.train_demog.shape[0])
        self.train_dataset = TensorDataset(self.train_demog, self.train_states, self.train_interventions,self.train_lengths,self.train_times, self.acuities, self.rewards, train_idx)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.minibatch_size, shuffle=True)

        self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards = torch.load(self.validation_data_file)
        val_idx = torch.arange(self.val_demog.shape[0])
        self.val_dataset = TensorDataset(self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards, val_idx)

        self.val_loader = DataLoader(self.val_dataset, batch_size=self.minibatch_size, shuffle=False)

        # self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_acuities, self.test_rewards = torch.load(self.test_data_file)
        # test_idx = torch.arange(self.test_demog.shape[0])
        # self.test_dataset = TensorDataset(self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_acuities, self.test_rewards, test_idx)

        # self.test_loader = DataLoader(self.test_dataset, batch_size=self.minibatch_size, shuffle=False)

        # all_loaders = [self.train_loader, self.val_loader, self.test_loader]

        all_loaders_list = [self.train_loader, self.val_loader]
        all_buffers = [replay_buffer_train, replay_buffer_val]
        all_buffers_save_files = [self.train_buffer_file, self.val_buffer_file]

        for i_set, loader in enumerate(all_loaders_list):
            print("Creating buffer for set ", i_set)
            replay_buffer = all_buffers[i_set]
            buffer_save_file = all_buffers_save_files[i_set]

            for dem, ob, ac, l, t, scores, rewards, idx in loader:
                
                dem = dem.to(self.device)
                ob = ob.to(self.device)
                ac = ac.to(self.device)
                l = l.to(self.device)
                t = t.to(self.device)
                scores = scores.to(self.device)
                rewards = rewards.to(self.device)

                max_length = int(l.max().item())

                ob = ob[:,:max_length,:]
                dem = dem[:,:max_length,:]
                ac = ac[:,:max_length,:]
                scores = scores[:,:max_length,:]
                rewards = rewards[:,:max_length]

                cur_obs, next_obs = ob[:,:-1,:], ob[:,1:,:]
                cur_dem, next_dem = dem[:,:-1,:], dem[:,1:,:]
                cur_actions = ac[:,:-1,:]
                cur_rewards = rewards[:,:-1]
                cur_scores = scores[:,:-1,:]
                mask = (cur_obs==0).all(dim=2)

                representations = torch.cat(cur_obs, cur_dem)

                # Remove values with the computed mask and add data to the experience replay buffer
                cur_rep = torch.cat((representations[:,:-1, :], torch.zeros((cur_obs.shape[0], 1, self.hidden_size)).to(self.device)), dim=1)
                next_rep = torch.cat((representations[:,1:, :], torch.zeros((cur_obs.shape[0], 1, self.hidden_size)).to(self.device)), dim=1)
                cur_rep = cur_rep[~mask].cpu()
                next_rep = next_rep[~mask].cpu()
                cur_actions = cur_actions[~mask].cpu()
                cur_rewards = cur_rewards[~mask].cpu()
                cur_obs = cur_obs[~mask].cpu()  # Need to keep track of the actual observations that were made to form the corresponding representations (for downstream WIS)
                next_obs = next_obs[~mask].cpu()
                cur_dem = cur_dem[~mask].cpu()
                next_dem = next_dem[~mask].cpu()
                
                
                # Loop over all transitions and add them to the replay buffer
                for i_trans in range(cur_rep.shape[0]):
                    done = cur_rewards[i_trans] != 0
                    if self.context_input:
                        self.replay_buffer.add(cur_rep[i_trans].numpy(), cur_actions[i_trans].argmax().item(), next_rep[i_trans].numpy(), cur_rewards[i_trans].item(), done.item(), torch.cat((cur_obs[i_trans],cur_dem[i_trans]),dim=-1).numpy(), torch.cat((next_obs[i_trans], next_dem[i_trans]), dim=-1).numpy())
                    else:
                        self.replay_buffer.add(cur_rep[i_trans].numpy(), cur_actions[i_trans].argmax().item(), next_rep[i_trans].numpy(), cur_rewards[i_trans].item(), done.item(), cur_obs[i_trans].numpy(), next_obs[i_trans].numpy())


       
            ## SAVE OFF DATA
            # --------------
            print("Saving buffer to ", buffer_save_file)
            replay_buffer.save(buffer_save_file)
           



if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    behavCloning_params = yaml.safe_load(open(os.path.join(dir_path, '../configs/config_behavCloning.yaml'), 'r'))
    common_params = yaml.safe_load(open(os.path.join(dir_path, '../configs/common.yaml'), 'r')) 

 

    buffer = Buffer(
        hidden_size = behavCloning_params['hidden_size'],
        minibatch_size = common_params['minibatch_size'],
        state_dim = common_params['state_dim'],
        context_dim = common_params['context_dim'],
        train_data_file = common_params['train_data_file'],
        validation_data_file = common_params['validation_data_file'],
        dem_context = common_params['dem_context'],
        train_buffer_file = behavCloning_params['train_buffer_file'],
        val_buffer_file = behavCloning_params['val_buffer_file'],

        device = 'cuda'
    )
    buffer.create_buffers_and_save()

