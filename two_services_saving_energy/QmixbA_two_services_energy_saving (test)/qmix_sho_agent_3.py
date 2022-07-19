import torch
import numpy as np
import random
from collections import deque
import itertools

from qmix_sho_qmixer_2 import *
from qmix_sho_configuration import *
# from  qmix_sho_agent_light import *
from qmix_sho_agent_temp_3 import *
from qmix_sho_agent_air_3 import *
# from qmix_sho_agent_common import *

import copy


use_cuda=torch.cuda.is_available()
device=torch.device('cpu')

class EpsilonGreedy:
    def __init__(self, final_step=288,
        epsilon_start=float(1), epsilon_end=0.05):

        self.epsilon=epsilon_start
        self.initial_epsilon=epsilon_start
        self.epsilon_end=epsilon_end
        # self.action_nb=action_nb
        self.final_step=final_step
        self.agent_nb=1

    def act(self, value_action, avail_action):
        
        action_idx=None
        
        
        # print(f'avail_action: {avail_action}')
        
        value_action2=copy.deepcopy(value_action.detach())
        value_action2[value_action<=0]=1e-6

        if np.random.random()>self.epsilon:
            
            # print(f'value_action: {value_action2}')

            action_idx=value_action.max(dim=1)[1]

        else:
            
            # print(f'value_action: {value_action2}')
            
            
            action_idx=torch.distributions.Categorical(value_action2).sample().long().cpu().detach().numpy()

        action=avail_action[:,int(action_idx.item())].reshape(-1,1)

        return action


    def epsilon_decay(self, step):

        progress=step/self.final_step

        decay=self.initial_epsilon-progress

        if decay<=self.epsilon_end:

            decay=self.epsilon_end

        self.epsilon=decay


class EpisodeBatch:

    def __init__(self,random_seed=123):
        pass

    def reset(self):
        pass

    def add(self, replay_buffer):
        if self.count < self.buffer_size: 
            self.buffer.append(replay_buffer)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(replay_buffer)

    def _get_max_episode_len(self, batch):
        max_episode_len = 0

        for replay_buffer in batch:
            _, _, _, t, _, _, _ = replay_buffer.sample_batch(replay_buffer.size())
            for idx, t_idx in enumerate(t):
                if t_idx == True:
                    if idx > max_episode_len:
                        max_episode_len = idx + 1
                    break
                    
        return max_episode_len
    
    
    def size(self):
        return self.count


class QMix:

    def __init__(self,training=True, lr=0.1, gamma=0.99, batch_size=16, replayMemorySize=1e6, update_target_network=5,final_step=5000,minibatchSize=288):

        # agent_nb: the number of agents (in each batch?)
        self.training=training
        self.gamma=gamma
        self.batch_size=batch_size
        self.update_target_network=update_target_network
        self.hidden_states_light=None
        self.hidden_states_temp=None
        self.hidden_states_air=None
        self.hidden_states_common=None

        self.target_hidden_states_light=None
        self.target_hidden_states_temp=None
        self.target_hidden_states_air=None
        self.target_hidden_states_common=None
        self.agent_nb=1
        self.replayMemorySize=int(replayMemorySize)
       
        self.minibatchSize=minibatchSize

        self.replayMemory=deque(maxlen=self.replayMemorySize)

        self.epsilon_greedy=EpsilonGreedy()

        self.episode_batch=EpisodeBatch()

        self.qmixer=QMixer(n_agents=1,state_shape=X_US.shape[1]+1+1+1+1,mixing_embed_dim=32).to(device)

        self.target_qmixer=QMixer(n_agents=1,state_shape=X_US.shape[1]+1+1+1+1,mixing_embed_dim=32).to(device)

        self.target_qmixer.update(self.qmixer)

        # self.light_service=LightService_cell()

        self.temp_service=TempService_cell()

        self.air_service=AirService_cell()

        # self.common_service=CommonService_cell()

        # concatenate the parameters

        self.params=self.temp_service.params+self.air_service.params+list(self.qmixer.parameters())

        self.optimizer = torch.optim.RMSprop(params=self.params, lr=lr, alpha=0.99, eps=0.00001)

    def updateReplayMemory(self,transition):
        self.replayMemory.append(transition)

    def save_model(self, path):
        # torch.save(self.light_service.agents.state_dict(), path+'/qmix_light_model.pth')
        torch.save(self.temp_service.agents.state_dict(), path+'/qmix_temp_model.pth')
        torch.save(self.air_service.agents.state_dict(), path+'/qmix_air_model.pth')
        # torch.save(self.common_service.agents.state_dict(), path+'/qmixer_common_model.pth')
        torch.save(self.qmixer.agents.state_dict(), path+'/qmixer_model.pth')

    def load_model(self, path):

        # self.light_service.agents.load_state_dict(torch.load(path+'/qmix_light_model.pth'))
        self.temp_service.agents.load_state_dict(torch.load(path+'/qmix_temp_model.pth'))
        self.air_service.agents.load_state_dict(torch.load((path+'/qmix_air_model.pth')))
        # self.common_service.agents.load_state_dict(torch.load(path+'/qmixer_common_model.pth'))
        self.qmixer.load_state_dict(torch.load(path+'/qmixer_model.pth'))

    def _init_hidden_states(self, batch_size):
        # self.hidden_states_light=self.light_service.agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)
        self.hidden_states_temp=self.temp_service.agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)
        self.hidden_states_air=self.air_service.agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)
        # self.hidden_states_common=self.common_service.agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)

        # self.target_hidden_states_light=self.light_service.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)
        self.target_hidden_states_temp=self.temp_service.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)
        self.target_hidden_states_air=self.air_service.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)
        # self.target_hidden_states_common=self.common_service.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)

    def decay_epsilon_greddy(self, step):
        self.epsilon_greedy.epsilon_decay(step)

    def on_reset(self, batch_size):
        self._init_hidden_states(batch_size)

    def update_targets(self, epoch):

        if epoch % self.update_target_network == 0 and self.training:

            # self.light_service.target_agents.update(self.light_service.agents)
            self.temp_service.target_agents.update(self.temp_service.agents)
            self.air_service.target_agents.update(self.air_service.agents)
            # self.common_service.target_agents.update(self.common_service.agents)

            self.target_qmixer.update(self.qmixer)
            

    def train(self):

        if len(self.replayMemory) % 288==0 and len(self.replayMemory)>0:
            pass

        else:

            return 
        
        self._init_hidden_states(1)

        minibatch=deque(itertools.islice(self.replayMemory, len(self.replayMemory) - 288, len(self.replayMemory)))


        states_list_tt=[transition[0] for transition in minibatch]

        # %% reward

        rewards=[transition[2] for transition in minibatch]

        r_batch=None
        for i in range(len(rewards)):
            if i==0:
                # ensure that rewards[i] is a tensor
                r_batch=rewards[i]
            else:
                r_batch_=rewards[i]
                r_batch=torch.cat((r_batch,r_batch_),0)

        r_batch=r_batch.reshape((self.minibatchSize,-1))

        # %% terminal mask

        dones=[transition[-1] for transition in minibatch]

        t_batch=None
        for i in range(len(dones)):
            if i==0:
                t_batch=dones[i]

            else:
                t_batch_=dones[i]
                t_batch=torch.cat((t_batch,t_batch_),0)

        t_batch=t_batch.reshape((self.minibatchSize,-1))


        # actions=[transition[1] for transition in minibatch]

        # temp_actions_history=None
        # air_action_history=None

        # for i in range(len(actions)):

        #     if i==0:

        #         ations_t_minus_one=actions[i]

        #         x_cur_t_minus_one,x_ac_t_minus_one,x_tt_t_minus_one,x_ap_t_minus_one,x_at_t_minus_one,x_win_t_minus_one,x_et_t_minus_one=ations_t_minus_one

        #         temp_actions_minus_one=torch.cat((x_cur_t_minus_one,x_ac_t_minus_one,x_win_t_minus_one,x_tt_t_minus_one,x_et_t_minus_one),1)
        #         air_actions_minus_one=torch.cat((x_cur_t_minus_one,x_win_t_minus_one,x_ap_t_minus_one, x_at_t_minus_one, x_et_t_minus_one),1)

        #         temp_actions_history=copy.deepcopy(temp_actions_minus_one)
        #         air_action_history=copy.deepcopy(air_actions_minus_one)

        #     else:

        #         ations_t_minus_one=actions[i]

        #         x_cur_t_minus_one,x_ac_t_minus_one,x_tt_t_minus_one,x_ap_t_minus_one,x_at_t_minus_one,x_win_t_minus_one,x_et_t_minus_one=ations_t_minus_one

        #         temp_actions_minus_one=torch.cat((x_cur_t_minus_one,x_ac_t_minus_one,x_win_t_minus_one,x_tt_t_minus_one,x_et_t_minus_one),1)
        #         air_actions_minus_one=torch.cat((x_cur_t_minus_one,x_win_t_minus_one,x_ap_t_minus_one, x_at_t_minus_one, x_et_t_minus_one),1)

        #         temp_actions_history=torch.cat((temp_actions_minus_one),0)
        #         air_action_history=torch.cat((air_action_history,air_actions_minus_one))


        # temp_actions_history=temp_actions_history.reshape((self.minibatchSize,-1))
        # air_action_history=air_action_history.reshape((self.minibatchSize,-1))


        # %% states at t
        
        states_list_t=None

        chosen_action_qvals=None
        
        for i in range(len(states_list_tt)):
            
            states_list_t=states_list_tt[i]

            # states_list_t: X_US, X_LE, X_TE, X_TR, X_AE, X_AR

            # light_agent_actions, self.hidden_states_light=self.light_service.agents(states_list_t[:,[i for i in range(X_US.shape[1])]+[X_US.shape[1]]], self.hidden_states_light)

            temp_agent_actions, self.hidden_states_temp=self.temp_service.agents(states_list_t[:,[j for j in range(X_US.shape[1])]+[X_US.shape[1]+0]+[X_US.shape[1]+1]], self.hidden_states_temp)

            air_agent_actions, self.hidden_states_air=self.air_service.agents(states_list_t[:,[j for j in range(X_US.shape[1])]+[X_US.shape[1]+2]+[X_US.shape[1]+3]], self.hidden_states_air)

            # common_agent_actions, self.hidden_states_common=self.common_service.agents(states_list_t, self.hidden_states_common)

            agent_action=torch.cat((temp_agent_actions,air_agent_actions),1)

            chosen_action_qval=self.qmixer(agent_action, states_list_t)


            if i==0:

                chosen_action_qvals=chosen_action_qval

            else:

                chosen_action_qvals=torch.cat((chosen_action_qvals,chosen_action_qval),0)
                
        # print(f'chosen_action_qvals: {chosen_action_qvals}')

        chosen_action_qvals=chosen_action_qvals.reshape((self.minibatchSize,-1))

        # %%

        states_list_tt_plus_1=[transition[3] for transition in minibatch]
        states_list_t_plus_1=None

        target_max_qvals=None
        
        for i in range(len(states_list_tt_plus_1)):
            
            states_list_t_plus_1=states_list_tt_plus_1[i]

            # light_agent_actions, self.target_hidden_states_light=self.light_service.target_agents(states_list_t_plus_1[:,[i for i in range(X_US.shape[1])]+[X_US.shape[1]]], self.target_hidden_states_light)
            temp_agent_actions, self.target_hidden_states_temp=self.temp_service.target_agents(states_list_t_plus_1[:,[j for j in range(X_US.shape[1])]+[X_US.shape[1]+0]+[X_US.shape[1]+1]], self.target_hidden_states_temp)
            air_agent_actions, self.target_hidden_states_air=self.air_service.target_agents(states_list_t_plus_1[:,[j for j in range(X_US.shape[1])]+[X_US.shape[1]+2]+[X_US.shape[1]+3]], self.target_hidden_states_air)
            # common_agent_actions, self.target_hidden_states_common=self.common_service.target_agents(states_list_t_plus_1,self.target_hidden_states_common)

            target_agent_action=torch.cat((temp_agent_actions,air_agent_actions),1)

            target_max_qval=self.target_qmixer(target_agent_action, states_list_t_plus_1)


            if i==0:

                target_max_qvals=target_max_qval

            else:

                target_max_qvals=torch.cat((target_max_qvals,target_max_qval),0)

        target_max_qvals=target_max_qvals.reshape((self.minibatchSize,-1))


        yi=r_batch+self.gamma*(1-t_batch)*target_max_qvals


        td_error=(chosen_action_qvals-yi.detach())

        loss=(td_error**2).sum()/td_error.sum()

        # print(f'loss: {loss}')

        self.optimizer.zero_grad()

        loss.backward()

        # grad_norm=torch.nn.utils.clip_grad_norm(self.params,10)

        self.optimizer.step



    def act(self, obs, training=False):

        # light_value_action, self.hidden_states_light=self.light_service.agents(obs[:,[i for i in range(X_US.shape[1])]+[X_US.shape[1]]],self.hidden_states_light)
        temp_value_action, self.hidden_states_temp=self.temp_service.agents(obs[:,[i for i in range(X_US.shape[1])]+[X_US.shape[1]+0]+[X_US.shape[1]+1]],self.hidden_states_temp)
        air_value_action, self.hidden_states_air=self.air_service.agents(obs[:,[i for i in range(X_US.shape[1])]+[X_US.shape[1]+2]+[X_US.shape[1]+3]], self.hidden_states_air)

        target_agent_action=torch.cat((temp_value_action,air_value_action),1)
        Q_final=self.qmixer(target_agent_action, obs)

        Q_cur_t=Q_final[:,:X_CUR.shape[1]]
        Q_temp_ac_t=Q_final[:,X_CUR.shape[1]:(X_CUR.shape[1]+X_AC.shape[1])]
        Q_win_t=Q_final[:,(X_CUR.shape[1]+X_AC.shape[1]):(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1])]
        Q_air_ap_t=Q_final[:,(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]):(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1])]
        Q_temp_tt_t=Q_final[:,(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1]):(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1]+X_T.shape[1])]
        Q_air_at_t=Q_final[:,(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1]+X_T.shape[1]):(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1]+X_T.shape[1]+X_T.shape[1])]
        Q_et_t=Q_final[:,(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1]+X_T.shape[1]+X_T.shape[1]):(X_CUR.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1]+X_T.shape[1]+X_T.shape[1]+X_ET.shape[1])]


        # common_value_action, self.hidden_states_common=self.common_service.agents(obs, self.hidden_states_common)

        if training:

            # x_ls_t=self.epsilon_greedy.act(light_value_action,X_LS)
            x_ac_t=self.epsilon_greedy.act(Q_temp_ac_t, X_AC)
            x_tt_t=self.epsilon_greedy.act(Q_temp_tt_t,X_T)
            x_ap_t=self.epsilon_greedy.act(Q_air_ap_t,X_AP)
            x_at_t=self.epsilon_greedy.act(Q_air_at_t,X_T)
            x_cur_t=self.epsilon_greedy.act(Q_cur_t,X_CUR)
            x_win_t=self.epsilon_greedy.act(Q_win_t, X_WIN)
            x_et_t=self.epsilon_greedy.act(Q_et_t, X_ET)


        else:

            # ls_idx=light_value_action.max(dim=1)[1].cpu.detach().numpy()
            # x_ls_t=X_LS[:,int(ls_idx.item())].reshape(-1,1)

            ac_idx=temp_value_action[:,:X_AC.shape[1]].max(dim=1)[1].cpu.detach().numpy()
            x_ac_t=X_AC[:,int(ac_idx.item())].reshape(-1,1)

            tt_idx=temp_value_action[:,X_AC.shape[1]:].max(dim=1)[1].cpu.detach().numpy()
            x_tt_t=X_T[:,int(tt_idx.item())].reshape(-1,1)

            ap_idx=air_value_action[:,:X_AP.shape[1]].max(dim=1)[1].cpu.detach().numpy()
            x_ap_t=X_AP[:,int(ap_idx.item())].reshape(-1,1)

            at_idx=air_value_action[:,X_AP.shape[1]:].max(dim=1)[1].cpu.detach().numpy()
            x_at_t=X_T[:,int(at_idx.item())].reshape(-1,1)

            cur_idx=common_value_action[:,:X_CUR.shape[1]].max(dim=1)[1].cpu.detach().numpy()
            x_cur_t=X_CUR[:,int(cur_idx.item())].reshape(-1,1)

            win_idx=common_value_action[:,:X_CUR.shape[1]:(X_CUR.shape[1]+X_WIN.shape[1])].max(dim=1)[1].cpu.detach().numpy()
            x_win_t=X_WIN[:,int(win_idx.item())].reshape(-1,1)

            et_idx=common_value_action[:,(X_CUR.shape[1]+X_WIN.shape[1]):].max(dim=1)[1].cpu.detach().numpy()
            x_et_t=X_ET[:,int(et_idx.item())].reshape(-1,1)

        return  x_ac_t, x_tt_t, x_ap_t, x_at_t, x_cur_t, x_win_t, x_et_t











        








            
        



















        






















