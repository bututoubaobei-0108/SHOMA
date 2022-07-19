import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qmix_sho_configuration import *

class QMixer(nn.Module):
    def __init__(self,n_agents, state_shape, mixing_embed_dim=64):

        super(QMixer,self).__init__()

        # n_agents: the number of agents
        self.n_agents=n_agents

        self.num_outputs=X_CUR.shape[1]+X_LS.shape[1]+X_AC.shape[1]+X_WIN.shape[1]+X_AP.shape[1]+X_T.shape[1]*2+X_ET.shape[1]

        # self.state_dim=int(np.prod(state_shape))
        self.state_dim=state_shape

        self.embed_dim=mixing_embed_dim

        self.hyper_w_1=nn.Linear(self.state_dim, self.embed_dim*self.num_outputs)
        

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_w_final=nn.Linear(self.state_dim,self.embed_dim)

        self.V=nn.Sequential(nn.Linear(self.state_dim,self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim,1))


    def forward(self,agent_qs,states):

        # bs=agent_qs.size(0)

        states=states.reshape(-1, self.state_dim).float()

        
        agent_qs = agent_qs.view(-1, self.num_outputs).float()        # (bs,self.num_outputs)

        w1=th.abs(self.hyper_w_1(states))                  # (bs,self.num_outputs)

        b1=self.hyper_b_1(states)                          # (bs,1)

        w1=w1.view(self.num_outputs, self.embed_dim)

        b1 = b1.view(-1, self.embed_dim)

        hidden=F.elu(th.matmul(agent_qs,w1)+b1)            

        # second layer
        w_final=th.abs(self.hyper_w_final(states))

        w_final=w_final.view(-1,self.embed_dim)

        v=self.V(states).view(-1,1)

        y=th.matmul(hidden,th.transpose(w_final,0,1))+v

        q_tot=y.view(-1,1)
        
        return q_tot

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)







