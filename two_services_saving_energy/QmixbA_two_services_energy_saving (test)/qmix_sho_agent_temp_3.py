import qmix_sho_rnn_agent
import qmix_sho_qmixer
import torch
import numpy as np
import random
from collections import deque
import copy


from qmix_sho_rnn_agent_3 import *

use_cuda=torch.cuda.is_available()
device=torch.device('cpu')

class TempService_cell:

    def __init__(self, replayMemorySize=1000000,minibatchSize=128,
                 discount=0.1, learningRate=0.9):
        
        self.replayMemorySize=replayMemorySize
        self.minibatchSize=minibatchSize
        self.discount=discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        
        self.agents=RNNAgent(input_shape=6,n_actions=X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]).to(device)
        self.target_agents=RNNAgent(input_shape=6,n_actions=X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]).to(device)

        self.target_agents.update(self.agents)

        self.params=list(self.agents.parameters())


    def setStates(self,x_us_t,x_tr_t,x_te_t,x_cur_t,x_ac_t,x_win_t,MAX_TEMP,x_t_t,x_et_t):
        self.x_us_t=x_us_t
        self.x_tr_t=x_tr_t
        self.x_te_t=x_te_t
        self.x_cur_t=x_cur_t
        self.x_ac_t=x_ac_t
        self.x_win_t=x_win_t
        self.MAX_TEMP=MAX_TEMP
        self.x_t_t=x_t_t
        self.x_et_t=x_et_t
        
    def getReward(self,x_tr_new_t_2,x_tr_new_t_ori):
        
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            if self.x_tr_t==x_tr_new_t_ori:
                self.reward=8
            else:
                self.reward=-torch.abs(x_tr_new_t_2-x_tr_new_t_ori).item()
                
        if self.x_us_t==torch.tensor([[1]],dtype=torch.float32,device=device):
            if self.x_tr_t>=torch.tensor([[23]],dtype=torch.float32,device=device) and self.x_tr_t<=torch.tensor([[25]],dtype=torch.float32,device=device):
                if self.x_ac_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=8
                
                else:
                    
                    x_ac_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    for i in range(X_CUR.shape[1]):
                        if stop:
                            break
                        for j in range(X_WIN.shape[1]):
                            if stop:
                                break
                            for k in range(X_T.shape[1]):
                                if stop:
                                    break
                                for k2 in range(X_ET.shape[1]):
                                    x_cur_new_t_int=X_CUR[[[0]],i]
                                    x_win_new_t_int=X_WIN[[[0]],j]
                                    x_t_new_t_int=X_T[[[0]],k]
                                    x_et_new_t_int=X_ET[[[0]],k2]
                                    x_tr_new_t_int,_=self.getIndoorTemp(self.x_tr_t,x_ac_new_t_int,x_cur_new_t_int,x_win_new_t_int,x_t_new_t_int, x_et_new_t_int)
                                    if x_tr_new_t_int>=torch.tensor([[23]],dtype=torch.float32,device=device) and x_tr_new_t_int<=torch.tensor([[25]],dtype=torch.float32,device=device):
                                        self.reward=-torch.abs(self.x_tr_t-24).item()*2
                                        stop=True
                                        break
                                    else:
                                        self.reward=4
            else:
                self.reward=-torch.abs(x_tr_new_t_2-24).item()
                
        if self.x_us_t==torch.tensor([[2]],dtype=torch.float32,device=device):
            if self.x_tr_t>=torch.tensor([[20]],dtype=torch.float32,device=device) and self.x_tr_t<=torch.tensor([[22]],dtype=torch.float32,device=device):
                if self.x_ac_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=8
                else:
                    x_ac_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    for i in range(X_CUR.shape[1]):
                        if stop:
                            break
                        for j in range(X_WIN.shape[1]):
                            if stop:
                                break
                            for k in range(X_T.shape[1]):
                                if stop:
                                    break
                                
                                for k2 in range(X_ET.shape[1]):
                                    x_cur_new_t_int=X_CUR[[[0]],i]
                                    x_win_new_t_int=X_WIN[[[0]],j]
                                    x_t_new_t_int=X_T[[[0]],k]
                                    x_et_new_t_int=X_ET[[[0]],k2]

                                    x_tr_new_t_int,_=self.getIndoorTemp(self.x_tr_t,x_ac_new_t_int,x_cur_new_t_int,x_win_new_t_int,x_t_new_t_int, x_et_new_t_int)
                                    if x_tr_new_t_int>=torch.tensor([[20]],dtype=torch.float32,device=device) and x_tr_new_t_int<=torch.tensor([[22]],dtype=torch.float32,device=device):
                                        self.reward=-torch.abs(self.x_tr_t-21).item()*2
                                        stop=True
                                        break
                                    else:
                                        self.reward=4
                                    
            else:
                self.reward=-torch.abs(x_tr_new_t_2-21).item()
                
        if self.x_us_t==torch.tensor([[3]],dtype=torch.float32,device=device): 
            if self.x_tr_t>=torch.tensor([[17]],dtype=torch.float32,device=device) and self.x_tr_t<=torch.tensor([[19]],dtype=torch.float32,device=device):
                if self.x_ac_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=8
                else:
                    x_ac_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    for i in range(X_CUR.shape[1]):
                        if stop:
                            break
                        for j in range(X_WIN.shape[1]):
                            if stop:
                                break
                            for k in range(X_T.shape[1]):
                                if stop:
                                    break
                                for k2 in range(X_ET.shape[1]):
                                    x_cur_new_t_int=X_CUR[[[0]],i]
                                    x_win_new_t_int=X_WIN[[[0]],j]
                                    x_t_new_t_int=X_T[[[0]],k]
                                    x_et_new_t_int=X_ET[[[0]],k2]
                                    x_tr_new_t_int,_=self.getIndoorTemp(self.x_tr_t,x_ac_new_t_int,x_cur_new_t_int,x_win_new_t_int,x_t_new_t_int,x_et_new_t_int)
                                    if x_tr_new_t_int>=torch.tensor([[17]],dtype=torch.float32,device=device) and x_tr_new_t_int<=torch.tensor([[19]],dtype=torch.float32,device=device):
                                        self.reward=-torch.abs(self.x_tr_t-18).item()*2
                                        stop=True
                                        break
                                    else:
                                        self.reward=4
                                    
            else:
                self.reward=-torch.abs(x_tr_new_t_2-18).item()
                
        return self.reward
                            
    
    
    def getIndoorTemp(self,x_tr_t,x_ac_t,x_cur_t,x_win_t,x_t_t,x_et_t):

        theta_ac=20.0*735.0
        cp=1.005
        pho=1.205
        v=60.0
        h=2.0
        d_l=2.0
        d_w=0.2
        g=9.81
        lamda=0.019
        epsilon=1.0
        
        delta_t=x_et_t*60 
        delta_t_ac=x_ac_t*x_t_t/60 # 3 minutes heating
        e_ap=theta_ac*delta_t_ac

        # print(f'e_ap: {e_ap}')
        # print(f'e_ap temp: {e_ap.item()/(cp*pho*v)}')
        # # print(f'========')
        
        
        openness=torch.min(x_cur_t,x_win_t)

        # print(f'x_win_t: {x_win_t}')
        # print(f'x_cur_t: {x_cur_t}')
        # print(f'openness: {openness}')
        
        # print(f'========')
        
        # the air intensity inside the room
        pho_r=1.293*273/(273+self.x_tr_t)
        pho_e=1.293*273/(273+self.x_te_t)
        
        # print(f'pho_r: {pho_r}')
        # print(f'pho_e: {pho_e}')
        
        # print(f'========')
        
        
        # air flow rate
        if self.x_te_t>=self.x_tr_t:
            L_t=h*openness*d_l*torch.sqrt((2*g*torch.abs((pho_e-pho_r))*h*openness)/(lamda*d_w*pho_r/d_l+epsilon*pho_r))
        else:
            L_t=-h*openness*d_l*torch.sqrt((2*g*torch.abs((pho_e-pho_r))*h*openness)/(lamda*d_w*pho_r/d_l+epsilon*pho_r))
        
        e_env=L_t*delta_t*pho*cp

        # print(f'e_env: {e_env}')
        # print(f'e_env temp: {e_env.item()/(cp*pho*v)}')
        # print(f'========')
        
        e=e_ap+e_env
        
        diff_t=e/(cp*pho*v)

        # print(f'diff_t: {diff_t}')
        
        x_tr_new_t=x_tr_t+diff_t

        x_tr_new_t_2=copy.deepcopy(x_tr_new_t)

        
        # print(f'========')

        x_tr_new_t=torch.min(torch.tensor([[26]], dtype=torch.float32), x_tr_new_t)      
        x_tr_new_t=torch.max(torch.tensor([[12]], dtype=torch.float32), x_tr_new_t)        
        
        # x_tr_new_t=x_tr_new_t.reshape((1,-1))
        return x_tr_new_t,x_tr_new_t_2
        
        
        
    def createModel(self):
        # hidden : 26, 58
        tempServiceModel=LstmServiceModel()
        
        return tempServiceModel
    
    def getActions(self,X_t_norm):
        
        
        
        # print(f'X_t_norm: {X_t_norm}')
        
        Q,_=self.tempServiceModel(X_t_norm.float())
        
        # print(f'Q: {Q}')
        Q_cur_t=Q[:,:len(X_CUR[0])].reshape(1,-1)
        Q_ac_t=Q[:,len(X_CUR[0]):(len(X_CUR[0])+len(X_AC[0]))].reshape(1,-1)
        Q_win_t=Q[:,len(X_CUR[0])+len(X_AC[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0]))].reshape(1,-1)
        Q_t_t=Q[:,(len(X_CUR[0])+len(X_AC[0]))+len(X_WIN[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0])+len(X_T[0]))].reshape(1,-1)
        
        Q_et_t=Q[:, -len(X_ET[0]):].reshape(1,-1)
        
        
        return Q_cur_t,Q_ac_t,Q_win_t,Q_t_t, Q_et_t
        
        
    
  