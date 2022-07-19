import qmix_sho_rnn_agent
import qmix_sho_qmixer
import torch
import numpy as np
import random
from collections import deque

from qmix_sho_rnn_agent_3 import *
import copy


use_cuda=torch.cuda.is_available()
device=torch.device('cpu')

class AirService_cell:

    def __init__(self,
                 replayMemorySize=1000000,minibatchSize=128,discount=0.1, learningRate=0.9):
        
        self.replayMemorySize=replayMemorySize
        self.minibatchSize=minibatchSize
        self.discount=discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        
        self.agents=RNNAgent(input_shape=6,n_actions=X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AP.shape[1]+X_T.shape[1]).to(device)
        self.target_agents=RNNAgent(input_shape=6,n_actions=X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AP.shape[1]+X_T.shape[1]).to(device)

        self.target_agents.update(self.agents)

        self.params=list(self.agents.parameters())
        
        # %%
        
  
    def setStates(self,x_us_t,x_ar_t,x_ae_t,x_ap_t,x_win_t,x_cur_t, x_tr_t,x_te_t, x_t_t, x_et_t, MAX_CO2):
        self.x_us_t=x_us_t
        self.x_ar_t=x_ar_t
        self.x_ae_t=x_ae_t
        self.x_ap_t=x_ap_t
        self.x_win_t=x_win_t
        self.x_cur_t=x_cur_t
        self.x_tr_t=x_tr_t
        self.x_te_t=x_te_t
        self.x_t_t=x_t_t
        self.x_et_t=x_et_t
        
        self.MAX_CO2=MAX_CO2
        
        # %%
        
    def getReward(self,x_ar_new_t_2,x_ar_new_t_ori):
        
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            if self.x_ar_t==x_ar_new_t_ori:
                self.reward=200
            else:
                self.reward=-torch.abs(x_ar_new_t_2-x_ar_new_t_ori).item()
                
        if self.x_us_t==torch.tensor([[1]],dtype=torch.float32,device=device):
            if self.x_ar_t<=torch.tensor([[300]],dtype=torch.float32,device=device) and self.x_ar_t>=torch.tensor([[100]],dtype=torch.float32,device=device):
                # if self.x_ap_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=200
                
                # else:
                    
                #     x_ap_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                #     stop=False
                #     for i in range(X_CUR.shape[1]):
                #         if stop:
                #             break
                #         for j in range(X_WIN.shape[1]):
                #             if stop:
                #                 break
                #             for k in range(X_T.shape[1]):
                #                 if stop:
                #                     break
                #                 for k2 in range(X_ET.shape[1]):
                #                     x_cur_new_t_int=X_CUR[[[0]],i]
                #                     x_win_new_t_int=X_WIN[[[0]],j]
                #                     x_t_new_t_int=X_T[[[0]],k]
                #                     x_et_new_t_int=X_ET[[[0]],k2]
                #                     x_ar_new_t_int,_=self.getIndoorAir(self.x_ar_t,x_ap_new_t_int,x_cur_new_t_int,x_win_new_t_int, x_t_new_t_int,x_et_new_t_int)
                #                     if x_ar_new_t_int<=torch.tensor([[300]],dtype=torch.float32,device=device) and x_ar_new_t_int>=torch.tensor([[100]],dtype=torch.float32,device=device):
                #                         self.reward=-torch.abs(self.x_ar_t-200).item()*2
                #                         stop=True
                #                         break
                #                     else:
                #                         self.reward=100
            else:
                self.reward=-torch.abs(x_ar_new_t_2-200).item()
                
        if self.x_us_t==torch.tensor([[2]],dtype=torch.float32,device=device):
            if self.x_ar_t<=torch.tensor([[400]],dtype=torch.float32,device=device) and self.x_ar_t>=torch.tensor([[200]],dtype=torch.float32,device=device):
                # if self.x_ap_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=200
                
                # else:
                    
                #     x_ap_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                #     stop=False
                #     for i in range(X_CUR.shape[1]):
                #         if stop:
                #             break
                #         for j in range(X_WIN.shape[1]):
                #             if stop:
                #                 break
                #             for k in range(X_T.shape[1]):
                #                 if stop:
                #                     break
                #                 for k2 in range(X_ET.shape[1]):
                #                     x_cur_new_t_int=X_CUR[[[0]],i]
                #                     x_win_new_t_int=X_WIN[[[0]],j]
                #                     x_t_new_t_int=X_T[[[0]],k]
                #                     x_et_new_t_int=X_ET[[[0]],k2]
                #                     x_ar_new_t_int,_=self.getIndoorAir(self.x_ar_t,x_ap_new_t_int,x_cur_new_t_int,x_win_new_t_int, x_t_new_t_int,x_et_new_t_int)
                #                     if x_ar_new_t_int<=torch.tensor([[400]],dtype=torch.float32,device=device) and x_ar_new_t_int>=torch.tensor([[200]],dtype=torch.float32,device=device):
                #                         self.reward=-torch.abs(self.x_ar_t-300).item()*2
                #                         stop=True
                #                         break
                #                     else:
                #                         self.reward=100
            else:
                self.reward=-torch.abs(x_ar_new_t_2-300).item()
        
        if self.x_us_t==torch.tensor([[3]],dtype=torch.float32,device=device): 
            if self.x_ar_t<=torch.tensor([[200]],dtype=torch.float32,device=device) and self.x_ar_t>=torch.tensor([[100]],dtype=torch.float32,device=device):
                # if self.x_ap_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=200
                
                # else:
                    
                #     x_ap_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                #     stop=False
                #     for i in range(X_CUR.shape[1]):
                #         if stop:
                #             break
                #         for j in range(X_WIN.shape[1]):
                #             if stop:
                #                 break
                #             for k in range(X_T.shape[1]):
                #                 if stop:
                #                     break
                #                 for k2 in range(X_ET.shape[1]):
                #                     x_cur_new_t_int=X_CUR[[[0]],i]
                #                     x_win_new_t_int=X_WIN[[[0]],j]
                #                     x_t_new_t_int=X_T[[[0]],k]
                #                     x_et_new_t_int=X_ET[[[0]],k2]
                #                     x_ar_new_t_int,_=self.getIndoorAir(self.x_ar_t,x_ap_new_t_int,x_cur_new_t_int,x_win_new_t_int, x_t_new_t_int, x_et_new_t_int)
                #                     if x_ar_new_t_int<=torch.tensor([[200]],dtype=torch.float32,device=device) and x_ar_new_t_int>=torch.tensor([[100]],dtype=torch.float32,device=device):
                #                         self.reward=-torch.abs(self.x_ar_t-150).item()*2
                #                         stop=True
                #                         break
                #                     else:
                #                         self.reward=100
            else:
                self.reward=-torch.abs(x_ar_new_t_2-150).item()
                
        return self.reward
    
    
    # %%
                              
    def getIndoorAir(self,x_ar_t,x_ap_t,x_cur_t,x_win_t,x_t_t,x_et_t):
        
        pho=1.205
        v=60
        delta_t=x_et_t*60
        delta_t_ap=x_ap_t*x_t_t/60
        delta_t_b=5*60
        
        self.h=2
        
        self.d_l=2
        
        self.d_w=0.2
        
        self.g=9.81
        
        self.lamda=0.019
        
        self.epsilon=1

        exhaled_co2=38000
        
        B_us=torch.tensor([[0,11.004,31.44,7.6635]],dtype=torch.float32,device=device)
        
        n_us=0
        
        
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            n_us=0
        else:
            n_us=1
            
        b_us=B_us[0,int(self.x_us_t.item())]*10e-6
        
        L_ap=torch.tensor([[0,60,170,280,390,500]])
        l_ap=L_ap[0,int(x_ap_t.item())]
        
        openness=torch.min(x_cur_t,x_win_t)
        
        # the air intensity inside the room
        pho_r=1.293*273/(273+self.x_tr_t)
        pho_e=1.293*273/(273+self.x_te_t)


        
        
        # air flow rate
        if self.x_ae_t>=x_ar_t:
            L_t=abs(self.h*openness*self.d_l*torch.sqrt((2*self.g*torch.abs((pho_e-pho_r))*self.h*openness)/(self.lamda*self.d_w*pho_r/self.d_l+self.epsilon*pho_r)))
        else:
            L_t=-abs(self.h*openness*self.d_l*torch.sqrt((2*self.g*torch.abs((pho_e-pho_r))*self.h*openness)/(self.lamda*self.d_w*pho_r/self.d_l+self.epsilon*pho_r)))
            
        x_ar_new_t=self.x_ar_t*(1 - torch.min(torch.tensor([[1]],dtype=torch.float32), (l_ap*delta_t_ap)/v) - torch.min(torch.tensor([[1]],dtype=torch.float32), (L_t*delta_t)/v) - torch.min(torch.tensor([[1]],dtype=torch.float32), (n_us*b_us*delta_t_b)/v))+self.x_ae_t*(torch.min(torch.tensor([[1]],dtype=torch.float32), (L_t*delta_t)/v))+exhaled_co2*torch.min(torch.tensor([[1]],dtype=torch.float32), (n_us*b_us*delta_t_b)/v)

        # print('==================')

        # print(f'self.x_ar_t: {self.x_ar_t}')

        # print('==================')

        # print(f'self.x_ae_t: {self.x_ae_t}')

        # print('==================')

        # print(f'(1. l_ap*delta_t_ap)/v*x_ar_t: {self.x_ar_t*(l_ap*delta_t_ap)/v}')

        # print('==================')

        # print(f'(2. self.x_ar_t*L_t*delta_t)/v: {self.x_ar_t*(L_t*delta_t)/v}')

        # print('==================')

        # print(f'(3. self.x_ar_t*(n_us*b_us*delta_t_b)/v: {self.x_ar_t*(n_us*b_us*delta_t_b)/v}')

        # print('==================')

        # print(f'(4. self.x_ae_t*((L_t*delta_t)/v): {self.x_ae_t*((L_t*delta_t)/v)}')

        # print('==================')

        # print(f'(5. (n_us*b_us*delta_t_b)/v*exhaled_co2: {exhaled_co2*(n_us*b_us*delta_t_b)/v}')

        # print('==================')

        

        # print(f'x_ar_new_t: {x_ar_new_t}')



        # print('==================')
        
        # x_ar_new_t=max(torch.tensor([[0]], dtype=torch.float32),x_ar_new_t.reshape((1,-1)))

        x_ar_new_t_2=copy.deepcopy(x_ar_new_t)
        
        

        x_ar_new_t=torch.min(torch.tensor([[800]],dtype=torch.float32), x_ar_new_t)
        x_ar_new_t=torch.max(torch.tensor([[0]],dtype=torch.float32), x_ar_new_t)
        
        # print(f'2nd x_ar_new_t: {x_ar_new_t}')

        return x_ar_new_t,x_ar_new_t_2
    
    # %%
  
    
   