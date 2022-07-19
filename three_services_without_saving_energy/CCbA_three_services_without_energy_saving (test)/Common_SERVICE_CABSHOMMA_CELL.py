
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

from simulated_light import *
from simulated_temp_3_2_tanh import *
from simulated_co2 import *

from LSTM_TEMP_SERVICE_CABSHOMMA_CELL import *


import copy

import math


# %%

random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%

batch_size=120
num_epochs=1000
steps=288

# %%
X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)

X_LS=torch.tensor([[0,1,2,3,4]],dtype=torch.float32,device=device)

X_CUR=torch.tensor([[0,1/2,1]],dtype=torch.float32, device=device)

X_WIN=torch.tensor([[0,1]],dtype=torch.float32,device=device)

X_AC=torch.tensor([[0,1,-1]],dtype=torch.float32,device=device)

X_T=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)
X_T=X_T[[0],1:]

X_ET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

# X_AC=torch.cat((torch.tensor([[0]],dtype=torch.float32,device=device),X_AC),axis=1)

X_AP=torch.tensor([[0,1,2,3,4,5]],dtype=torch.float32,device=device)

# %%

class LstmServiceModel(torch.nn.Module):
    def __init__(self,hidden_layers=400):
        super(LstmServiceModel,self).__init__()
        
        self.num_output_features=X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]

        self.input=X_US.shape[1]+1+1+1+1
        
        self.hidden_layers=hidden_layers
        
        self.lstm1=torch.nn.LSTMCell(self.input,self.hidden_layers)
        self.lstm2=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.lstm3=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.linear=torch.nn.Linear(self.hidden_layers,self.num_output_features)
            
    def forward(self,y):
       
        
        if len(y.shape)!=3:
            y=torch.from_numpy(np.expand_dims(y, axis=0))
            
    
        n_samples =y.size(0)
        
        h_t, c_t = self.lstm1(y[0])
        h_t2, c_t2 = self.lstm2(h_t)
        h_t3, c_t3 = self.lstm2(h_t2)
        
        output = self.linear(h_t3)
       

        return output
    

# %%

def user_simulator():
    X_us_t=torch.randint(0,4,(1,1)).to(device)
    return X_us_t



class OneHotEncoderClass:
    def __init__(self):
        pass
    def _one_hot_encoder(self,X,x):
        
        zeros=torch.zeros(X.shape,dtype=torch.float32,device=device)
        # print(f'zeros shape: {zeros.shape}')
        pos=torch.where(X==x)[1].item()
        zeros[0,pos]=1
        one_hot_encod=zeros
        
        return one_hot_encod
    
# %%

class CommonService_cell:
    
    def __init__(self,
                 replayMemorySize=1000000,minibatchSize=128,discount=0.1, learningRate=0.9):
        
        self.replayMemorySize=replayMemorySize
        self.minibatchSize=minibatchSize
        self.discount=discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        
        self.replayMemory=deque(maxlen=self.replayMemorySize)
        
        self.commonServiceModel=self.createModel().to(device)
        
        
        self.optimizer=torch.optim.Adam(self.commonServiceModel.parameters(),lr=0.1)
        
        
        self.targetCommonServiceModel=self.createModel().to(device)
        
        self.targetCommonServiceModel.load_state_dict(self.commonServiceModel.state_dict())
        
        self.targetCommonServiceModel.eval()
        
        # %%
        
    def updateReplayMemory(self,transition):
        self.replayMemory.append(transition)
        
        # %%
        
    def setStates(self,x_us_t,x_ar_t,x_ae_t,x_ap_t,x_win_t,x_cur_t, x_tr_t,x_te_t, x_t_t, x_et_t, MAX_CO2, MAX_LE, MAX_TEMP):
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
        self.MAX_LE=MAX_LE
        self.MAX_TEMP=MAX_TEMP        
        # %%
                              
   
    def createModel(self):
        
       commonServiceModel=LstmServiceModel()
        
       return commonServiceModel
   
   # %%
    
    def getActions(self,X_t_norm):
        
        Q=self.commonServiceModel(X_t_norm.float())
       
        Q_cur_t=Q[:,:len(X_CUR[0])].reshape(1,-1)
        Q_win_t=Q[:,len(X_CUR[0]):(X_CUR.shape[1]+X_WIN.shape[1])].reshape(1,-1)
        Q_et_t=Q[:,(X_CUR.shape[1]+X_WIN.shape[1]):].reshape(1,-1)
       
        
        return Q_cur_t,Q_win_t,Q_et_t
        
      # %%
      
      
    def train(self,epoch):
        if len(self.replayMemory) < self.minibatchSize:
            return 
        
        minibatch=random.sample(self.replayMemory,self.minibatchSize)
        
        states_list_tt=[transition[0] for transition in minibatch]
        
        states_list_t=None
        
        for i in range(len(states_list_tt)):
            if i==0:
                states_list_t=states_list_tt[i]
            else:
                states_list_ttt=states_list_tt[i]
                states_list_t=torch.cat((states_list_t,states_list_ttt),axis=0)
        
        states_list_t=states_list_t.reshape((self.minibatchSize,-1))
        
        q_list_t=self.commonServiceModel(states_list_t.float()).reshape((self.minibatchSize,-1))
        
        q_cur_list_t=q_list_t[:,:len(X_CUR[0])]
        q_win_list_t=q_list_t[:,len(X_CUR[0]):(X_CUR.shape[1]+X_WIN.shape[1])]
        q_et_list_t=q_list_t[:,(X_CUR.shape[1]+X_WIN.shape[1]):]
       
        
        
        states_list_tt_plus_1=[transition[3] for transition in minibatch]
        states_list_t_plus_1=None
        
        for i in range(len(states_list_tt_plus_1)):
            if i==0:
                states_list_t_plus_1=states_list_tt_plus_1[i]
            else:
                states_list_ttt_plus_1=states_list_tt_plus_1[i]
                states_list_t_plus_1=torch.cat((states_list_t_plus_1,states_list_ttt_plus_1),axis=0)
                
        states_list_t_plus_1=states_list_t_plus_1.reshape((self.minibatchSize,-1))
        
        q_list_t_plus_1=self.targetCommonServiceModel(states_list_t_plus_1.float())[:,:].detach().reshape((self.minibatchSize,-1))
        
        # q_ap_list_t_plus_1=q_list_t_plus_1[:,:len(X_CUR[0])]
        # q_t_list_t_plus_1=q_list_t_plus_1[:,(len(X_CUR[0])+len(X_AP[0]))+len(X_WIN[0]):(len(X_CUR[0])+len(X_AP[0])+len(X_WIN[0])+len(X_T[0]))]
        
        q_cur_list_t_plus_1=q_list_t_plus_1[:,:len(X_CUR[0])]
        q_win_list_t_plus_1=q_list_t_plus_1[:,len(X_CUR[0]):(X_CUR.shape[1]+X_WIN.shape[1])]
        q_et_list_t_plus_1=q_list_t_plus_1[:,(X_CUR.shape[1]+X_WIN.shape[1]):]

        
        X=None
        Y=None
        
        for index, (states_t,actions,reward_t,states_t_plus_1) in enumerate(minibatch):
            # if index!=23:
            
           
            max_q_cur_t_plus_1=torch.max(q_cur_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_cur_t_plus_1=reward_t+self.discount*max_q_cur_t_plus_1
            
            max_q_win_t_plus_1=torch.max(q_win_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_win_t_plus_1=reward_t+self.discount*max_q_win_t_plus_1

            max_q_et_t_plus_1=torch.max(q_et_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_et_t_plus_1=reward_t+self.discount*max_q_et_t_plus_1

          

            q_win_t=q_win_list_t[index,:]
            q_win_t=q_win_t.reshape(1,-1)
            
            q_cur_t=q_cur_list_t[index,:]
            q_cur_t=q_cur_t.reshape(1,-1)

            q_et_t=q_et_list_t[index,:]
            q_et_t=q_et_t.reshape(1,-1)


            action_cur_t,action_win_t,action_et_t=actions

            
            action_win_t_item=action_win_t.item()
            action_win_t_item=(X_WIN==action_win_t_item).nonzero(as_tuple=True)[1].item()
            
            action_cur_t_item=action_cur_t.item()
            action_cur_t_item=(X_CUR==action_cur_t_item).nonzero(as_tuple=True)[1].item()
            
            action_et_t_item=action_et_t.item()
            action_et_t_item=(X_ET==action_et_t_item).nonzero(as_tuple=True)[1].item()


            q_cur_t[0,int(action_cur_t_item)]=(1-self.learningRate)*q_cur_t[0,int(action_cur_t_item)]+self.learningRate*new_q_cur_t_plus_1
          
            q_win_t[0,int(action_win_t_item)]=(1-self.learningRate)*q_win_t[0,int(action_win_t_item)]+self.learningRate*new_q_win_t_plus_1
            # q_ap_t[0,:int(action_ap_t_item)]=self.learningRate*q_ap_t[0,:int(action_ap_t_item)]-(1-self.learningRate)*reward_t
            # q_ap_t[0,int(action_ap_t_item)+1:]=self.learningRate*q_ap_t[0,int(action_ap_t_item)+1:]-(1-self.learningRate)*reward_t
          
            q_et_t[0,int(action_et_t_item)]=(1-self.learningRate)*q_et_t[0,int(action_et_t_item)]+self.learningRate*new_q_et_t_plus_1
            
            if index==0:
                X=copy.deepcopy(states_t)
                Y=torch.cat((q_cur_t, q_win_t, q_et_t),axis=1)
                
            else:
                X=torch.cat((X,states_t),axis=0)
                q_t=torch.cat((q_cur_t, q_win_t, q_et_t),axis=1)
                Y=torch.cat((Y,q_t),axis=0)

        # print(f'Y: {Y.shape}')
                
        data_size=len(X)
        validation_pct=0.2
        
        train_size=math.ceil(data_size*(1-validation_pct))
        
        X_train,Y_train=X[:train_size,:],Y[:train_size,:]
        X_test,Y_test=X[train_size:,:],Y[train_size:,:]

        if epoch>=0:
            self.optimizer=torch.optim.Adam(self.commonServiceModel.parameters())
            
        least_val_loss=0
        total_patience=50
        patience=0
        
            
        outputs=self.commonServiceModel(X_train.float()).reshape((train_size,-1))
        
        criterion=torch.nn.MSELoss()
        
        # print(f'outputs: {outputs.shape}')
        # print(f'Y_train: {Y_train.shape}')


        loss=criterion(outputs,Y_train)
        
        
        self.optimizer.zero_grad()
        
        loss.backward(retain_graph=True)
        
        self.optimizer.step()
        
        criterion_val=torch.nn.MSELoss()
        
        self.commonServiceModel.eval()
        
        total_val_loss=torch.tensor([[0]],dtype=torch.float32,device=device)
        
        with torch.no_grad():
            n_correct=0
            n_samples=0
            
            x_test=X_test
            y_test=Y_test
            
            outputs=self.commonServiceModel(x_test.float()).reshape((x_test.shape[0],-1))
            
            val_loss=criterion_val(outputs,y_test)
            
            print(f'val loss: {val_loss.item():.4f}, train loss: {(loss.item()):.4f}')
            
            # if epoch==0:
            #     least_val_loss=total_val_loss
            # else:
            #     if least_val_loss>total_val_loss:
            #         least_val_loss=total_val_loss
            #     else:
            #         patience+=1
            
            # if patience==50:
            #     torch.save(airService.airServiceModel.state_dict(),'data/structure1/airService_v1.pth')
            #     print("end training")
            #     return
        
        self.commonServiceModel.train()
        
        return minibatch

