# this version regards all services as one lstm-based system
# inputs changes from (5,6,6) to (5,5,5)
# 
# %%
import torch 
import torch.nn.functional as F
import numpy as np
import pandas as pd

import math

from collections import deque

import random
import itertools

from simulated_light import *
from simulated_temp_3_2_tanh import *
from simulated_co2 import *

from LSTM_AIR_SERVICE_RANGE_8_v1_NOECO_CELL import *
from LSTM_TEMP_SERVICE_88_NOECO_CELL import *

# from CustomLstmCell import *

# %%

random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%

batch_size=120
num_epochs=100
steps=288

# %%

X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)

X_LS=torch.tensor([[0,1,2,3,4]],dtype=torch.float32,device=device)

X_CUR=torch.tensor([[0,1/2,1]],dtype=torch.float32, device=device)

X_WIN=torch.tensor([[0,1]],dtype=torch.float32,device=device)

X_AC=torch.tensor([[0,1,-1]],dtype=torch.float32,device=device)

X_T=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)
X_T=X_T[[0],1:]

# X_AT=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)
# X_AT=X_T[[0],1:]

X_TET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

X_AET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

X_ET=copy.deepcopy(X_AET)

X_AP=torch.tensor([[0,1,2,3,4,5]],dtype=torch.float32,device=device)

# %%

def user_simulator():
    X_us_t=torch.randint(0,4,(1,1)).to(device)
    return X_us_t

# %%

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

# lightService=LightService_cell()
# lightService.lightServiceModel.load_state_dict(torch.load('data/hydra/lightService_v7_mlt_5.pth'))
# lightService.targetLightServiceModel.load_state_dict(torch.load('data/hydra/lightService_v7_mlt_5.pth'))

tempService=TempService_cell()
# tempService.tempServiceModel.load_state_dict(torch.load('data/hydra/tempService_v7_mlt_5.pth'))
# tempService.targetTempServiceModel.load_state_dict(torch.load('data/hydra/tempService_v7_mlt_5.pth'))

airService=AirService_cell()
# airService.airServiceModel.load_state_dict(torch.load('data/hydra/airService_v7_mlt_5.pth'))
# airService.targetAirServiceModel.load_state_dict(torch.load('data/hydra/airService_v7_mlt_5.pth'))


# %%

class MultiServiceModel(torch.nn.Module):
    
    def __init__(self,hidden_layers=1000):
        
        super(MultiServiceModel,self).__init__()

        # user:5; le:1; te:1; ae:1
        
        self.input=X_US.shape[1]+1+1+1+1
        
        self.hidden_layers=hidden_layers

        self.output_size=X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]+X_AP.shape[1]+X_T.shape[1]
        
        self.lstm1=torch.nn.LSTMCell(self.input, self.hidden_layers)
        self.lstm2=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.lstm3=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        # self.lstm3=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        
        self.linear1=torch.nn.Linear(self.hidden_layers,self.output_size)
        # self.linear2=torch.nn.Linear(self.hidden_layers,self.hidden_layers)
        # self.linear3=torch.nn.Linear(self.hidden_layers,self.output_size)
       
    def forward(self,y):

        if len(y.shape)!=3:
            y=torch.from_numpy(np.expand_dims(y, axis=0))
        
        n_samples=1
        
        # samples=len(y)
        
        # h_t=torch.randn((n_samples, self.hidden_layers), dtype=torch.float32)
        # c_t=torch.randn((n_samples, self.hidden_layers), dtype=torch.float32)
        
        # # %%
        # h_t2=torch.randn((n_samples, self.hidden_layers), dtype=torch.float32)
        # c_t2=torch.randn((n_samples, self.hidden_layers), dtype=torch.float32)
        
        # # %%
        # h_t3=torch.randn((n_samples, self.hidden_layers), dtype=torch.float32)
        # c_t3=torch.randn((n_samples, self.hidden_layers), dtype=torch.float32)
        
 
        # outs=torch.zeros(len(y),self.output_size,dtype=torch.float32)
       
        
        # for i in range(len(y)):
            
            # print(y[i])
            
        h_t,c_t=self.lstm1(y[0])
        h_t2,c_t2=self.lstm2(h_t)
        h_t2,c_t2=self.lstm2(h_t2)
        # h_t3,c_t3=self.lstm3(h_t2)

        x=self.linear1(h_t2)
        # x=F.relu(self.linear2(x))
        # x=self.linear3(x)
        
        # outs[i,:]=x
        
        return x

def getActions_temp(Q):
        
    # print(f'Q: {Q}')
    Q_cur_t=Q[:,:len(X_CUR[0])].reshape(1,-1)
    Q_ac_t=Q[:,len(X_CUR[0]):(len(X_CUR[0])+len(X_AC[0]))].reshape(1,-1)
    Q_win_t=Q[:,len(X_CUR[0])+len(X_AC[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0]))].reshape(1,-1)
    Q_t_t=Q[:,(len(X_CUR[0])+len(X_AC[0]))+len(X_WIN[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0])+len(X_T[0]))].reshape(1,-1)
    
    Q_et_t=Q[:, -len(X_ET[0]):].reshape(1,-1)
    
    return Q_cur_t,Q_ac_t,Q_win_t,Q_t_t, Q_et_t

def getActions_air(Q):
        
    # Q=self.airServiceModel(X_t_norm.float())
    Q_cur_t=Q[:,:len(X_CUR[0])].reshape(1,-1)
    Q_win_t=Q[:,len(X_CUR[0]):(len(X_CUR[0])+len(X_WIN[0]))].reshape(1,-1)
    Q_ap_t=Q[:,(len(X_CUR[0])+len(X_WIN[0])):(len(X_CUR[0])+len(X_AP[0])+len(X_WIN[0]))].reshape(1,-1)
    Q_t_t=Q[:,(len(X_CUR[0])+len(X_AP[0])+len(X_WIN[0])):(len(X_CUR[0])+len(X_AP[0])+len(X_WIN[0])+len(X_T[0]))].reshape(1,-1)
    Q_et_t=Q[:, -len(X_ET[0]):].reshape(1,-1)
    
    return Q_cur_t,Q_win_t,Q_ap_t,Q_t_t, Q_et_t
    
# %%
class MultiService:
    
    def __init__(self,ReplayMemorySize=1000000,minibatchSize=128,discount=0.1, learningRate=0.9):
        
        self.minibatchSize=minibatchSize
        self.ReplayMemorySize=ReplayMemorySize
        self.discount=discount
        
        self.learningRate=learningRate
        
        self.replayMemory=deque(maxlen=self.ReplayMemorySize)
        
        self.multiServiceModel=self.createModel().to(device)
        
        self.optimizer=torch.optim.Adam(self.multiServiceModel.parameters(),lr=0.1)
        
        self.targetMultiServiceModel=self.createModel().to(device)
        
        self.targetMultiServiceModel.load_state_dict(self.multiServiceModel.state_dict())
        
        self.targetMultiServiceModel.eval()
        
    def createModel(self):
        multiServiceModel=MultiServiceModel().to(device)
        return multiServiceModel
    
    def updateReplayMemory(self, transition):
        self.replayMemory.append(transition)
        
   
    def train(self):
        
        if len(self.replayMemory) < self.minibatchSize:
            return 
        
        # start = random.randint(0, len(self.replayMemory) - self.minibatchSize)

        # minibatch=deque(itertools.islice(self.replayMemory, start, start+self.minibatchSize))

        minibatch=random.sample(self.replayMemory,self.minibatchSize)
        
  
        states_list_tt=[transition[0].float() for transition in minibatch]

        states_list_t=None
        
        for i in range(len(states_list_tt)):
            if i==0:
                states_list_t=states_list_tt[i]
            else:
                states_list_ttt=states_list_tt[i]
                states_list_t=torch.cat((states_list_t,states_list_ttt),axis=0)
        
        states_list_t=states_list_t.reshape((self.minibatchSize,-1))
       
        q_list_t=self.multiServiceModel(states_list_t)
        
        # q_ls_list_t=q_list_t[:,:X_LS.shape[1]]
        q_cur_list_t=q_list_t[:,:(X_CUR.shape[1])]
        q_win_list_t=q_list_t[:,(X_CUR.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1])]
        q_t_list_t=q_list_t[:,(X_CUR.shape[1]+X_WIN.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1])]
        
        
        q_ac_list_t=q_list_t[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1])]
        q_tet_list_t=q_list_t[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1])]
        q_ap_list_t=q_list_t[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]+X_AP.shape[1])]
        q_aet_list_t=q_list_t[:,-X_T.shape[1]:]
        
   
        
        states_list_tt_plus_1=[transition[3].float() for transition in minibatch]

        states_list_t_plus_1=None
        
        for i in range(len(states_list_tt_plus_1)):
            if i==0:
                states_list_t_plus_1=states_list_tt_plus_1[i]
            else:
                states_list_ttt_plus_1=states_list_tt_plus_1[i]
                states_list_t_plus_1=torch.cat((states_list_t_plus_1,states_list_ttt_plus_1),axis=0)
                
        states_list_t_plus_1=states_list_t_plus_1.reshape((self.minibatchSize,-1))
        
        
        q_list_t_plus_1=self.multiServiceModel(states_list_t_plus_1)
        
        # q_ls_list_t_plus_1=q_list_t_plus_1[:,:X_LS.shape[1]]
        q_cur_list_t_plus_1=q_list_t_plus_1[:,:(X_CUR.shape[1])]
        q_win_list_t_plus_1=q_list_t_plus_1[:,(X_CUR.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1])]
        q_t_list_t_plus_1=q_list_t_plus_1[:,(X_CUR.shape[1]+X_WIN.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1])]
        
        
        q_ac_list_t_plus_1=q_list_t_plus_1[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1])]
        q_tet_list_t_plus_1=q_list_t_plus_1[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1])]
        q_ap_list_t_plus_1=q_list_t_plus_1[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]+X_AP.shape[1])]
        q_aet_list_t_plus_1=q_list_t_plus_1[:,-X_T.shape[1]:]
        
     
        X=None
        Y=None
        
        y=torch.zeros(len(minibatch),X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]+X_AP.shape[1]+X_T.shape[1],dtype=torch.float32)
        
        X=states_list_t
        
        
        criterion=torch.nn.MSELoss()
        
        for index, (states_t,actions,reward_t,states_t_plus_1) in enumerate(minibatch):
            
            reward_temp_t,reward_air_t=reward_t
            
            reward_comm_t=(reward_temp_t+reward_air_t)/3
            reward_comm_2_t=(reward_temp_t+reward_air_t)/2
            
            # max_q_ls_t_plus_1=torch.max(q_ls_list_t_plus_1[index],dim=0,keepdim=True)[0]
            # new_q_ls_t_plus_1=reward_light_t+self.discount*max_q_ls_t_plus_1
            
            max_q_cur_t_plus_1=torch.max(q_cur_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_cur_t_plus_1=reward_comm_t+self.discount*max_q_cur_t_plus_1
            
            max_q_win_t_plus_1=torch.max(q_win_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_win_t_plus_1=reward_comm_2_t+self.discount*max_q_win_t_plus_1
            
            max_q_t_t_plus_1=torch.max(q_t_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_t_t_plus_1=reward_comm_2_t+self.discount*max_q_t_t_plus_1
            
            max_q_ac_t_plus_1=torch.max(q_ac_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_ac_t_plus_1=reward_temp_t+self.discount*max_q_ac_t_plus_1
            
            max_q_tet_t_plus_1=torch.max(q_tet_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_tet_t_plus_1=reward_temp_t+self.discount*max_q_tet_t_plus_1
            
            max_q_ap_t_plus_1=torch.max(q_ap_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_ap_t_plus_1=reward_air_t+self.discount*max_q_ap_t_plus_1
            
            max_q_aet_t_plus_1=torch.max(q_aet_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_aet_t_plus_1=reward_air_t+self.discount*max_q_aet_t_plus_1
            
            
            # q_ls_t=q_ls_list_t[index,:]
            # q_ls_t=q_ls_t.reshape(1,-1)
            
            q_cur_t=q_cur_list_t[index,:]
            q_cur_t=q_cur_t.reshape(1,-1)
            
            q_win_t=q_win_list_t[index,:]
            q_win_t=q_win_t.reshape(1,-1)
            
            q_t_t=q_t_list_t[index,:]
            q_t_t=q_t_t.reshape(1,-1)
            
            q_ac_t=q_ac_list_t[index,:]
            q_ac_t=q_ac_t.reshape(1,-1)
            
            q_tet_t=q_tet_list_t[index,:]
            q_tet_t=q_tet_t.reshape(1,-1)
            
            q_ap_t=q_ap_list_t[index,:]
            q_ap_t=q_ap_t.reshape(1,-1)
            
            q_aet_t=q_aet_list_t[index,:]
            q_aet_t=q_aet_t.reshape(1,-1)
            
            action_cur_t,action_win_t,action_t_t,action_ac_t,action_tet_t,action_ap_t, action_aet_t=actions
            
            # action_ls_t_item=action_ls_t.item()
            # action_ls_t_item=(X_LS==action_ls_t_item).nonzero(as_tuple=True)[1].item()
            
            action_cur_t_item=action_cur_t.item()
            action_cur_t_item=(X_CUR==action_cur_t_item).nonzero(as_tuple=True)[1].item()
            
            action_win_t_item=action_win_t.item()
            action_win_t_item=(X_WIN==action_win_t_item).nonzero(as_tuple=True)[1].item()
            
            action_t_t_item=action_t_t.item()
            action_t_t_item=(X_ET==action_t_t_item).nonzero(as_tuple=True)[1].item()
            
            action_ac_t_item=action_ac_t.item()
            action_ac_t_item=(X_AC==action_ac_t_item).nonzero(as_tuple=True)[1].item()
            
            action_tet_t_item=action_tet_t.item()
            action_tet_t_item=(X_T==action_tet_t_item).nonzero(as_tuple=True)[1].item()
            
            action_ap_t_item=action_ap_t.item()
            action_ap_t_item=(X_AP==action_ap_t_item).nonzero(as_tuple=True)[1].item()
            
            action_aet_t_item=action_aet_t.item()
            action_aet_t_item=(X_T==action_aet_t_item).nonzero(as_tuple=True)[1].item()
            
            # q_ls_t[0,int(action_ls_t_item)]=(1-self.learningRate)*q_ls_t[0,int(action_ls_t_item)]+self.learningRate*new_q_ls_t_plus_1
            q_cur_t[0,int(action_cur_t_item)]=(1-self.learningRate)*q_cur_t[0,int(action_cur_t_item)]+self.learningRate*new_q_cur_t_plus_1
            q_win_t[0,int(action_win_t_item)]=(1-self.learningRate)*q_win_t[0,int(action_win_t_item)]+self.learningRate*new_q_win_t_plus_1
            q_t_t[0,int(action_t_t_item)]=(1-self.learningRate)*q_t_t[0,int(action_t_t_item)]+self.learningRate*new_q_t_t_plus_1
            q_ac_t[0,int(action_ac_t_item)]=(1-self.learningRate)*q_ac_t[0,int(action_ac_t_item)]+self.learningRate*new_q_ac_t_plus_1
            q_tet_t[0,int(action_tet_t_item)]=(1-self.learningRate)*q_tet_t[0,int(action_tet_t_item)]+self.learningRate*new_q_tet_t_plus_1
            q_ap_t[0,int(action_ap_t_item)]=(1-self.learningRate)*q_ap_t[0,int(action_ap_t_item)]+self.learningRate*new_q_ap_t_plus_1
            q_aet_t[0,int(action_aet_t_item)]=(1-self.learningRate)*q_aet_t[0,int(action_aet_t_item)]+self.learningRate*new_q_aet_t_plus_1
            
            y[index,:]=torch.cat((q_cur_t,q_win_t,q_t_t,q_ac_t,q_tet_t,q_ap_t,q_aet_t),1)
           
        
        data_size=len(minibatch)
        validation_pct=0.2
        train_size=math.ceil(data_size*(1-validation_pct))
        
        X_train=X[:train_size]
        y_train=y[:train_size,:]
       
        
        X_test=X[train_size:]
        y_test=y[train_size:,:]
       
        
        self.optimizer=torch.optim.Adam(self.multiServiceModel.parameters(),lr=0.1)
        
        self.optimizer.zero_grad()
        
        l2_reg = None
        for W in self.multiServiceModel.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        
        out = self.multiServiceModel(X_train)
        
        loss=criterion(out,y_train)
      
        
        loss.backward(retain_graph=True)
        
        
        self.optimizer.step()
        
        criterion_val=torch.nn.MSELoss()
        
        self.multiServiceModel.eval()
        
        total_val_loss=torch.tensor([[0]],dtype=torch.float32,device=device)
        
        # %%
        
        out = self.multiServiceModel(X_test)
        
        val_loss=criterion(out,y_test)
        
        
        
        
        print(f'val loss: {val_loss.item():.4f}, train loss: {(loss.item()):.4f}')
        
        self.multiServiceModel.train()
                
    
        
# %%
MultiService=MultiService()
MultiService.multiServiceModel.load_state_dict(torch.load('data/lstm/multiService_lstm_v1_temp_air_olbshomma_noeco_even.pth'))
MultiService.targetMultiServiceModel.load_state_dict(torch.load('data/lstm/multiService_lstm_v1_temp_air_olbshomma_noeco_even.pth'))
# %%

X_US_2=torch.from_numpy(np.load(f'data/lstm/X_US_lstm_v1_eishomma_even_100.npy',allow_pickle=True))
X_LE_2=torch.from_numpy(np.load(f'data/lstm/X_LE_lstm_v1_eishomma_even_100.npy',allow_pickle=True))
X_TE_2=torch.from_numpy(np.load(f'data/lstm/X_TE_lstm_v1_eishomma_even_100.npy',allow_pickle=True))
X_AE_2=torch.from_numpy(np.load(f'data/lstm/X_AE_lstm_v1_eishomma_even_100.npy',allow_pickle=True))


seenTrainingStates=deque(maxlen=100000)

totalEpochRewards=np.full([num_epochs,3],0)

totalAcc=np.full([num_epochs,4],0)

# %%
for epoch in range(num_epochs):
    
    OneHotEncoder=OneHotEncoderClass()
    
    x_cur_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    # x_ls_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    x_ac_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_win_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_t_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_tet_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_aet_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    x_ap_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    # x_lr_t=torch.tensor([[0]],dtype=torch.float32, device=device)
            
    x_tr_t=torch.tensor([[20]],dtype=torch.float32,device=device)
            
    x_ar_t=torch.tensor([[200]],dtype=torch.float32,device=device)
    
    # %%
    
    # X_us_t=user_simulator()
    
    # X_le_t=intensity_simulator()
    # idx_x_le_t=(torch.max(X_le_t,dim=1,keepdim=True)[1]).to(device)
    # MAX_LE=X_le_t[[[0]],idx_x_le_t.item()]
    # MAX_LE=int(MAX_LE)+1
    
    X_te_t=X_TE_2[[epoch],:]
    idx_x_te_t=(torch.max(X_te_t,dim=1,keepdim=True)[1]).to(device)
    MAX_TEMP=X_te_t[[[0]],idx_x_te_t.item()]
    MAX_TEMP=int(MAX_TEMP)+1
    
    
    X_ae_t=X_AE_2[[epoch],:]
    idx_x_ae_t=(torch.max(X_ae_t,dim=1,keepdim=True)[1]).to(device)
    MAX_CO2=X_ae_t[[[0]],idx_x_ae_t.item()]
    MAX_CO2=int(MAX_CO2)+1

    
# %%
    num_corr_light=0
    num_corr_temp=0
    num_corr_air=0   
    
    for step in range(0,steps):
        
        x_us_t=X_US_2[epoch,step]
        x_us_t=x_us_t.reshape((1,-1))
            
        # x_le_t=X_le_t[0,step]
        # x_le_t=x_le_t.reshape((1,-1))
        
        x_te_t=X_te_t[0,step]
        x_te_t=x_te_t.reshape((1,-1))
        
        x_ae_t=X_ae_t[0,step]
        x_ae_t=x_ae_t.reshape((1,-1))
        
        # lightService.setStates(x_us_t, x_lr_t, x_le_t, x_ls_t, x_cur_t, MAX_LE)
        
        tempService.setStates(x_us_t, x_tr_t, x_te_t, x_cur_t, x_ac_t, x_win_t, MAX_TEMP,x_tet_t,x_t_t)
        
        airService.setStates(x_us_t, x_ar_t, x_ae_t, x_ap_t, x_win_t, x_cur_t, x_tr_t, x_te_t, x_aet_t,x_t_t, MAX_CO2)
        
        x_us_t_norm=OneHotEncoder._one_hot_encoder(X_US,x_us_t)
        
        # x_le_t_norm=x_le_t/MAX_LE
        # x_light_t_norm=torch.cat((x_us_t_norm,x_le_t_norm),axis=1)
        
        x_te_t_norm=x_te_t/MAX_TEMP
        x_tr_t_norm=x_tr_t/MAX_TEMP
        x_temp_t_norm=torch.cat((x_us_t_norm,x_te_t_norm),axis=1)
        
        x_ae_t_norm=x_ae_t/MAX_CO2
        x_ar_t_norm=x_ar_t/MAX_CO2
        x_air_t_norm=torch.cat((x_us_t_norm,x_ae_t_norm),axis=1)
        
        x_t_norm=torch.cat((x_us_t_norm,x_te_t_norm,x_tr_t_norm,x_ae_t_norm,x_ar_t_norm),1)

        sigma=torch.rand(1).item()
      
        if sigma>0.1:
            
            # cur_light_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)
            
            # cur_temp_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)
            
            # cur_air_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)

            # win_temp_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)
            
            # win_air_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)

            # t_temp_p_t_norm=torch.tensor([[1]],dtype=torch.float32,device=device)
            
            # t_air_p_t_norm=torch.tensor([[1]],dtype=torch.float32,device=device)
            
            # %%
            
           

            Q=MultiService.multiServiceModel(x_t_norm.float())

            # q_ls=Q[:,:X_LS.shape[1]]
            q_cur=Q[:,:(X_CUR.shape[1])]
            q_win=Q[:,(X_CUR.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1])]
            q_t=Q[:,(X_CUR.shape[1]+X_WIN.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1])]
            
            
            q_ac=Q[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1])]
            q_tet=Q[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1])]
            q_ap=Q[:,(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]):(X_CUR.shape[1]+X_WIN.shape[1]+X_ET.shape[1]+X_AC.shape[1]+X_T.shape[1]+X_AP.shape[1])]
            q_aet=Q[:,-X_T.shape[1]:]
            
            
            # Q_light=torch.cat((q_ls,q_cur),1)
            # Q_temp=torch.cat((q_ac,q_tet,q_cur,q_win,q_t),1)
            # Q_air=torch.cat((q_ap,q_aet,q_cur,q_win,q_t),1)

            # Q_light_cur_t,Q_light_ls_t=lightService.getActions(Q_light)
            
            # Q_temp_cur_t,Q_temp_ac_t,Q_temp_win_t,Q_temp_tet_t,Q_temp_t_t=getActions_temp(Q_temp)
            
            # Q_air_cur_t,Q_air_win_t,Q_air_ap_t,Q_air_aet_t,Q_air_t_t=getActions_air(Q_air)
            
            # %%

            # Q_light_cur_t_norm=Q_light_cur_t/(torch.abs(Q_light_cur_t).sum())

            # Q_temp_cur_t_norm=Q_temp_cur_t/(torch.abs(Q_temp_cur_t).sum())

            # Q_air_cur_t_norm=Q_air_cur_t/(torch.abs(Q_air_cur_t).sum())

            # Q_cur_t_norm=cur_light_p_t_norm*Q_light_cur_t_norm+cur_temp_p_t_norm*Q_temp_cur_t_norm+cur_air_p_t_norm*Q_air_cur_t_norm
            
            idx_x_cur_t_new=(torch.max(q_cur,dim=1,keepdim=True)[1]).to(device)
            x_cur_t_new=X_CUR[[[0]],idx_x_cur_t_new.item()]
            # %%
            
            # x_ls_t_new=torch.max(Q_light_ls_t,dim=1,keepdim=True)[1].to(device)
            
            idx_x_ac_t_new=(torch.max(q_ac,dim=1,keepdim=True)[1]).to(device)
            x_ac_t_new=X_AC[[[0]],idx_x_ac_t_new.item()]
            
            idx_x_ap_t_new=(torch.max(q_ap,dim=1,keepdim=True)[1]).to(device)
            x_ap_t_new=X_AP[[[0]],idx_x_ap_t_new.item()]
            
            idx_x_tet_t_new=(torch.max(q_tet,dim=1,keepdim=True)[1]).to(device)
            x_tet_t_new=X_T[[[0]],idx_x_tet_t_new.item()]


            idx_x_aet_t_new=(torch.max(q_aet,dim=1,keepdim=True)[1]).to(device)
            x_aet_t_new=X_T[[[0]],idx_x_aet_t_new.item()]

            
            # %%

            # Q_temp_win_t_norm=Q_temp_win_t/(torch.abs(Q_temp_win_t).sum())
            # Q_air_win_t_norm=Q_air_win_t/(torch.abs(Q_air_win_t).sum())
            
            # Q_win_t_norm=win_temp_p_t_norm*Q_temp_win_t_norm+win_air_p_t_norm*Q_air_win_t_norm
            
            idx_x_win_t_new=(torch.max(q_win,dim=1,keepdim=True)[1]).to(device)
            x_win_t_new=X_WIN[[[0]],idx_x_win_t_new.item()]
            
            # %%

            # Q_temp_t_t_norm=Q_temp_t_t/(torch.abs(Q_temp_t_t).sum())
            # Q_air_t_t_norm=Q_air_t_t/(torch.abs(Q_air_t_t).sum())
            # Q_t_t_norm=t_temp_p_t_norm*Q_temp_t_t_norm+t_air_p_t_norm*Q_air_t_t_norm
            
            idx_x_t_t_new=(torch.max(q_t,dim=1,keepdim=True)[1]).to(device)
            x_t_t_new=X_ET[[0],idx_x_t_t_new.item()]
            
            

           
        if sigma<0.1:
           
            idx_x_cur_t_new=(torch.randint(0,len(X_CUR[0]),(1,1))).to(device)
            x_cur_t_new=X_CUR[[[0]],idx_x_cur_t_new]
           
            # x_ls_t_new=torch.randint(0,len(X_LS[0]),(1,1)).to(device)
           
            idx_x_ac_t_new=(torch.randint(0,len(X_AC[0]),(1,1))).to(device)
            x_ac_t_new=X_AC[[[0]],idx_x_ac_t_new]

            x_win_t_new=torch.randint(0,len(X_WIN[0]),(1,1)).to(device)

            idx_x_t_t_new=(torch.randint(0,len(X_ET[0]),(1,1))).to(device)
            x_t_t_new=X_ET[[[0]],idx_x_t_t_new]

            idx_x_tet_t_new=(torch.randint(0,len(X_T[0]),(1,1))).to(device)
            x_tet_t_new=X_T[[[0]],idx_x_tet_t_new]

            idx_x_aet_t_new=(torch.randint(0,len(X_T[0]),(1,1))).to(device)
            x_aet_t_new=X_T[[[0]],idx_x_aet_t_new]

            x_ap_t_new=torch.randint(0,len(X_AP[0]),(1,1)).to(device)
            

        # x_lr_t_new=lightService.getIndoorLight(x_ls_t_new,x_cur_t_new)
        x_tr_t_new,x_tr_new_t_2=tempService.getIndoorTemp(x_tr_t,x_ac_t_new, x_cur_t_new, x_win_t_new, x_tet_t_new, x_t_t_new)
        x_ar_t_new,x_ar_new_t_2=airService.getIndoorAir(x_ar_t,x_ap_t_new, x_cur_t_new, x_win_t_new, x_aet_t_new, x_t_t_new)

        # print(f'x_us_t: {x_us_t}')

        # # print(f'x_le_t: {x_le_t}')
        # # print(f'x_lr_t: {x_lr_t}')
        # # print(f'x_lr_t_new: {x_lr_t_new}')
        # # print(f'x_lr_new_t_2: {x_lr_new_t_2}')
        

        # print(f'x_te_t: {x_te_t}')
        # print(f'x_tr_t: {x_tr_t}')
        # print(f'x_tr_t_new: {x_tr_t_new}')
        # print(f'x_tr_new_t_2: {x_tr_new_t_2}')

        # print(f'x_ae_t: {x_ae_t}')
        # print(f'x_ar_t: {x_ar_t}')
        # print(f'x_ar_t_new: {x_ar_t_new}')
        # print(f'x_ar_new_t_2: {x_ar_new_t_2}')
        
        x_us_t_new=copy.deepcopy(x_us_t)
        
        # x_lr_t_new=x_lr_t_new
        x_tr_t_new=copy.deepcopy(x_tr_t_new)
        x_ar_t_new=x_ar_t_new
        
        # x_le_t_new=x_le_t
        x_te_t_new=copy.deepcopy(x_te_t)
        x_ae_t_new=x_ae_t
        
        x_cur_t_new=copy.deepcopy(x_cur_t_new)
        # x_ls_t_new=copy.deepcopy(x_ls_t_new)
        x_ac_t_new=copy.deepcopy(x_ac_t_new)
        x_t_t_new=copy.deepcopy(x_t_t_new)
        x_tet_t_new=copy.deepcopy(x_tet_t_new)
        x_win_t_new=copy.deepcopy(x_win_t_new)
        x_ap_t_new=copy.deepcopy(x_ap_t_new)
        
        x_us_t_new_norm=OneHotEncoder._one_hot_encoder(X_US, x_us_t_new)
        
        # x_le_t_new_norm=x_le_t_new/MAX_LE
        # x_light_t_new_norm=torch.cat((x_us_t_new_norm,x_le_t_new_norm),axis=1)
        
        x_te_t_new_norm=x_te_t_new/MAX_TEMP
        x_tr_t_new_norm=x_tr_t_new/MAX_TEMP
        x_temp_t_new_norm=torch.cat((x_us_t_new_norm,x_te_t_new_norm),axis=1)
        
        x_ae_t_new_norm=x_ae_t_new/MAX_CO2
        x_ar_t_new_norm=x_ar_t_new/MAX_CO2
        x_air_t_new_norm=torch.cat((x_us_t_new_norm,x_ae_t_new_norm),axis=1)
        
        x_t_new_norm=torch.cat((x_us_t_new_norm,x_te_t_new_norm,x_tr_t_new_norm,x_ae_t_new_norm,x_ar_t_new_norm),1)
        
        actions=(x_cur_t_new,x_win_t_new,x_t_t_new,x_ac_t_new,x_tet_t_new,x_ap_t_new,x_aet_t_new)
        
        # lightService.setStates(x_us_t_new, x_lr_t_new, x_le_t_new, x_ls_t_new, x_cur_t_new, MAX_LE)
        tempService.setStates(x_us_t_new, x_tr_t_new, x_te_t_new, x_cur_t_new, x_ac_t_new, x_win_t_new, MAX_TEMP,x_tet_t_new,x_t_t_new)
        airService.setStates(x_us_t_new, x_ar_t_new, x_ae_t_new, x_ap_t_new, x_win_t_new, x_cur_t_new, x_tr_t_new, x_te_t_new, x_aet_t_new,x_t_t_new, MAX_CO2)
        
        # light_reward=lightService.getRewards()
        # light_reward_t=torch.tensor([[light_reward]],dtype=torch.float32,device=device)
        
        temp_reward=tempService.getReward(x_tr_new_t_2,x_tr_t)
        temp_reward_t=torch.tensor([[temp_reward]],dtype=torch.float32,device=device)
        
        air_reward=airService.getReward(x_ar_new_t_2,x_ar_t)
        air_reward_t=torch.tensor([[air_reward]],dtype=torch.float32,device=device)
        
        rewards=(temp_reward,air_reward)

        if temp_reward>0:
            num_corr_temp+=1

        print(f'num_corr_temp: {num_corr_temp}')

        if air_reward>0:
            num_corr_air+=1

        print(f'num_corr_air: {num_corr_air}')

        if temp_reward>0 and air_reward>0:

            totalAcc[epoch,3]+=1

        print(f'three corr: {totalAcc[epoch,3]}')

        if step==steps-1:
            
            totalAcc[epoch,1]=num_corr_temp
            totalAcc[epoch,2]=num_corr_air
        
        # totalEpochRewards[epoch,0]+=light_reward
        totalEpochRewards[epoch,1]+=temp_reward
        totalEpochRewards[epoch,2]+=air_reward

        transitions=(x_t_norm, actions, rewards, x_t_new_norm)
        
        MultiService.updateReplayMemory(transitions)
        
        # MultiService.train()
        
        # x_lr_t=copy.deepcopy(x_lr_t_new)
        x_tr_t=copy.deepcopy(x_tr_t_new)
        x_ar_t=copy.deepcopy(x_ar_t_new)
        
        x_cur_t=copy.deepcopy(x_cur_t_new)
        # x_ls_t=copy.deepcopy(x_ls_t_new)
        x_ac_t=copy.deepcopy(x_ac_t_new)
        x_t_t=copy.deepcopy(x_t_t_new)
        x_tet_t=copy.deepcopy(x_tet_t_new)
        x_aet_t=copy.deepcopy(x_aet_t_new)
        x_win_t=copy.deepcopy(x_win_t_new)
        x_ap_t=copy.deepcopy(x_ap_t_new)
        
        print(f'epoch:{epoch}, step:{step}, reward:{temp_reward_t.item(), air_reward_t.item()}, totalEpochRewards:{totalEpochRewards[epoch,:]}')

        print(f'=================================================================================')
        
        # %%
        
    
#     if epoch%5==0:
#         MultiService.targetMultiServiceModel.load_state_dict(MultiService.multiServiceModel.state_dict())
             
#     if epoch%200==0 and epoch!=0:      
#         torch.save(MultiService.multiServiceModel.state_dict(),f'data/lstm/multiService_lstm_v1_{1+int(epoch/200)}_temp_air_olbshomma_noeco_even.pth')
        

# torch.save(MultiService.multiServiceModel.state_dict(),'data/lstm/multiService_lstm_v1_temp_air_olbshomma_noeco_even.pth')

# np.save(f'data/lstm/multiService_totalEpochRewards_lstm_v1_temp_air_olbshomma_noeco_even.npy',totalEpochRewards)

print(f'100 acc: {np.average(totalAcc,axis=0)}')  
np.save(f'data/lstm/totalAcc_multiService_totalEpochRewards_lstm_v1_temp_air_olbshomma_noeco_even_100.npy',totalAcc) 
        
np.save(f'data/lstm/replayMemory_MultiService_lstm_v1_temp_air_olbshomma_noeco_even_100.npy',MultiService.replayMemory)
# np.save(f'data/lstm/replayMemory_air_lstm_v1_temp_air_olbshomma_noeco_even_100.npy',airService.replayMemory)

        
    
        
        
        
        
        
    
    
 
        

