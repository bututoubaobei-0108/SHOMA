
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

from simulated_light import *
from simulated_temp_3_2_tanh import *
from simulated_co2 import *

from configurations import *
from PRIORITIES_Model import *


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

class CalculationPriorities_cell:
    
    def __init__(self,num_inputs, num_outputs,
                 replayMemorySize=1000000,
                 minibatchSize=128,discount=0.1, learningRate=0.9):
        
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.replayMemorySize=replayMemorySize
        self.replayMemory=deque(maxlen=self.replayMemorySize)
        self.minibatchSize=minibatchSize
        self.discount=discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        
        self.calPrioModel=self.createModel().to(device)
        
        self.optimizer=torch.optim.Adam(self.calPrioModel.parameters(),lr=0.1)
        
        self.targetCalPrioModel=self.createModel().to(device)
        self.targetCalPrioModel.load_state_dict(self.calPrioModel.state_dict())
        
        self.targetCalPrioModel.eval()
        
    def updateReplayMemory(self,transition):
        self.replayMemory.append(transition)
        
    # def setStates(self,x_us_t,x_le_t,x_te_t,x_ae_t,x_cur_t,x_win_t,MAX_LE,MAX_TEMP,MAX_CO2):
    #     self.x_us_t=x_us_t
    #     self.x_le_t=x_le_t
    #     self.x_te_t=x_te_t
    #     self.x_ae_t=x_ae_t
    #     self.x_cur_t=x_cur_t
    #     self.x_win_t=x_win_t
    
    #     self.MAX_LE=MAX_LE
    #     self.MAX_TEMP=MAX_TEMP
    #     self.MAX_CO2=MAX_CO2
        
    # def getStates(self):
    #     pass
    
    def setReward(self,reward):
        self.reward=reward
        
        return self.reward
    
    def getIndoor(self):
        pass
        
    def createModel(self):
        calPrioModel=PriorityModel(self.num_inputs, self.num_outputs, 200)
        return calPrioModel
    
    def getPriorities(self,X_t_norm):
        
        Q=self.calPrioModel(X_t_norm.float())
        
        p=torch.tensor([[0,0,0]],dtype=torch.float32,device=device)
        
        for i in range(self.num_outputs):
        
            p[0,i]=Q[0,[i]]/(torch.abs(Q[0,[0]])+torch.abs(Q[0,[1]]))
            
            
        return p
    
    def train(self,epoch):
        if len(self.replayMemory)<self.minibatchSize:
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
        
        q_list_t=self.calPrioModel(states_list_t.float()).reshape((self.minibatchSize,-1))
        
        q_0_list_t=q_list_t[:,[0]]
        q_1_list_t=q_list_t[:,[1]]
        
        if self.num_outputs==3:
            q_2_list_t=q_list_t[:,[2]]
        
        states_list_tt_plus_1=[transition[3] for transition in minibatch]
        states_list_t_plus_1=None
        for i in range(len(states_list_tt_plus_1)):
            if i==0:
                states_list_t_plus_1=states_list_tt_plus_1[i]
            else:
                states_list_ttt_plus_1=states_list_tt_plus_1[i]
                states_list_t_plus_1=torch.cat((states_list_t_plus_1,states_list_ttt_plus_1),axis=0)
                
        states_list_t_plus_1=states_list_t_plus_1.reshape((self.minibatchSize,-1))
        
        q_list_t_plus_1=self.targetCalPrioModel(states_list_t_plus_1.float())[:,:].detach().reshape((self.minibatchSize,-1))
        q_0_list_t_plus_1=q_list_t_plus_1[:,[0]]
        q_1_list_t_plus_1=q_list_t_plus_1[:,[1]]
        
        if self.num_outputs==3:
            q_2_list_t_plus_1=q_list_t_plus_1[:,[2]]
        
        X=None
        Y=None
        
        for index, (states_t,actions,reward_t,states_t_plus_1) in enumerate(minibatch):
            # if index!=23
            if self.num_outputs==2:
                reward_0_t,reward_1_t=reward_t
                # reward_0_t_item=reward_0_t
                # reward_1_t_item=reward_1_t
                reward_t_item=reward_0_t+reward_1_t
                
            elif self.num_outputs==3:
                reward_0_t,reward_1_t,reward_2_t=reward_t
                # reward_0_t_item=reward_0_t
                # reward_1_t_item=reward_1_t
                # reward_2_t_item=reward_2_t
                reward_t_item=reward_0_t+reward_1_t+reward_2_t
                
            
                
            max_q_0_t_plus_1=torch.max(q_0_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_0_t_plus_1=reward_t_item+self.discount*max_q_0_t_plus_1
            
            max_q_1_t_plus_1=torch.max(q_1_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_1_t_plus_1=reward_t_item+self.discount*max_q_1_t_plus_1
            
            if self.num_outputs==3:
                max_q_2_t_plus_1=torch.max(q_2_list_t_plus_1[index],dim=0,keepdim=True)[0]
                new_q_2_t_plus_1=reward_t_item+self.discount*max_q_2_t_plus_1
            
            q_0_t=q_0_list_t[index,:]
            q_0_t=q_0_t.reshape(1,-1)
            
            q_1_t=q_1_list_t[index,:]
            q_1_t=q_1_t.reshape(1,-1)
            
            if self.num_outputs==3:
                q_2_t=q_2_list_t[index,:]
                q_2_t=q_2_t.reshape(1,-1)
            
            # action_0_t, action_1_t=actions
            
            # action_0_t_item=action_0_t.item()
            # action_1_t_item=action_1_t.item()
            
            q_0_t=(1-self.learningRate)*q_0_t+self.learningRate*new_q_0_t_plus_1
            q_1_t=(1-self.learningRate)*q_1_t+self.learningRate*new_q_1_t_plus_1
            
            if self.num_outputs==3:
                q_2_t=(1-self.learningRate)*q_2_t+self.learningRate*new_q_2_t_plus_1
            
            if index==0:
                X=copy.deepcopy(states_t)
                if self.num_outputs==2:
                    Y=torch.cat((q_0_t,q_1_t),axis=1)
                elif self.num_outputs==3:
                    Y=torch.cat((q_0_t,q_1_t,q_2_t),axis=1)
            
            else:
                X=torch.cat((X,states_t),axis=0)
                if self.num_outputs==2:
                    q_t=torch.cat((q_0_t,q_1_t),axis=1)
                elif self.num_outputs==3:
                    q_t=torch.cat((q_0_t,q_1_t,q_2_t),axis=1)
                    
                Y=torch.cat((Y,q_t),axis=0)
                
        data_size=len(X)
        validation_pct=0.2
        
        train_size=math.ceil(data_size*(1-validation_pct))
        
        X_train,Y_train=X[:train_size,:],Y[:train_size,:]
        X_test,Y_test=X[train_size:,:],Y[train_size:,:]

        if epoch>=0:
            self.optimizer=torch.optim.Adam(self.calPrioModel.parameters())
            
            
        least_val_loss=0
        total_patience=50
        patience=0
        
        # for i in range(self.num_epochs):
            
        outputs=self.calPrioModel(X_train.float()).reshape((train_size,-1))
        
        criterion=torch.nn.MSELoss()
        
        # outputs=torch.cat((outputs_ls,outputs_cur),axis=1)
        
        loss=criterion(outputs,Y_train)
        
        # loss.requres_grad = True
        # backward pass
        self.optimizer.zero_grad()
        
        # back gradient
        loss.backward(retain_graph=True)
        
        # weights updating
        self.optimizer.step()
        
       
        criterion_val=torch.nn.MSELoss()
        
        
        self.calPrioModel.eval()
        
        total_val_loss=torch.tensor([[0]],dtype=torch.float32,device=device)
        
        with torch.no_grad():
            n_correct=0
            n_samples=0
            # for index in range(len(X_test)):
            x_test=X_test
            y_test=Y_test
            
            outputs=self.calPrioModel(x_test.float()).reshape((x_test.shape[0],-1))

            val_loss=criterion_val(outputs,y_test)
                
            print(f'val loss: {val_loss.item():.4f}, train loss: {(loss.item()):.4f}')
            
            if epoch==0:
                least_val_loss=total_val_loss
            else:
                if least_val_loss>total_val_loss:
                    least_val_loss=total_val_loss
                else:
                    patience+=1
            
            if patience==50:
                torch.save(self.calPrioModel.state_dict(),dirName+f'/model_structure_2_v3_2.pth')
                print("end training")
                return
            
        # print('==========================test end=====================')
        
        self.calPrioModel.train()
        
        return minibatch
    