
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

# from simulated_light import *
from simulated_temp_3_2_tanh import *
from simulated_co2 import *

from qmix_sho_agent_3 import *

from qmix_sho_configuration import *

# %%

random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    
# %%

qmix_sho_agent=QMix()



seenTrainingStates=deque(maxlen=100000)

totalEpochRewards=np.full([num_epochs,3],0)

# %%
for epoch in range(num_epochs):
    
    # print(f'epoch: {epoch}')

    # qmix_sho_agent.on_reset()

    qmix_sho_agent._init_hidden_states(1)
    
    OneHotEncoder=OneHotEncoderClass()
    
    x_cur_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    # x_ls_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    x_ac_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_win_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_tt_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_et_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_at_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    x_ap_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    # x_lr_t=torch.tensor([[0]],dtype=torch.float32, device=device)
            
    x_tr_t=torch.tensor([[20]],dtype=torch.float32,device=device)
            
    x_ar_t=torch.tensor([[200]],dtype=torch.float32,device=device)
    
    # %%
    
    X_us_t=user_simulator()
    
    # X_le_t=intensity_simulator()
    # idx_x_le_t=(torch.max(X_le_t,dim=1,keepdim=True)[1]).to(device)
    # MAX_LE=X_le_t[[[0]],idx_x_le_t.item()]
    # MAX_LE=int(MAX_LE)+1
    
    X_te_t=temp_simulator()
    idx_x_te_t=(torch.max(X_te_t,dim=1,keepdim=True)[1]).to(device)
    MAX_TEMP=X_te_t[[[0]],idx_x_te_t.item()]
    MAX_TEMP=int(MAX_TEMP)+1
    
    
    X_ae_t=co2_simulator()
    idx_x_ae_t=(torch.max(X_ae_t,dim=1,keepdim=True)[1]).to(device)
    MAX_CO2=X_ae_t[[[0]],idx_x_ae_t.item()]
    MAX_CO2=int(MAX_CO2)+1
    
    
# %%

    num_corr_temp=0
    num_corr_air=0   
    
    for step in range(0,steps):
        
        # print(f'epoch: {epoch}')
        
        # print(f'step: {step}')
        
        
        
        if step==0:
            # user states
            x_us_t=X_us_t[0,step]
            x_us_t=x_us_t.reshape(1,-1)
   
            
        else:
            if step%12==0:
                x_us_t=user_simulator()
            x_us_t=x_us_t.reshape((1,-1))
            
            
        # outside environment states
        # x_le_t=X_le_t[0,step]
        # x_le_t=x_le_t.reshape((1,-1))
        
        x_te_t=X_te_t[0,step]
        x_te_t=x_te_t.reshape((1,-1))
        
        x_ae_t=X_ae_t[0,step]
        x_ae_t=x_ae_t.reshape((1,-1))
        
        # service setting environments
        # qmix_sho_agent.light_service.setStates(x_us_t, x_lr_t, x_le_t, x_ls_t, x_cur_t, MAX_LE)
        
        qmix_sho_agent.temp_service.setStates(x_us_t, x_tr_t, x_te_t, x_cur_t, x_ac_t, x_win_t, MAX_TEMP,x_tt_t, x_et_t)
        
        qmix_sho_agent.air_service.setStates(x_us_t, x_ar_t, x_ae_t, x_ap_t, x_win_t, x_cur_t, x_tr_t, x_te_t, x_at_t, x_et_t, MAX_CO2)
        
        
        # data normalization
        x_us_t_norm=OneHotEncoder._one_hot_encoder(X_US,x_us_t)
        
        # data normalization: light service
        # x_le_t_norm=x_le_t/MAX_LE
        # x_light_t_norm=torch.cat((x_us_t_norm,x_le_t_norm),axis=1)
        
        # data normalization: temperature service
        x_te_t_norm=x_te_t/MAX_TEMP
        x_tr_t_norm=x_tr_t/MAX_TEMP
        x_temp_t_norm=torch.cat((x_us_t_norm,x_te_t_norm,x_tr_t_norm),axis=1)
        
        # data normalization: air quality service
        x_ae_t_norm=x_ae_t/MAX_CO2
        x_ar_t_norm=x_ar_t/MAX_CO2
        x_air_t_norm=torch.cat((x_us_t_norm,x_ae_t_norm,x_ar_t_norm),axis=1)
        
        x_t_norm=torch.cat((x_us_t_norm, x_te_t_norm, x_tr_t_norm, x_ae_t_norm, x_ar_t_norm),axis=1)

        # ations_t_minus_one=qmix_sho_agent.replayMemory[-1][1]

        # x_cur_t_minus_one,x_ac_t_minus_one,x_tt_t_minus_one,x_ap_t_minus_one,x_at_t_minus_one,x_win_t_minus_one,x_et_t_minus_one=ations_t_minus_one

        # temp_actions_minus_one=torch.cat((x_cur_t_minus_one,x_ac_t_minus_one,x_win_t_minus_one,x_tt_t_minus_one,x_et_t_minus_one),1)
        # air_actions_minus_one=torch.cat((x_cur_t_minus_one,x_win_t_minus_one,x_ap_t_minus_one, x_at_t_minus_one, x_et_t_minus_one),1)

        x_ac_t_new, x_tt_t_new, x_ap_t_new, x_at_t_new, x_cur_t_new, x_win_t_new, x_et_t_new=qmix_sho_agent.act(x_t_norm, True)

        qmix_sho_agent.decay_epsilon_greddy(step)

        
        # update the indoor states
        # x_lr_t_new=qmix_sho_agent.light_service.getIndoorLight(x_ls_t_new,x_cur_t_new)
        x_tr_t_new,x_tr_new_t_2=qmix_sho_agent.temp_service.getIndoorTemp(x_tr_t,x_ac_t_new, x_cur_t_new, x_win_t_new, x_tt_t_new, x_et_t_new)
        x_ar_t_new,x_ar_new_t_2=qmix_sho_agent.air_service.getIndoorAir(x_ar_t,x_ap_t_new, x_cur_t_new, x_win_t_new, x_at_t_new, x_et_t_new)

        print(f'x_us_t: {x_us_t}')

        # print(f'x_le_t: {x_le_t}')
        # print(f'x_lr_t: {x_lr_t}')
        # print(f'x_lr_t_new: {x_lr_t_new}')
        # print(f'x_lr_new_t_2: {x_lr_new_t_2}')
        

        print(f'x_te_t: {x_te_t}')
        print(f'x_tr_t: {x_tr_t}')
        print(f'x_tr_t_new: {x_tr_t_new}')
        print(f'x_tr_new_t_2: {x_tr_new_t_2}')

        print(f'x_ae_t: {x_ae_t}')
        print(f'x_ar_t: {x_ar_t}')
        print(f'x_ar_t_new: {x_ar_t_new}')
        print(f'x_ar_new_t_2: {x_ar_new_t_2}')

        
        # update the current state values
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
        x_tt_t_new=copy.deepcopy(x_tt_t_new)
        x_at_t_new=copy.deepcopy(x_at_t_new)
        x_et_t_new=copy.deepcopy(x_et_t_new)
        x_win_t_new=copy.deepcopy(x_win_t_new)
        x_ap_t_new=copy.deepcopy(x_ap_t_new)
        
        # new data normalization
        x_us_t_new_norm=OneHotEncoder._one_hot_encoder(X_US, x_us_t_new)
        
        # data normalization: light service
        # x_le_t_new_norm=x_le_t_new/MAX_LE
        # x_light_t_new_norm=torch.cat((x_us_t_new_norm,x_le_t_new_norm),axis=1)
        
        # data normalization: temperature service
        x_te_t_new_norm=x_te_t_new/MAX_TEMP
        x_tr_t_new_norm=x_tr_t_new/MAX_TEMP
        x_temp_t_new_norm=torch.cat((x_us_t_new_norm,x_te_t_new_norm,x_tr_t_new_norm),axis=1)
        
        # data normalization: air quality service
        x_ae_t_new_norm=x_ae_t_new/MAX_CO2
        x_ar_t_new_norm=x_ar_t_new/MAX_CO2
        x_air_t_new_norm=torch.cat((x_us_t_new_norm,x_ae_t_new_norm,x_ar_t_new_norm),axis=1)
        
        
        x_t_new_norm=torch.cat((x_us_t_new_norm, x_te_t_new_norm, x_tr_t_new_norm, x_ae_t_new_norm, x_ar_t_new_norm),axis=1)

        
        # light_actions=(x_cur_t_new,x_ls_t_new)
        temp_actions=(x_cur_t_new,x_ac_t_new,x_win_t_new,x_tt_t_new,x_et_t_new)
        air_actions=(x_cur_t_new,x_win_t_new,x_ap_t_new, x_at_t_new, x_et_t_new)

        actions=(x_cur_t_new,x_ac_t_new,x_tt_t_new,x_ap_t_new,x_at_t_new,x_win_t_new,x_et_t_new)
       
        # set states in each service
        # qmix_sho_agent.light_service.setStates(x_us_t_new, x_lr_t_new, x_le_t_new, x_ls_t_new, x_cur_t_new, MAX_LE)
        qmix_sho_agent.temp_service.setStates(x_us_t_new, x_tr_t_new, x_te_t_new, x_cur_t_new, x_ac_t_new, x_win_t_new, MAX_TEMP,x_tt_t_new,x_et_t_new)
        qmix_sho_agent.air_service.setStates(x_us_t_new, x_ar_t_new, x_ae_t_new, x_ap_t_new, x_win_t_new, x_cur_t_new, x_tr_t_new, x_te_t_new, x_at_t_new, x_et_t_new, MAX_CO2)
        
        # get reward values
        # light_reward=qmix_sho_agent.light_service.getRewards()
        # light_reward_t=torch.tensor([[light_reward]],dtype=torch.float32,device=device)
        
        temp_reward=qmix_sho_agent.temp_service.getReward(x_tr_new_t_2,x_tr_t)
        temp_reward_t=torch.tensor([[temp_reward]],dtype=torch.float32,device=device)
        
        air_reward=qmix_sho_agent.air_service.getReward(x_ar_new_t_2,x_ar_t)
        air_reward_t=torch.tensor([[air_reward]],dtype=torch.float32,device=device)

        reward_t=(temp_reward_t+air_reward_t)/2
        # reward_t=(light_reward_t+temp_reward_t+air_reward_t)/3
       

        
        # calculate the epoch total rewards
        # totalEpochRewards[epoch,0]+=light_reward
        totalEpochRewards[epoch,1]+=temp_reward
        totalEpochRewards[epoch,2]+=air_reward

        if temp_reward>0:
            num_corr_temp+=1

        print(f'num_corr_temp: {num_corr_temp}')

        if air_reward>0:
            num_corr_air+=1

        print(f'num_corr_air: {num_corr_air}')

        # print(f'totalEpochRewards: {totalEpochRewards[epoch,:]}')
        
        # calculate the transitions
        # light_transition=(x_light_t_norm,light_actions,light_reward_t,x_light_t_new_norm)
        temp_transition=(x_temp_t_norm,temp_actions,temp_reward_t,x_temp_t_new_norm)
        air_transition=(x_air_t_norm, air_actions, air_reward_t,x_air_t_new_norm)

        done=torch.tensor([[0]],dtype=torch.float32)

        if step==steps-1:
            done=torch.tensor([[1]],dtype=torch.float32)

        transition=(x_t_norm, actions, reward_t, x_t_new_norm, done)
        
       
        # lightService.updateReplayMemory(light_transition)
        # tempService.updateReplayMemory(temp_transition)
        # airService.updateReplayMemory(air_transition)

        qmix_sho_agent.updateReplayMemory(transition)        
        
        

        
        # x_lr_t=copy.deepcopy(x_lr_t_new)
        x_tr_t=copy.deepcopy(x_tr_t_new)
        # x_tr_t=torch.tensor([[12]], dtype=torch.float32)
        x_ar_t=copy.deepcopy(x_ar_t_new)
        
        x_cur_t=copy.deepcopy(x_cur_t_new)
        # x_ls_t=copy.deepcopy(x_ls_t_new)
        x_ac_t=copy.deepcopy(x_ac_t_new)
        x_tt_t=copy.deepcopy(x_tt_t_new)
        x_et_t=copy.deepcopy(x_et_t_new)
        x_at_t=copy.deepcopy(x_at_t_new)
        x_win_t=copy.deepcopy(x_win_t_new)
        x_ap_t=copy.deepcopy(x_ap_t_new)
        
        print(f'epoch:{epoch}, step:{step}, temp:{temp_reward_t}, air:{air_reward_t}, totalEpochRewards:{totalEpochRewards[epoch,:]}')

        print(f'=================================================================================')
        
        qmix_sho_agent.train()  
        qmix_sho_agent.update_targets(epoch)
        
    
             
    if epoch%200==0 and epoch!=0:      
        # torch.save(qmix_sho_agent.light_service.agents.state_dict(),f'data/lstm/lightService_lstm_v1_{1+int(epoch/100)}_mlt_qmix.pth')
        torch.save(qmix_sho_agent.temp_service.agents.state_dict(),f'data/lstm/tempService_lstm_v1_{1+int(epoch/200)}_temp_air_qmixshomma_3.pth')
        torch.save(qmix_sho_agent.air_service.agents.state_dict(),f'data/lstm/airService_lstm_v1_{1+int(epoch/200)}_temp_air_qmixshomma_3.pth')
        # torch.save(qmix_sho_agent.common_service.agents.state_dict(),f'data/lstm/commonService_lstm_v1_{1+int(epoch/100)}_mlt_qmix.pth')
        torch.save(qmix_sho_agent.qmixer.state_dict(), f'data/lstm/qmixer_lstm_v1_{1+int(epoch/200)}_temp_air_qmixshomma_3.pth')

        np.save(f'data/lstm/totalEpochRewards_lstm_v1_{1+int(epoch/200)}_temp_air_qmixshomma_3.npy',totalEpochRewards)
        

# torch.save(qmix_sho_agent.light_service.agents.state_dict(),f'data/lstm/lightService_lstm_v1_mlt_qmix.pth')
torch.save(qmix_sho_agent.temp_service.agents.state_dict(),f'data/lstm/tempService_lstm_v1_temp_air_qmixshomma_3.pth')
torch.save(qmix_sho_agent.air_service.agents.state_dict(),f'data/lstm/airService_lstm_v1_temp_air_qmixshomma_3.pth')
# torch.save(qmix_sho_agent.common_service.agents.state_dict(),f'data/lstm/commonService_lstm_v1_mlt_qmix.pth')
torch.save(qmix_sho_agent.qmixer.state_dict(), f'data/lstm/qmixer_lstm_v1_temp_air_qmixshomma_3.pth')

np.save(f'data/lstm/totalEpochRewards_lstm_v1_temp_air_qmixshomma_3.npy',totalEpochRewards)

        
        