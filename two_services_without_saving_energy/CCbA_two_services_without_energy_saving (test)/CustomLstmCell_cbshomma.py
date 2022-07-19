import torch 

import torch.nn as nn

from torch.nn.parameter import Parameter

# %%
# a custom lstm cell considering the hidden states from another agent
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size,hidden_size=400):
        super(CustomLSTMCell,self).__init__()
        
        self.input_size=input_size
        
        self.hidden_size=hidden_size
        
        self.w_f_light=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size), dtype=torch.float32))
        self.b_f_light=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_f_light_temp=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_f_light_temp=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_f_light_air=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_f_light_air=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        
        self.w_i_light=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_i_light=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_i_light_temp=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_i_light_temp=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_i_light_air=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_i_light_air=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        
        self.w_c_light=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_c_light=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_c_light_temp=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_c_light_temp=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_c_light_air=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_c_light_air=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        
        self.w_o_light=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_o_light=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_o_light_temp=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_o_light_temp=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        self.w_o_light_air=Parameter(torch.randn((self.hidden_size,self.hidden_size+self.input_size),dtype=torch.float32))
        self.b_o_light_air=Parameter(torch.randn((self.hidden_size),dtype=torch.float32))
        
        # %%
        
        # self.w_f_temp=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_f_temp=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_f_temp_light=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_f_temp_light=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_f_temp_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_f_temp_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # self.w_i_temp=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_i_temp=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_i_temp_light=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_i_temp_light=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_i_temp_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_i_temp_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # self.w_c_temp=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_c_temp=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_c_temp_light=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_c_temp_light=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_c_temp_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_c_temp_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # self.w_o_temp=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_o_temp=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_o_temp_light=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_o_temp_light=Parameter(torch.randn(torch.randn(self.hidden_size)))
        # self.w_o_temp_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_o_temp_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # # %%
        
        # self.w_f_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_f_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # self.w_i_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_i_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # self.w_c_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_c_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # self.w_o_air=Parameter(torch.randn(self.hidden_size,self.hidden_size+self.input_size))
        # self.b_o_air=Parameter(torch.randn(torch.randn(self.hidden_size)))
        
        # %%
    def forward(self, y, h):
        
        # outputs=torch.zeros(y.shape[0],self.num_output_features,dtype=torch.float32)
        
        x_light=y
        x_light=x_light.float()
        
        # h=((h_light,c_light),(h_temp,c_temp),(h_air,c_air))
        
        h_light,c_light=h[0]
        h_light=h_light.float()
        c_light=c_light.float()
        
        h_temp,c_temp=h[1]
        h_temp=h_temp.float()
        c_temp=c_temp.float()
        
        # h_air,c_air=h[2]
        # h_air=h_air.float()
        # c_air=c_air.float()
        
        
        sigmoid=torch.nn.Sigmoid()
        tanh=torch.nn.Tanh()
        
        # %%
        # h_light=torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_light=torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        # # %%
        # h_temp=torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_temp=torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        # # %%
        # h_air=torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_air=torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        # %%
        f_light=sigmoid(torch.mm(torch.cat((h_light,x_light),1),torch.transpose(self.w_f_light,0,1))+self.b_f_light)
        # print(f'f_light: {f_light}')
        i_light=sigmoid(torch.mm(torch.cat((h_light,x_light),1),torch.transpose(self.w_i_light,0,1))+self.b_i_light)
        # print(f'i_light: {i_light}')
        c_hat_light=tanh(torch.mm(torch.cat((h_light,x_light),1),torch.transpose(self.w_c_light,0,1))+self.b_c_light)
        # print(f'c_hat_light: {c_hat_light}')
        
        f_light_temp=sigmoid(torch.mm(torch.cat((h_temp,x_light),1),torch.transpose(self.w_f_light_temp,0,1))+self.b_f_light_temp)
        # print(f'f_light_temp: {f_light_temp}')
        i_light_temp=sigmoid(torch.mm(torch.cat((h_temp,x_light),1),torch.transpose(self.w_i_light_temp,0,1))+self.b_i_light_temp)
        # print(f'i_light_temp: {i_light_temp}')
        c_hat_light_temp=tanh(torch.mm(torch.cat((h_temp,x_light),1),torch.transpose(self.w_c_light_temp,0,1))+self.b_c_light_temp)
        # print(f'c_hat_light_temp: {c_hat_light_temp}')
        
        # f_light_air=sigmoid(torch.mm(torch.cat((h_air,x_light),1),torch.transpose(self.w_f_light_air,0,1))+self.b_f_light_air)
        # # print(f'f_light_air: {f_light_air}')
        # i_light_air=sigmoid(torch.mm(torch.cat((h_air,x_light),1),torch.transpose(self.w_i_light_air,0,1))+self.b_i_light_air)
        # # print(f'i_light_air: {i_light_air}')
        # c_hat_light_air=tanh(torch.mm(torch.cat((h_air,x_light),1),torch.transpose(self.w_c_light_air,0,1))+self.b_c_light_air)
        # print(f'c_hat_light_air: {c_hat_light_air}')
        
        c_light_0=torch.mul(f_light,c_light)+torch.mul(i_light,c_hat_light) 
        # print(f'c_light_0: {c_light_0}')
        c_light_1=torch.mul(f_light_temp,c_light)+torch.mul(i_light_temp,c_hat_light_temp)
        # c_light_1=torch.mul(f_light_temp,c_light)
        # print(f'c_light_1: {c_light_1}')
        # c_light_2=torch.mul(f_light_air,c_light)+torch.mul(i_light_air,c_hat_light_air)
        # c_light_2=torch.mul(f_light_air,c_light)
        # print(f'c_light_2: {c_light_2}')
        c_light=c_light_0+c_light_1
        # c_light=c_light_0
        # print(f'c_light: {c_light}')
        
        o_light=sigmoid(torch.mm(torch.cat((h_light,x_light),1),torch.transpose(self.w_o_light,0,1))+self.b_o_light)
        # print(f'o_light: {o_light}')
        o_light_temp=sigmoid(torch.mm(torch.cat((h_temp,x_light),1),torch.transpose(self.w_o_light_temp,0,1))+self.b_o_light_temp)
        # print(f'o_light_temp: {o_light_temp}')
        # o_light_air=sigmoid(torch.mm(torch.cat((h_air,x_light),1),torch.transpose(self.w_o_light_air,0,1))+self.b_o_light_air)
        # print(f'o_light_air: {o_light_air}')
        
        o_light=o_light+o_light_temp
        # print(f'o_light: {o_light}')
        
        h_light=torch.mul(o_light,tanh(c_light))
        
        
        # %%
        
        return (h_light, c_light)
        
        
        
        
        
        
        