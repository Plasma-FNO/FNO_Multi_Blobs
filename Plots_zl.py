#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum
FNO modelled over the MHD data built using JOREK for multi-blob diffusion. 

Multivariable FNO
"""
# %%
configuration = {"Case": 'Multi-Blobs',
                 "Field": 'rho, Phi, T',
                 "Field_Mixing": 'Channel',
                 "Type": '2D Time',
                 "Epochs": 0,
                 "Batch Size": 4,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No',
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 10,    
                 "T_out": 40,
                 "Step": 5,
                 "Modes": 16,
                 "Width_time":32, #FNO
                 "Width_vars": 0, #U-Net
                 "Variables":3, 
                 "Noise":0.0, 
                 "Loss Function": 'LP Loss',
                 "Spatial Resolution": 1,
                 "Temporal Resolution": 1,
                #  "UQ": 'Dropout',
                #  "Dropout Rate": 0.9
                 }

# %%
from simvue import Run
run = Run(mode='disabled')
run.init(folder="/FNO_MHD", tags=['FNO', 'MHD', 'JOREK', 'Multi-Blobs', 'MultiVariable'], metadata=configuration)

# %% 
import os 
CODE = ['FNO_multiple.py']
# Save code files
for code_file in CODE:
    if os.path.isfile(code_file):
        run.save(code_file, 'code')
    elif os.path.isdir(code_file):
        run.save_directory(code_file, 'code', 'text/plain', preserve_path=True)
    else:
        print('ERROR: code file %s does not exist' % code_file)

# %% 

import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 

import operator
from functools import reduce
from functools import partial
from collections import OrderedDict

import time 
from timeit import default_timer
from tqdm import tqdm 

torch.manual_seed(0)
np.random.seed(0)

# %% 
path = os.getcwd()
data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# model_loc = os.path.dirname(os.path.dirname(os.getcwd()))
file_loc = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
##################################
#Normalisation Functions 
##################################


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x


    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

# #normalization, rangewise but single value. 


# normalization, rangewise but single value.
class MinMax_Normalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        min_u = torch.min(x[:, 0, :, :, :])
        max_u = torch.max(x[:, 0, :, :, :])

        self.a_u = (high - low) / (max_u - min_u)
        self.b_u = -self.a_u * max_u + high

        min_v = torch.min(x[:, 1, :, :, :])
        max_v = torch.max(x[:, 1, :, :, :])

        self.a_v = (high - low) / (max_v - min_v)
        self.b_v = -self.a_v * max_v + high

        min_p = torch.min(x[:, 2, :, :, :])
        max_p = torch.max(x[:, 2, :, :, :])

        self.a_p = (high - low) / (max_p - min_p)
        self.b_p = -self.a_p * max_p + high

        print(min_u, max_u, min_v, max_v, min_p, max_p)

    def encode(self, x):
        s = x.size()

        u = x[:, 0, :, :, :]
        u = self.a_u * u + self.b_u

        v = x[:, 1, :, :, :]
        v = self.a_v * v + self.b_v

        p = x[:, 2, :, :, :]
        p = self.a_p * p + self.b_p

        x = torch.stack((u, v, p), dim=1)

        return x

    def decode(self, x):
        s = x.size()

        u = x[:, 0, :, :, :]
        u = (u - self.b_u) / self.a_u

        v = x[:, 1, :, :, :]
        v = (v - self.b_v) / self.a_v

        p = x[:, 2, :, :, :]
        p = (p - self.b_p) / self.a_p

        x = torch.stack((u, v, p), dim=1)

        return x

    def cuda(self):
        self.a_u = self.a_u.cuda()
        self.b_u = self.b_u.cuda()

        self.a_v = self.a_v.cuda()
        self.b_v = self.b_v.cuda()

        self.a_p = self.a_p.cuda()
        self.b_p = self.b_p.cuda()

    def cpu(self):
        self.a_u = self.a_u.cpu()
        self.b_u = self.b_u.cpu()

        self.a_v = self.a_v.cpu()
        self.b_v = self.b_v.cpu()

        self.a_p = self.a_p.cpu()
        self.b_p = self.b_p.cpu()


        

# #normalization, rangewise but across the full domain 
# class MinMax_Normalizer(object):
#     def __init__(self, x, low=-1.0, high=1.0):
#         super(MinMax_Normalizer, self).__init__()
#         mymin = torch.min(x)
#         mymax = torch.max(x)

#         self.a = (high - low)/(mymax - mymin)
#         self.b = -self.a*mymax + high

#     def encode(self, x):
#         s = x.size()
#         x = x.reshape(s[0], -1)
#         x = self.a*x + self.b
#         x = x.view(s)
#         return x

#     def decode(self, x):
#         s = x.size()
#         x = x.reshape(s[0], -1)
#         x = (x - self.b)/self.a
#         x = x.view(s)
#         return x

#     def cuda(self):
#         self.a = self.a.cuda()
#         self.b = self.b.cuda()

#     def cpu(self):
#         self.a = self.a.cpu()
#         self.b = self.b.cpu()


# %%
##################################
# Loss Functions
##################################

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# %% 
x_grid = np.arange(0, 106)
y_grid = np.arange(0, 106)
S = 106 #Grid Size
size_x = S
size_y = S


modes = configuration['Modes']
width_time = configuration['Width_time']
width_vars = configuration['Width_vars']
output_size = configuration['Step']

batch_size = configuration['Batch Size']

batch_size2 = batch_size

t1 = default_timer()

T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']
num_vars = configuration['Variables']


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# %%
################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, num_vars):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.num_vars = num_vars

        self.scale = (1 / (in_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bivxy,iovxy->bovxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.num_vars, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_vars):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_vars = num_vars

        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.num_vars)
        self.mlp = MLP(self.width, self.width, self.width)
        self.w = nn.Conv3d(self.width, self.width, 1)
        self.b = nn.Conv3d(2, self.width, 1)

    def forward(self, x, grid):
        x1 = self.conv(x)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x3 = self.b(grid)
        x = x1 + x2 + x3
        x = F.gelu(x)
        return x

# %%
class FNO_multi(nn.Module):
    def __init__(self,T_in, step, num_vars, modes1, modes2, width_vars, width_time):
        super(FNO_multi, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
        """
        self.T_in = T_in
        self.step = step 
        self.num_vars = num_vars
        self.modes1 = modes1
        self.modes2 = modes2
        self.width_vars = width_vars
        self.width_time = width_time

        self.fc0_time = nn.Linear(self.T_in + 2, self.width_time)

        # self.padding = 8 # pad the domain if input is non-periodic

        self.f0 = FNO2d(self.modes1, self.modes2, self.width_time, self.num_vars)
        self.f1 = FNO2d(self.modes1, self.modes2, self.width_time, self.num_vars)
        self.f2 = FNO2d(self.modes1, self.modes2, self.width_time, self.num_vars)
        self.f3 = FNO2d(self.modes1, self.modes2, self.width_time, self.num_vars)
        self.f4 = FNO2d(self.modes1, self.modes2, self.width_time, self.num_vars)
        self.f5 = FNO2d(self.modes1, self.modes2, self.width_time, self.num_vars)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = nn.Identity()

        self.fc1_time = nn.Linear(self.width_time, 256)
        self.fc2_time = nn.Linear(256, self.step)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0_time(x)
        x = x.permute(0, 4, 1, 2, 3)
        grid = grid.permute(0, 4, 1, 2, 3)
        # x = self.dropout(x)

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x0 = self.f0(x, grid)
        x = self.f1(x0, grid)
        x = self.f2(x, grid) + x0
        x1 = self.f3(x, grid)
        x = self.f4(x1, grid)
        x = self.f5(x, grid) + x1

        # x = self.dropout(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic

        x = x.permute(0, 2, 3, 4, 1)
        x = x

        x = self.fc1_time(x)
        x = F.gelu(x)
        # x = self.dropout(x)
        x = self.fc2_time(x)

        return x

    # Using x and y values from the simulation discretisation
    def get_grid(self, shape, device):
        batchsize, num_vars, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(9.5, 10.5, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, num_vars, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(-0.5, 0.5, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat([batchsize, num_vars, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    ## Arbitrary grid discretisation
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


# %%

################################################################
# Loading Data 
################################################################

# %%
# data = data_loc + '/Data/MHD_multi_blobs.npz'
data = data_loc + '/Data/FNO_MHD_data_multi_blob_2000_T50.npz' #2000 simulation dataset
# data = data_loc + '/Data/FNO_MHD_data_multi_blob_2000_T50.npz'# new dataset

# %%
field = configuration['Field']
dims = ['rho', 'Phi', 'T']
num_vars = configuration['Variables']

u_sol = np.load(data)['rho'].astype(np.float32)  / 1e20
v_sol = np.load(data)['Phi'].astype(np.float32)  / 1e5
p_sol = np.load(data)['T'].astype(np.float32)    / 1e6

u_sol = np.nan_to_num(u_sol)
v_sol = np.nan_to_num(v_sol)
p_sol = np.nan_to_num(p_sol)

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)

v = torch.from_numpy(v_sol)
v = v.permute(0, 2, 3, 1)

p = torch.from_numpy(p_sol)
p = p.permute(0, 2, 3, 1)

t_res = configuration['Temporal Resolution']
x_res = configuration['Spatial Resolution']
uvp = torch.stack((u,v,p), dim=1)[:,::t_res]
# uvp = np.delete(uvp, (153, 229), axis=0)  # Outlier T values
uvp = np.delete(uvp, (11, 160, 222, 273, 303, 357, 620, 797, 983, 1275, 1391, 1458, 1554, 1600, 1613, 1888, 1937, 1946, 1959), axis=0) #Only for the new dataset 
    
# np.random.shuffle(uvp)

x_grid = np.load(data)['Rgrid'][0,:].astype(np.float32)
y_grid = np.load(data)['Zgrid'][:,0].astype(np.float32)
t_grid = np.load(data)['time'].astype(np.float32)

# ntrain =240
# ntest = 36

ntrain = 1500 #new dataset
ntest = 85 #new dataset

S = 106 #Grid Size
size_x = S
size_y = S

batch_size = configuration['Batch Size']

batch_size2 = batch_size

t1 = default_timer()


train_a = uvp[:ntrain,:,:,:,:T_in]
train_u = uvp[:ntrain,:,:,:,T_in:T+T_in]

test_a = uvp[-ntest:,:,:,:,:T_in]
test_u = uvp[-ntest:,:,:,:,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)


# %%

a_normalizer = MinMax_Normalizer(uvp[...,:T_in])

train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = MinMax_Normalizer(uvp[...,T_in:T+T_in])

train_u = y_normalizer.encode(train_u)
test_u_encoded = y_normalizer.encode(test_u)

# %%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %% 

################################################################
# training and evaluation
################################################################

# = FNO_multi(16, 16, width_vars, width_time)
model = FNO_multi(T_in, step, num_vars, modes, modes, width_vars, width_time)
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_weary-tactics.pth', map_location=torch.device('cpu'))) #Min-Max Diff
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_polite-comment.pth', map_location=torch.device('cpu'))) #Min-Max Same 
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_dynamic-duck.pth', map_location=torch.device('cpu'))) #Proper Skip - Finals 
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_violent-spot.pth', map_location=torch.device('cpu'))) #500 Epochs 
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_future-pavilion.pth', map_location=torch.device('cpu'))) #Normalisation test train 
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_equidistant-pint.pth', map_location=torch.device('cpu'))) #250 Epochs 
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_wary-deck.pth', map_location=torch.device('cpu'))) #Different Scaling
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_randomized-levee.pth', map_location=torch.device('cpu'))) #100 Epochs 
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_cool-subcompact.pth', map_location=torch.device('cpu'))) #1500 ntrain 
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_humane-score.pth', map_location=torch.device('cpu'))) #1750 ntrain 
model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_diachronic-restaurant.pth', map_location=torch.device('cpu'))) #500 ntrain

model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))


# %%
epochs = configuration['Epochs']
if torch.cuda.is_available():
    y_normalizer.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

myloss = LpLoss(size_average=False)

# %%
#Testing 
batch_size = 1
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=1, shuffle=False)
pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        # xx = additive_noise(xx)
        t1 = default_timer()
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            out = model(xx)
            loss += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       

            xx = torch.cat((xx[..., step:], out), dim=-1)

        t2 = default_timer()
        # pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
        # print(t2-t1)

# %% 
print(pred_set.shape, test_u.shape)
#Logging Metrics 
MSE_error = (pred_set - test_u_encoded).pow(2).mean()
MAE_error = torch.abs(pred_set - test_u_encoded).mean()
LP_error = loss / (ntest*T/step)
rel_error = torch.abs((pred_set - test_u_encoded)/test_u_encoded).mean() * 100 
nmse = ((pred_set - test_u_encoded).pow(2).mean() / test_u_encoded.pow(2).mean())
nrmse = torch.sqrt((pred_set - test_u_encoded).pow(2).mean()) / torch.std(test_u_encoded)

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))
# print('(LP) Testing Error: %.3e' % (LP_error))
# print('(MAPE) Testing Error %.3e' % (rel_error))
# print('(NMSE) Testing Error %.3e' % (nmse))
# print('(NRMSE) Testing Error %.3e' % (nrmse))

# run.update_metadata({'MSE Test Error': float(MSE_error),
#                      'MAE Test Error': float(MAE_error),
#                      'LP Test Error': float(LP_error)
#                     })


pred_set_encoded = pred_set
pred_set = y_normalizer.decode(pred_set.to(device)).cpu()

nmse= 0 
for ii in range(num_vars):
    nmse += (pred_set[:,ii] - test_u[:,ii]).pow(2).mean() / test_u[:,ii].pow(2).mean()
    print(test_u[:,ii].pow(2).mean())
nmse = nmse/num_vars
print('(NMSE) Testing Error %.3e' % (nmse))

pred_set_scaled = pred_set
test_u_scaled = test_u

# %%
#Plotting the error growth across time.
err_rho = [] 
err_phi = []
err_T = []

for ii in range(T):
    # err_rho.append((pred_set_scaled[:,0,:,:,ii] - test_u_scaled[:,0,:,:,ii]).pow(2).mean() / torch.std(test_u_scaled[:,0].pow(2).mean()))
    # err_phi.append((pred_set_scaled[:,1,:,:,ii] - test_u_scaled[:,1,:,:,ii]).pow(2).mean() / torch.std(test_u_scaled[:,1].pow(2).mean()))
    # err_T.append((pred_set_scaled[:,2,:,:,ii] - test_u_scaled[:,2,:,:,ii]).pow(2).mean() / torch.std(test_u_scaled[:,2].pow(2).mean()))
    
    err_rho.append(torch.abs(pred_set_encoded[:,0,:,:,ii] - test_u_encoded[:,0,:,:,ii]).mean())
    err_phi.append(torch.abs(pred_set_encoded[:,1,:,:,ii] - test_u_encoded[:,1,:,:,ii]).mean())
    err_T.append(torch.abs(pred_set_encoded[:,2,:,:,ii] - test_u_encoded[:,2,:,:,ii]).mean())

err_rho = np.asarray(err_rho)
err_phi = np.asarray(err_phi)
err_T = np.asarray(err_T)

# %% 
if configuration["Physics Normalisation"] == 'Yes':
    pred_set[:,0:1,...] = pred_set[:,0:1,...] * 1e20
    pred_set[:,1:2,...] = pred_set[:,1:2,...] * 1e5 / 1e2
    pred_set[:,2:3,...] = pred_set[:,2:3,...] * 1e6 /1e4


    test_u[:,0:1,...] = test_u[:,0:1,...] * 1e20
    test_u[:,1:2,...] = test_u[:,1:2,...] * 1e5 / 1e2
    test_u[:,2:3,...] = test_u[:,2:3,...] * 1e6 / 1e4

# %%
#Plotting the comparison plots
idx = np.random.randint(0,ntest) 
# idx = 5
# idx = 36
# idx = 3
# idx = 5  
idx = 15
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['font.size']=16

output_plot = []
for dim in range(num_vars):
    u_field = test_u[idx]

    v_min_1 = torch.min(u_field[dim,:,:,0])
    v_max_1 = torch.max(u_field[dim,:,:,0])

    v_min_2 = torch.min(u_field[dim, :, :, int(T/2)])
    v_max_2 = torch.max(u_field[dim, :, :, int(T/2)])

    v_min_3 = torch.min(u_field[dim, :, :, -1])
    v_max_3 = torch.max(u_field[dim, :, :, -1])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(2,3,1)
    pcm =ax.imshow(u_field[dim,:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t='+ str(T_in))
    ax.set_ylabel('Solution', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    
    ax = fig.add_subplot(2,3,2)
    pcm = ax.imshow(u_field[dim,:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t='+ str(int((T+T_in)/2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(2,3,3)
    pcm = ax.imshow(u_field[dim,:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t='+str(T+T_in))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    u_field = pred_set[idx]

    ax = fig.add_subplot(2,3,4)
    pcm = ax.imshow(u_field[dim,:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    ax.set_ylabel('FNO', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(2,3,5)
    pcm = ax.imshow(u_field[dim,:,:,int(T/2)], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(2,3,6)
    pcm = ax.imshow(u_field[dim,:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

# %% 
#Error Plots
idx = 35
output_plot = []
for dim in range(num_vars):
    u_field = test_u[idx]

    v_min_1 = torch.min(u_field[dim,:,:,0])
    v_max_1 = torch.max(u_field[dim,:,:,0])

    v_min_2 = torch.min(u_field[dim, :, :, int(T/2)])
    v_max_2 = torch.max(u_field[dim, :, :, int(T/2)])

    v_min_3 = torch.min(u_field[dim, :, :, -1])
    v_max_3 = torch.max(u_field[dim, :, :, -1])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(3,3,1)
    pcm =ax.imshow(u_field[dim,:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t='+ str(T_in))
    ax.set_ylabel('Solution', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    
    ax = fig.add_subplot(3,3,2)
    pcm = ax.imshow(u_field[dim,:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t='+ str(int((T+T_in)/2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(3,3,3)
    pcm = ax.imshow(u_field[dim,:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t='+str(T+T_in))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    u_field = pred_set[idx]

    ax = fig.add_subplot(3,3,4)
    pcm = ax.imshow(u_field[dim,:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    ax.set_ylabel('FNO', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(3,3,5)
    pcm = ax.imshow(u_field[dim,:,:,int(T/2)], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(3,3,6)
    pcm = ax.imshow(u_field[dim,:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    u_field = torch.abs(test_u[idx] - pred_set[idx])

    ax = fig.add_subplot(3,3,7)
    pcm = ax.imshow(u_field[dim,:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5])
    ax.set_ylabel('Abs Error', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(3,3,8)
    pcm = ax.imshow(u_field[dim,:,:,int(T/2)], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(3,3,9)
    pcm = ax.imshow(u_field[dim,:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    plt.savefig("multiblobs_" + dims[dim] + "_" + str(idx) + "_cool-subcompact_error.pdf", format="pdf", bbox_inches='tight', transparent='True')


# %% 
#Error in the Physical Domain

#Plotting the error growth across time.
err_rho = [] 
err_phi = []
err_T = []

for ii in range(T):

    err_rho.append(torch.abs(pred_set[:,0,:,:,ii] - test_u[:,0,:,:,ii]).mean())
    err_phi.append(torch.abs(pred_set[:,1,:,:,ii] - test_u[:,1,:,:,ii]).mean())
    err_T.append(torch.abs(pred_set[:,2,:,:,ii] - test_u[:,2,:,:,ii]).mean())

err_rho = np.asarray(err_rho)
err_phi = np.asarray(err_phi)
err_T = np.asarray(err_T)

# %%
import matplotlib as mpl
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 3
mpl.rcParams['axes.titlepad'] = 20
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
mpl.rcParams['axes.titlepad'] = 20

# %%
# plt.plot(np.arange(T_in, T_in + T), err_rho, label='Density', alpha=0.8,  color = 'navy')
plt.plot(np.arange(T_in, T_in + T), err_phi, label='Potential', alpha=0.8,  color = 'darkgreen')
# plt.plot(np.arange(T_in, T_in + T), err_T, label='Temp', alpha=0.8,  color = 'maroon')
# plt.plot(np.arange(T_in, T_in + T), (err_rho+err_phi+err_T), label='Cumulative', alpha=0.8,  color = 'black', ls='--', linewidth=5)
plt.legend()
plt.grid()
plt.xlabel('Time Steps')
plt.ylabel('MAE')


plt.rcParams['text.usetex'] = True
# plt.ylabel('MAE $(m^{-3})$ ')
plt.ylabel('MAE $(V)$ ')
# plt.ylabel('MAE $(eV)$ ')

# plt.savefig('multiblobs_error_growth_T.png')  
# plt.savefig("multiblobs_error_growth_cum.pdf", bbox_inches='tight')
plt.savefig("multiblobs_error_growth_phi.svg", bbox_inches='tight')
# %%
# #Dropout Plots
# #Cyan-Provolone
# configuration = {"Case": 'Multi-Blobs',
#                  "Field": 'rho, Phi, T',
#                  "Field_Mixing": 'Channel',
#                  "Type": '2D Time',
#                  "Epochs": 500,
#                  "Batch Size": 10,
#                  "Optimizer": 'Adam',
#                  "Learning Rate": 0.005,
#                  "Scheduler Step": 100,
#                  "Scheduler Gamma": 0.5,
#                  "Activation": 'GELU',
#                  "Normalisation Strategy": 'Min-Max',
#                  "Instance Norm": 'No',
#                  "Log Normalisation":  'No',
#                  "Physics Normalisation": 'Yes',
#                  "T_in": 10,    
#                  "T_out": 40,
#                  "Step": 5,
#                  "Modes":16,
#                  "Width_time":32, #FNO
#                  "Width_vars": 0, #U-Net
#                  "Variables":3, 
#                  "Noise":0.0, 
#                  "Loss Function": 'LP Loss',
#                  "Spatial Resolution": 1,
#                  "Temporal Resolution": 1,
#                  "UQ": 'Dropout',
#                  "Dropout Rate": 0.9
#                  }

# T_in  = configuration['T_in']
# T_out = configuration['T_out']
# step = configuration['Step']
# modes = configuration['Modes']
# width_vars = configuration['Width_vars']
# width_time = configuration['Width_time']


# class FNO_multi_dropout(nn.Module):
#     def __init__(self, modes1, modes2, width_vars, width_time):
#         super(FNO_multi_dropout, self).__init__()

#         """
#         The overall network. It contains 4 layers of the Fourier layer.
#         1. Lift the input to the desire channel dimension by self.fc0 .
#         2. 4 layers of the integral operators u' = (W + K)(u).
#             W defined by self.w; K defined by self.conv .
#         3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
#         input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
#         input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
#         output: the solution of the next timestep
#         output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
#         """

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.width_vars = width_vars
#         self.width_time = width_time

#         self.fc0_time  = nn.Linear(T_in+2, self.width_time)

#         # self.padding = 8 # pad the domain if input is non-periodic

#         self.f0 = FNO2d(self.modes1, self.modes2, self.width_time)
#         self.f1 = FNO2d(self.modes1, self.modes2, self.width_time)
#         self.f2 = FNO2d(self.modes1, self.modes2, self.width_time)
#         self.f3 = FNO2d(self.modes1, self.modes2, self.width_time)
#         self.f4 = FNO2d(self.modes1, self.modes2, self.width_time)
#         self.f5 = FNO2d(self.modes1, self.modes2, self.width_time)

#         self.dropout = nn.Dropout(p=0.1)

#         # self.norm = nn.InstanceNorm2d(self.width)
#         self.norm = nn.Identity()


#         self.fc1_time = nn.Linear(self.width_time, 128)
#         self.fc2_time = nn.Linear(128, step)


#     def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)


#         x = self.fc0_time(x)
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.dropout(x)

#         # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

#         x0 = self.f0(x)
#         x = self.f1(x0)
#         x = self.f2(x) + x0 
#         x = self.dropout(x)
#         x1 = self.f3(x)
#         x = self.f4(x1)
#         x = self.f5(x) + x1 
#         x = self.dropout(x)

#         # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic

#         x = x.permute(0, 2, 3, 4, 1)
#         x = x 

#         x = self.fc1_time(x)
#         x = F.gelu(x)
#         x = self.dropout(x)
#         x = self.fc2_time(x)
        
#         return x

# #Using x and y values from the simulation discretisation 
#     def get_grid(self, shape, device):
#         batchsize, num_vars, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
#         gridx = gridx = torch.tensor(x_grid, dtype=torch.float)
#         gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, num_vars, 1, size_y, 1])
#         gridy = torch.tensor(y_grid, dtype=torch.float)
#         gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat([batchsize, num_vars, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)

# ## Arbitrary grid discretisation 
#     # def get_grid(self, shape, device):
#     #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#     #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#     #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#     #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#     #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#     #     return torch.cat((gridx, gridy), dim=-1).to(device)


#     def count_params(self):
#         c = 0
#         for p in self.parameters():
#             c += reduce(operator.mul, list(p.size()))

#         return c

# model = FNO_multi_dropout(modes, modes, width_vars, width_time)
# model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_cyan-provolone.pth', map_location=torch.device('cpu')))
# model.to(device)



# # %%
# #Extracting the Mean and Variance across the time roll out to plot later. 

# # %%
# idx = 36
# model.eval()
# xx = test_a[idx:idx+1]
# yy = test_u_encoded[idx:idx+1,:, :,:,:10]
# var = 0 
# preds = []

# for i in tqdm(range(100)):
#         preds.append(model(xx).detach().numpy())

# preds_mean = np.mean(preds, axis=0)
# preds_std = np.std(preds, axis=0)
# # %%
# test_yy= y_normalizer.decode(torch.Tensor(yy)) * 1e20
# preds_mean = y_normalizer.decode(torch.Tensor(preds_mean)).detach().numpy() * 1e20
# preds_std= y_normalizer.decode(torch.Tensor(preds_std)).detach().numpy() * 1e20


# T = step

# u_field = test_yy[0][var]

# v_min_1 = torch.min(u_field[:,:,0])
# v_max_1 = torch.max(u_field[:,:,0])

# v_min_2 = torch.min(u_field[:, :, int(T/2)])
# v_max_2 = torch.max(u_field[:, :, int(T/2)])

# v_min_3 = torch.min(u_field[:, :, -1])
# v_max_3 = torch.max(u_field[:, :, -1])

# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(2,3,1)
# pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# # ax.title.set_text('Initial')
# ax.title.set_text('t='+ str(T_in))
# ax.set_ylabel('Solution')
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(2,3,2)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# # ax.title.set_text('Middle')
# ax.title.set_text('t='+ str(int((T/2+T_in))))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(2,3,3)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# # ax.title.set_text('Final')
# ax.title.set_text('t='+str(T+T_in))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# u_field = preds_mean[0][var]

# ax = fig.add_subplot(2,3,4)
# pcm = ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# ax.set_ylabel('FNO')

# fig.colorbar(pcm, pad=0.05)

# ax = fig.add_subplot(2,3,5)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(2,3,6)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)



# from mpl_toolkits.axes_grid1 import make_axes_locatable

# u_field = preds_std[0][var]

# v_min_1 = np.min(u_field[:,:,0])
# v_max_1 = np.max(u_field[:,:,0])

# v_min_2 = np.min(u_field[:, :, int(T/2)])
# v_max_2 = np.max(u_field[:, :, int(T/2)])

# v_min_3 = np.min(u_field[:, :, -1])
# v_max_3 = np.max(u_field[:, :, -1])

# fig = plt.figure(figsize=plt.figaspect(0.4))
# ax = fig.add_subplot(1,3,1)
# pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# # ax.title.set_text('Initial')
# ax.title.set_text('t='+ str(T_in))
# ax.set_ylabel('STD')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax = cax)


# ax = fig.add_subplot(1,3,2)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# # ax.title.set_text('Middle')
# ax.title.set_text('t='+ str(int((T/2+T_in))))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax = cax)


# ax = fig.add_subplot(1,3,3)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# # ax.title.set_text('Final')
# ax.title.set_text('t='+str(T+T_in))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax = cax)

# # %%
# idx = 36
# model.eval()
# xx = test_a[idx:idx+1]
# yy = test_u_encoded[idx:idx+1]
# preds = []
# with torch.no_grad():
#     for i in tqdm(range(100)):
#         xx = test_a[idx:idx+1]
#         for t in range(0, T, step):
#             out = model(xx)

#             if t == 0:
#                 pred = out
#             else:
#                 pred = torch.cat((pred, out), -1)       

#             xx = torch.cat((xx[..., step:], out), dim=-1)
#         preds.append(pred.detach().numpy())



# # %%
# preds_mean = np.mean(preds, axis=0)
# preds_std = np.std(preds, axis=0)
# # %%
# test_yy = y_normalizer.decode(torch.Tensor(yy)) * 1e20
# preds_mean = y_normalizer.decode(torch.Tensor(preds_mean)).detach().numpy() * 1e20
# preds_std= y_normalizer.decode(torch.Tensor(preds_std)).detach().numpy() * 1e20

# # %%

# T = configuration['T_out']
# T = 5

# u_field = test_yy[0][var][...,:20]

# v_min_1 = torch.min(u_field[:,:,0])
# v_max_1 = torch.max(u_field[:,:,0])

# v_min_2 = torch.min(u_field[:, :, int(T/2)])
# v_max_2 = torch.max(u_field[:, :, int(T/2)])

# v_min_3 = torch.min(u_field[:, :, -1])
# v_max_3 = torch.max(u_field[:, :, -1])

# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(2,3,1)
# pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# # ax.title.set_text('Initial')
# ax.title.set_text('t='+ str(T_in))
# ax.set_ylabel('Solution')
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(2,3,2)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# # ax.title.set_text('Middle')
# ax.title.set_text('t='+ str(int((T/2+T_in))))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(2,3,3)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# # ax.title.set_text('Final')
# ax.title.set_text('t='+str(T+T_in))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# u_field = preds_mean[0][var]

# ax = fig.add_subplot(2,3,4)
# pcm = ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# ax.set_ylabel('FNO')

# fig.colorbar(pcm, pad=0.05)

# ax = fig.add_subplot(2,3,5)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(2,3,6)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)



# from mpl_toolkits.axes_grid1 import make_axes_locatable

# u_field = preds_std[0][var]

# v_min_1 = np.min(u_field[:,:,0])
# v_max_1 = np.max(u_field[:,:,0])

# v_min_2 = np.min(u_field[:, :, int(T/2)])
# v_max_2 = np.max(u_field[:, :, int(T/2)])

# v_min_3 = np.min(u_field[:, :, -1])
# v_max_3 = np.max(u_field[:, :, -1])

# fig = plt.figure(figsize=plt.figaspect(0.4))
# ax = fig.add_subplot(1,3,1)
# pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# # ax.title.set_text('Initial')
# ax.title.set_text('t='+ str(T_in))
# ax.set_ylabel('STD')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax = cax)


# ax = fig.add_subplot(1,3,2)
# pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# # ax.title.set_text('Middle')
# ax.title.set_text('t='+ str(int((T/2+T_in))))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax = cax)


# ax = fig.add_subplot(1,3,3)
# pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# # ax.title.set_text('Final')
# ax.title.set_text('t='+str(T+T_in))
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(pcm, cax = cax)

# %%
#Individual Models
configuration = {"Case": 'Multi-Blobs',
                 "Field": 'rho',
                 "Field_Mixing": 'Channel',
                 "Type": '2D Time',
                 "Epochs": 500,
                 "Batch Size": 4,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No',
                 "Log Normalisation": 'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 10,
                 "T_out": 40,
                 "Step":5,
                 "Modes": 16,
                 "Width_time": 32,  # FNO
                 "Width_vars": 0,  # U-Net
                 "Variables": 1,
                 "Noise": 0.0,
                 "Loss Function": 'LP Loss',
                 "Spatial Resolution": 1,
                 "Temporal Resolution": 1,
                 "Gradient Clipping Norm": None,
                 #  "UQ": 'Dropout',
                 #  "Dropout Rate": 0.9
                 }
dims = ['rho', 'Phi', 'T']

# %%
##################################
#Normalisation Functions 
##################################

# normalization, rangewise but single value.
#normalization, rangewise but across the full domain
class MinMax_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)
        # mymin = torch.tensor(0.0)
        # mymax = torch.tensor(0.3)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high
        print(mymin, mymax)

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

# %%

################################################################
# Preparing Data 
################################################################

errs = []
for field in dims:  
    print(field)
        
    if field == 'Phi':
        u_sol = np.load(data)['Phi'].astype(np.float32)   / 1e5
    elif field == 'T':
        u_sol = np.load(data)['T'].astype(np.float32)     / 1e6
    elif field == 'rho':
        u_sol = np.load(data)['rho'].astype(np.float32)   / 1e20

    if configuration['Log Normalisation'] == 'Yes':
        u_sol = np.log(u_sol)

    u_sol = np.nan_to_num(u_sol)
    # u_sol = np.delete(u_sol, (11, 160, 222, 273, 303, 357, 620, 797, 983, 1275, 1391, 1458, 1554, 1600, 1613, 1888, 1937, 1946, 1959), axis=0)
    u_sol= np.delete(u_sol, (153, 229), axis=0) #Outlier T values

    x_grid = np.load(data)['Rgrid'][0,:].astype(np.float32)
    y_grid = np.load(data)['Zgrid'][:,0].astype(np.float32)
    t_grid = np.load(data)['time'].astype(np.float32)

    ntrain = 240
    ntest = 36
    S =106  # Grid Size

    #Extracting hyperparameters from the config dict
    modes = configuration['Modes']

    output_size = configuration['Step']
    batch_size = configuration['Batch Size']
    T_in = configuration['T_in']
    T = configuration['T_out']
    step = configuration['Step']
    num_vars = configuration['Variables']
    width_time = configuration['Width_time']
    width_vars = configuration['Width_vars']
    t1 = default_timer()

    #At this stage the data needs to be [Batch_Size, X, Y, T]

    # np.random.shuffle(u_sol)
    u = torch.from_numpy(u_sol)
    u = u.permute(0, 2, 3, 1)

    #At this stage the data needs to be [Batch_Size, X, Y, T]

    train_a = u[:ntrain,:,:,:T_in]
    train_u = u[:ntrain,:,:,T_in:T+T_in]

    test_a = u[-ntest:,:,:,:T_in]
    test_u = u[-ntest:,:,:,T_in:T+T_in]


    #Normalising the train and test datasets with the preferred normalisation. 
    norm_strategy = configuration['Normalisation Strategy']

    if norm_strategy == 'Min-Max':
        a_normalizer = MinMax_Normalizer(u[:,:,:,:T_in])
        y_normalizer = MinMax_Normalizer(u[:,:,:,T_in:T+T_in])

    if norm_strategy == 'Range':
        a_normalizer = RangeNormalizer(train_a)
        y_normalizer = RangeNormalizer(train_u)

    if norm_strategy == 'Gaussian':
        a_normalizer = GaussianNormalizer(train_a)
        y_normalizer = GaussianNormalizer(train_u)


    test_a = a_normalizer.encode(test_a)
    test_u_encoded = y_normalizer.encode(test_u)


    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    ################################################################
    # training and evaluation
    ################################################################

    #Instantiating the Model. 
    # model = FNO(modes, modes, width)
    num_vars = 1
    model = FNO_multi(T_in, step, num_vars, modes, modes, width_vars, width_time)

    if field == 'rho':
        # model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_isothermal-vignette.pth', map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_big-tag.pth', map_location=torch.device('cpu')))
    if field == 'Phi':
        # model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_khaki-error.pth', map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_stable-snipe.pth', map_location=torch.device('cpu')))

    if field == 'T':
        # model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_tempered-ladder.pth', map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(file_loc + '/Models/FNO_multi_blobs_thin-house.pth', map_location=torch.device('cpu')))

    model.to(device)

    run.update_metadata({'Number of Params': int(model.count_params())})
    print("Number of model params : " + str(model.count_params()))

    if torch.cuda.is_available():
        y_normalizer.cuda()

    # Testing
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a.unsqueeze(1), test_u_encoded.unsqueeze(1)), batch_size=1,
                                            shuffle=False)
    pred_set = torch.zeros(test_u.shape)
    index = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx, yy = xx.to(device), yy.to(device)
            # xx = additive_noise(xx)
            t1 = default_timer()
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                out = model(xx)

                if t == 0:
                    pred = out
                else:
                    pred = torch.cat((pred, out), -1)

                xx = torch.cat((xx[..., step:], out), dim=-1)

            t2 = default_timer()
            # pred = y_normalizer.decode(pred)
            pred_set[index] = pred
            index += 1
            # print(t2 - t1)

    #Logging Metrics 
    MSE_error = (pred_set - test_u_encoded).pow(2).mean()
    MAE_error = torch.abs(pred_set - test_u_encoded).mean()
    # LP_error = loss / (ntest*T/step)
    # rel_error = torch.abs((pred_set - test_u_encoded)/test_u_encoded).mean() * 100 



    print('(MSE) Testing Error: %.3e' % (MSE_error))
    print('(MAE) Testing Error: %.3e' % (MAE_error))
    # print('(LP) Testing Error: %.3e' % (LP_error))
    # print('(MAPE) Testing Error %.3e' % (rel_error))
    # print('(NMSE) Testing Error %.3e' % (nmse))
    # print('(NRMSE) Testing Error %.3e' % (nrmse))

    # run.update_metadata({'MSE Test Error': float(MSE_error),
    #                     'MAE Test Error': float(MAE_error),
    #                     'LP Test Error': float(LP_error)
    #                     })

    pred_set_encoded = pred_set
    pred_set = y_normalizer.decode(pred_set.to(device)).cpu()
    pred_set_scaled = pred_set
    test_u_scaled = test_u

    nmse = ((pred_set_scaled - test_u_scaled).pow(2).mean() / test_u_scaled.pow(2).mean())
    print('(NMSE) Testing Error %.3e' % (nmse))



    if configuration["Physics Normalisation"] == 'Yes':
        if field == 'rho':
            pred_set = pred_set * 1e20
            test_u = test_u * 1e20
        if field == 'Phi':
            pred_set = pred_set * 1e5
            test_u = test_u * 1e5
        if field == 'T':
            pred_set= pred_set * 1e6
            test_u = test_u * 1e6

    #Plotting the comparison plots

    idx = np.random.randint(0,ntest) 
    # idx = 3
    idx = 15

    if configuration['Log Normalisation'] == 'Yes':
        test_u = torch.exp(test_u)
        pred_set = torch.exp(pred_set)

    u_field = test_u[idx]

    v_min_1 = torch.min(u_field[:,:,0])
    v_max_1 = torch.max(u_field[:,:,0])

    v_min_2 = torch.min(u_field[:, :, int(T/2)])
    v_max_2 = torch.max(u_field[:, :, int(T/2)])

    v_min_3 = torch.min(u_field[:, :, -1])
    v_max_3 = torch.max(u_field[:, :, -1])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2,3,1)
    pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t='+ str(T_in))
    ax.set_ylabel('Solution')
    fig.colorbar(pcm, pad=0.05)


    ax = fig.add_subplot(2,3,2)
    pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t='+ str(int((T+T_in)/2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)


    ax = fig.add_subplot(2,3,3)
    pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t='+str(T+T_in))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)


    u_field = pred_set[idx]

    ax = fig.add_subplot(2,3,4)
    pcm = ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    ax.set_ylabel('FNO')

    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2,3,5)
    pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)


    ax = fig.add_subplot(2,3,6)
    pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    #Plotting the error growth across time.
    err = [] 

    for ii in range(T):
        # err.append((pred_set_scaled[...,ii] - test_u_scaled[...,ii]).pow(2).mean() / test_u_scaled.pow(2).mean())
        err.append(torch.abs(pred_set_encoded[...,ii] - test_u_encoded[...,ii]).mean())

    print(test_u_scaled.pow(2).mean())
    err = np.asarray(err)
    errs.append(err)

# %%
# plt.plot(np.arange(T_in, T_in + T), err, label=field + ' - solo', alpha=0.8,  color = 'tab:brown')
# plt.plot(np.arange(T_in, T_in + T), err_rho, label='Density', alpha=0.8,  color = 'tab:blue')
# # plt.plot(np.arange(T_in, T_in + T), err_phi, label='Potential', alpha=0.8,  color = 'tab:orange')
# # plt.plot(np.arange(T_in, T_in + T), err_T, label='Temp', alpha=0.8,  color = 'tab:green')
# # plt.plot(np.arange(T_in, T_in + T), (err_rho+err_phi+err_T), label='Cumulative', alpha=0.8,  color = 'tab:red', ls='--')
# plt.legend()
# plt.xlabel('Time Steps')
# plt.ylabel('NMAE ')
# %%
err_rho_solo, err_phi_solo, err_T_solo = errs[0], errs[1], errs[2]
# %%
import matplotlib as mpl

plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['font.size']=45
mpl.rcParams['figure.figsize']=(16,16)
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['axes.linewidth']= 1
mpl.rcParams['axes.titlepad'] = 30
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 10.0
plt.rcParams['ytick.minor.size'] = 10.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
mpl.rcParams['lines.linewidth'] = 1
plt.figure()
plt.grid()
plt.plot(np.arange(T_in, T_in + T), err_rho_solo, label='Density - Single', alpha=0.8,  color = 'royalblue', ls='--')
plt.plot(np.arange(T_in, T_in + T), err_rho, label='Density - Multi', alpha=0.7,  color = 'navy')
plt.plot(np.arange(T_in, T_in + T), err_phi_solo, label='Potential - Single', alpha=0.8,  color = 'mediumseagreen', ls='--')
plt.plot(np.arange(T_in, T_in + T), err_phi, label='Potential - Multi', alpha=0.7,  color = 'darkgreen')
plt.plot(np.arange(T_in, T_in + T), err_T_solo, label='Temp - Single', alpha=0.8,  color = 'lightcoral', ls='--')
plt.plot(np.arange(T_in, T_in + T), err_T, label='Temp - Multi', alpha=0.7,  color = 'maroon')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('MAE ')

# plt.savefig("multiblobs_error_growth.pdf", bbox_inches='tight')
# plt.savefig("multiblobs_error_growth.svg", bbox_inches='tight')
# %%
