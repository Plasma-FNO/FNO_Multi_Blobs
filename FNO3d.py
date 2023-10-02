# !/usr/bin/env python3
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
                 "Epochs": 10,
                 "Batch Size": 2,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.00025,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max. Different',
                 "Instance Norm": 'No',
                 "Log Normalisation": 'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 10,
                 "T_out": 40,
                 "Step": 10,
                 "Modes": 12,
                 "Width_time": 32,  # FNO
                 "Width_vars": 0,  # U-Net
                 "Variables": 3,
                 "Noise": 0.0,
                 "Loss Function": 'LP Loss',
                 "Spatial Resolution": 1,
                 "Temporal Resolution": 1,
                 "Gradient Clipping Norm": None,
                 #  "UQ": 'Dropout',
                 #  "Dropout Rate": 0.9
                 }

# %%
# from simvue import Run
# run = Run()
# run.init(folder="/FNO_MHD", tags=['FNO', 'MHD', 'JOREK', 'Multi-Blobs', 'MultiVariable', "Skip_Connect", "discretisation-invariant"], metadata=configuration)
#
# # %%
import os
# CODE = ['FNO_multiple.py']
#
# # Save code files
# for code_file in CODE:
#     if os.path.isfile(code_file):
#         run.save(code_file, 'code')
#     elif os.path.isdir(code_file):
#         run.save_directory(code_file, 'code', 'text/plain', preserve_path=True)
#     else:
#         print('ERROR: code file %s does not exist' % code_file)


# %%

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
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
# Normalisation Functions
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
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

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

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


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



# %%
##################################
# Loss Functions
##################################

# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# %%
# Extracting the configuration settings

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

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
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


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic

        self.p = nn.Linear(3+30,
                           self.width)  # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, 3, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        # x = x.repeat_interleave(4, dim=-1)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[3] * x.shape[4])
        x = x.repeat(1,1,1,T,1)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = self.q(x)
        # x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# Loading Data
################################################################

# %%
data = '/media/wumming/HHD/HHD_data/MHD_multi_blobs.npz'
# data = data_loc + '/Data/FNO_MHD_data_multi_blob_500x500.npz' # For Performing SuperResolution.
# %%
field = configuration['Field']
dims = ['rho', 'Phi', 'T']
num_vars = configuration['Variables']

u_sol = np.load(data)['rho'].astype(np.float32) / 1e20
v_sol = np.load(data)['Phi'].astype(np.float32) / 1e5
p_sol = np.load(data)['T'].astype(np.float32) / 1e6

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
uvp = torch.stack((u, v, p), dim=1)[:, ::t_res]
uvp = np.delete(uvp, (153, 229), axis=0)  # Outlier T values

x_grid = np.load(data)['Rgrid'][0, :].astype(np.float32)
y_grid = np.load(data)['Zgrid'][:, 0].astype(np.float32)
t_grid = np.load(data)['time'].astype(np.float32)

# #Padding Removed
# uvp = uvp[:, :, 3:-3, 3:-3, :]
# x_grid = x_grid[3:-3]
# y_grid = y_grid[3:-3]

ntrain = 240
ntest = 38
S = uvp.shape[3]  # Grid Size
size_x = S
size_y = S

batch_size = configuration['Batch Size']

batch_size2 = batch_size

t1 = default_timer()

train_a = uvp[:ntrain, :, :, :, :T_in]
train_u = uvp[:ntrain, :, :, :, T_in:T + T_in]

test_a = uvp[-ntest:, :, :, :, :T_in]
test_u = uvp[-ntest:, :, :, :, T_in:T + T_in]

print(train_u.shape)
print(test_u.shape)

# %%
# a_normalizer = RangeNormalizer(train_a)
a_normalizer = MinMax_Normalizer(train_a)
# a_normalizer = GaussianNormalizer(train_a)

train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

# y_normalizer = RangeNormalizer(train_u)
y_normalizer = MinMax_Normalizer(train_u)
# y_normalizer = GaussianNormalizer(train_u)

train_u = y_normalizer.encode(train_u)
test_u_encoded = y_normalizer.encode(test_u)

# %%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size,
                                          shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2 - t1)

# %%

################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width_time)
model.to(device)

# run.update_metadata({'Number of Params': int(model.count_params())})


# print("Number of model params : " + str(model.count_params()))

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'],
                                            gamma=configuration['Scheduler Gamma'])

myloss = LpLoss(size_average=False)

# %%
epochs = configuration['Epochs']
if torch.cuda.is_available():
    y_normalizer.cuda()

# %%

max_grad_clip_norm = configuration['Gradient Clipping Norm']

start_time = time.time()

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)




    # run.log_metrics({'Train Loss': train_loss,
    #                 'Test Loss': test_loss})

train_time = time.time() - start_time
# %%
# Saving the Model
# model_loc = file_loc + '/Models/FNO_multi_blobs_' + run.name + '.pth'
torch.save(model.state_dict(), 'model.pt')

# %%
# Testing
batch_size = 1
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=1,

                                          shuffle=False)
pred = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    for x, y in tqdm(test_loader):
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1

# %%
print(pred.shape, test_u.shape)
# Logging Metrics
MSE_error = (pred - test_u_encoded).pow(2).mean()
MAE_error = torch.abs(pred - test_u_encoded).mean()

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))

# run.update_metadata({'Training Time': float(train_time),
#                      'MSE Test Error': float(MSE_error),
#                      'MAE Test Error': float(MAE_error),
#                      'LP Test Error': float(LP_error)
#                     })

pred_set = y_normalizer.decode(pred.to(device)).cpu()

# %%
# Plotting the comparison plots

idx = np.random.randint(0, ntest)
idx = 5

if configuration['Log Normalisation'] == 'Yes':
    test_u = torch.exp(test_u)
    pred_set = torch.exp(pred_set)

# %%
output_plot = []
for dim in range(num_vars):
    u_field = test_u[idx]

    v_min_1 = torch.min(u_field[dim, :, :, 0])
    v_max_1 = torch.max(u_field[dim, :, :, 0])

    v_min_2 = torch.min(u_field[dim, :, :, int(T / 2)])
    v_max_2 = torch.max(u_field[dim, :, :, int(T / 2)])

    v_min_3 = torch.min(u_field[dim, :, :, -1])
    v_max_3 = torch.max(u_field[dim, :, :, -1])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 3, 1)
    pcm = ax.imshow(u_field[dim, :, :, 0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t=' + str(T_in))
    ax.set_ylabel('Solution')
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 2)
    pcm = ax.imshow(u_field[dim, :, :, int(T / 2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t=' + str(int((T + T_in) / 2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 3)
    pcm = ax.imshow(u_field[dim, :, :, -1], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t=' + str(T + T_in))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    u_field = pred_set[idx]

    ax = fig.add_subplot(2, 3, 4)
    pcm = ax.imshow(u_field[dim, :, :, 0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    ax.set_ylabel('FNO')

    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 5)
    pcm = ax.imshow(u_field[dim, :, :, int(T / 2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 6)
    pcm = ax.imshow(u_field[dim, :, :, -1], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    plt.title(dims[dim])

    # output_plot.append(file_loc + '/Plots/MultiBlobs_' + dims[dim] + '_' + run.name + '.png')
    plt.savefig('plot'+str(dim)+'.png', bbox_inches='tight')

# %%


# # %%
#
# INPUTS = []
# OUTPUTS = [model_loc, output_plot[0], output_plot[1], output_plot[2]]
#
#
# # Save input files
# for input_file in INPUTS:
#     if os.path.isfile(input_file):
#         run.save(input_file, 'input')
#     elif os.path.isdir(input_file):
#         run.save_directory(input_file, 'input', 'text/plain', preserve_path=True)
#     else:
#         print('ERROR: input file %s does not exist' % input_file)
#
#
# # Save output files
# for output_file in OUTPUTS:
#     if os.path.isfile(output_file):
#         run.save(output_file, 'output')
#     elif os.path.isdir(output_file):
#         run.save_directory(output_file, 'output', 'text/plain', preserve_path=True)
#     else:
#         print('ERROR: output file %s does not exist' % output_file)
#
# run.close()
#
# # %%
