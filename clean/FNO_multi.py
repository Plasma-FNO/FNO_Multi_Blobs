# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum

Multivariable FNO modelling gyrokinetic data. 2 variables along the x-y axis with autoregressive time roll-outs. 
"""
# %%
configuration = {"Case": 'Multi-Blobs',
                 "Field": 'rho, Phi, T',
                 "Field_Mixing": 'Channel',
                 "Type": '2D Time',
                 "Epochs": 500,
                 "Batch Size": 5,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max. Same',
                 "Instance Norm": 'No',
                 "Log Normalisation": 'No',
                 "Physics Normalisation": 'Yes',
                 "T_in": 10,
                 "T_out": 40,
                 "Step":5,
                 "Modes": 16,
                 "Width_time": 32,  # FNO
                 "Width_vars": 0,  # U-Net
                 "Variables": 3,
                 "Noise": 0.0,
                 "Loss Function": 'LP Loss',
                 "Spatial Resolution": 1,
                 "Temporal Resolution": 1,
                 "Gradient Clipping Norm": None,
                 "Ntrain": 250
                 #  "UQ": 'Dropout',
                 #  "Dropout Rate": 0.9
                 }

# %%
from simvue import Run
run = Run(mode='online')
run.init(folder="/FNO_MHD/pre_IAEA", tags=['Multi-Blobs', 'MultiVariable', "Z_Li", "Skip-connect", "Diff", "Recon"], metadata=configuration)

# %%
import os

#Saving the code. 
run.save(os.path.abspath(__file__), 'code')
run.save(os.getcwd() + '/utils.py', 'code')
# %%

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

import time
from timeit import default_timer
from tqdm import tqdm

from utils import *
torch.manual_seed(0)
np.random.seed(0)

# %%
path = os.getcwd()
data_loc = '/rds/project/iris_vol2/rds-ukaea-ap001/ir-gopa2/Data'
model_loc = os.getcwd() + '/Models'
file_loc = os.getcwd()
plot_loc = os.getcwd() + '/Plots'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Extracting the configuration settings

modes = configuration['Modes']
width_time = configuration['Width_time']
width_vars = configuration['Width_vars']
batch_size = configuration['Batch Size']
T_in = configuration['T_in']
T_out = configuration['T_out']
step = configuration['Step']
output_size = step
num_vars = configuration['Variables']

# %%
################################################################
# Loading Data
################################################################
t1 = default_timer()

data = data_loc + '/FNO_MHD_data_multi_blob_2000_T50.npz' #2000 simulation dataset

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

x_grid = np.load(data)['Rgrid'][0, :].astype(np.float32)
y_grid = np.load(data)['Zgrid'][:, 0].astype(np.float32)
t_grid = np.load(data)['time'].astype(np.float32)

t_res = configuration['Temporal Resolution']
x_res = len(x_grid) // configuration['Spatial Resolution']

uvp = torch.stack((u, v, p), dim=1)[:, ::t_res]
# uvp = np.delete(uvp, (153, 229), axis=0)  # Outlier T values
uvp = np.delete(uvp, (11, 160, 222, 273, 303, 357, 620, 797, 983, 1275, 1391, 1458, 1554, 1600, 1613, 1888, 1937, 1946, 1959), axis=0) #2000 dataset

# #Padding Removed
# uvp = uvp[:, :, 3:-3, 3:-3, :]
# x_grid = x_grid[3:-3]
# y_grid = y_grid[3:-3]

ntrain = configuration['Ntrain'] # 2000 dataset
ntest = 85 #2000 dataset

S = uvp.shape[3]  # Grid Size
size_x = S
size_y = S

batch_size = configuration['Batch Size']

batch_size2 = batch_size


train_a = uvp[:ntrain, :, :, :, :T_in]
train_u = uvp[:ntrain, :, :, :, T_in:T_out + T_in]

test_a = uvp[-ntest:, :, :, :, :T_in]
test_u = uvp[-ntest:, :, :, :, T_in:T_out + T_in]


print("Training Input: " + str(train_a.shape))
print("Training Output: " + str(train_u.shape))

# %%
#Normalisations 
 
#Normalising the Input Data 
a_normalizer = MinMax_Normalizer_variable(uvp[...,:T_in])
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

#Normalising the Ouput Data 
y_normalizer = MinMax_Normalizer_variable(uvp[...,T_in:T_out+T_in])
train_u = y_normalizer.encode(train_u)
test_u_encoded = y_normalizer.encode(test_u)

# %%
#Data Loaders 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size,
                                          shuffle=False)
t2 = default_timer()
print('preprocessing finished, time used:', t2 - t1)

# %%
################################################################
# Model and Optimizer setup
################################################################
model = FNO_multi(T_in, step, modes, modes, num_vars, width_vars, width_time)
model.to(device)
run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

# optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'],
                                            gamma=configuration['Scheduler Gamma'])

loss_func = LpLoss(size_average=False)

epochs = configuration['Epochs']

#Misc
if torch.cuda.is_available():
    y_normalizer.cuda()
max_grad_clip_norm = configuration['Gradient Clipping Norm']

# %%
################################################################
# Training 
################################################################
start_time = time.time()
for ep in tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for xx, yy in train_loader:
        optimizer.zero_grad()
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        y_old = xx[..., -step:]

        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)

            #Recon Loss
            loss += loss_func(im.reshape(xx.shape[0], -1), y.reshape(xx.shape[0], -1))

            #Residual Loss
            pred_diff = im - xx[..., -step:]
            y_diff = y - y_old
            loss += loss_func(pred_diff.reshape(xx.shape[0]*num_vars, x_res, x_res, step), y_diff.reshape(xx.shape[0]*num_vars, x_res, x_res, step))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            y_old = y

        train_l2 += loss.item()

        loss.backward()
        
        #Gradient Clipping 
        # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_grad_clip_norm, norm_type=2.0)
        
        optimizer.step()
        scheduler.step()

    train_loss = train_l2 / ntrain 
    train_loss_step = train_loss / ntrain / (T_out / step)

    # Validation Loop
    test_loss = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)

            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                out = model(xx)

                if t == 0:
                    pred = out
                else:
                    pred = torch.cat((pred, out), -1)

                xx = torch.cat((xx[..., step:], out), dim=-1)
            test_loss += loss_func(pred, yy).item()
            
        test_loss = test_loss / ntest 

    t2 = default_timer()

    print('Epochs: %d, Time: %.2f, Train Loss per step: %.3e, Train Loss: %.3e, Test Loss: %.3e' % (
    ep, t2 - t1, train_loss_step, train_loss, test_loss))

    run.log_metrics({'Train Loss': train_loss,
                     'Train Loss per Step': train_loss_step,
                    'Test Loss': test_loss})

train_time = time.time() - start_time
# %%
#Saving the Model
model_loc = file_loc + '/Models/FNO_multi_blobs_' + run.name + '.pth'
torch.save(model.state_dict(),  model_loc)
run.save(model_loc, 'output')

# %%
# Testing
batch_size = 1
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=1,
                                          shuffle=False)
pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        t1 = default_timer()
        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            out = model(xx)

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)

            xx = torch.cat((xx[..., step:], out), dim=-1)

        t2 = default_timer()
        pred_set[index] = pred
        index += 1
        print(t2 - t1)

# %%
print(pred_set.shape, test_u.shape)
# Logging Metrics
MSE_error = (pred_set - test_u_encoded).pow(2).mean()
MAE_error = torch.abs(pred_set - test_u_encoded).mean()

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))
# print('(LP) Testing Error: %.3e' % (LP_error))
# print('(MAPE) Testing Error %.3e' % (rel_error))
# print('(NMSE) Testing Error %.3e' % (nmse))
# print('(NRMSE) Testing Error %.3e' % (nrmse))

# %% 
run.update_metadata({'Training Time': float(train_time),
                     'MSE': float(MSE_error),
                     'MAE': float(MAE_error)
                    })

pred_set = y_normalizer.decode(pred_set.to(device)).cpu()

pred_set_encoded = pred_set
pred_set = y_normalizer.decode(pred_set.to(device)).cpu()

#Normalised MSE
nmse= 0 
for ii in range(num_vars):
    nmse += (pred_set[:,ii] - test_u[:,ii]).pow(2).mean() / test_u[:,ii].pow(2).mean()
    # print(test_u[:,ii].pow(2).mean())

print('(NMSE) Testing Error %.3e' % (nmse))
run.update_metadata({'NMSE': float(nmse)})
# %%
# Plotting the comparison plots

idx = np.random.randint(0, ntest)
idx = 0

output_plot = []
for dim in range(num_vars):
    u_field = test_u[idx]

    v_min_1 = torch.min(u_field[dim, :, :, 0])
    v_max_1 = torch.max(u_field[dim, :, :, 0])

    v_min_2 = torch.min(u_field[dim, :, :, int(T_out/ 2)])
    v_max_2 = torch.max(u_field[dim, :, :, int(T_out/ 2)])

    v_min_3 = torch.min(u_field[dim, :, :, -1])
    v_max_3 = torch.max(u_field[dim, :, :, -1])

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 3, 1)
    pcm = ax.imshow(u_field[dim, :, :, 0], cmap=mpl.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t=' + str(T_in))
    ax.set_ylabel('Solution')
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 2)
    pcm = ax.imshow(u_field[dim, :, :, int(T_out/ 2)], cmap=mpl.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t=' + str(int((T_out+ T_in) / 2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 3)
    pcm = ax.imshow(u_field[dim, :, :, -1], cmap=mpl.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t=' + str(T_out+ T_in))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    u_field = pred_set[idx]

    ax = fig.add_subplot(2, 3, 4)
    pcm = ax.imshow(u_field[dim, :, :, 0], cmap=mpl.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    ax.set_ylabel('FNO')

    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 5)
    pcm = ax.imshow(u_field[dim, :, :, int(T_out/ 2)], cmap=mpl.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2,
                    vmax=v_max_2)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    ax = fig.add_subplot(2, 3, 6)
    pcm = ax.imshow(u_field[dim, :, :, -1], cmap=mpl.cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    fig.colorbar(pcm, pad=0.05)

    plt.title(dims[dim])

    plot_name = plot_loc + '/' + dims[dim] + '_' + run.name + '.png'
    plt.savefig(plot_name)
    run.save(plot_name, 'output')

run.close()

# %%
