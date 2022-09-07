"""
================================
Title:  pysindy-id.py
Author: Antony Pulikapparambil
Date:   June 2022
================================
"""
# Runs nonlinear data-driven system identification using PySINDy.
# PySINDy paper: https://arxiv.org/pdf/2111.08481.pdf

# NOTE: to run this code, you need to create a '/PySINDy_Data' subfolder
# and create/load your own data

# The data should be 'csv' files with four columns.
# The first row of the data will be skipped (so place a title here)
# col 1: Time
# col 2: Input (e.g., fan inputs)
# col 3: Output (e.g., ball heights) 

import matplotlib, pprint, sys, os
import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt

def plot_model_data_vs_time(time, data_h, sim_h, input_ff, mse, data_type):
    """ Plot the model response vs. experimental data response to same input """
    if data_type == 'train':
        title_data_type = 'Training'
    elif data_type == 'val':
        title_data_type = 'Validation'
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    ax[0].plot(time, data_h, "b", label="Experiment", alpha=0.4, linewidth=3)
    ax[0].plot(time, sim_h, "k--", label="SINDy model", linewidth=3)
    ax[0].set_ylabel("Ball Height (cm)")
    ax[0].set_title("Ball Height vs. Time")
    ax[0].legend()

    ax[1].plot(time, input_ff, "r", alpha=0.6, linewidth=3)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Fan Speed Factor")
    ax[1].set_title("Fan Speed Factor vs. Time")

    fig.suptitle(f'PySINDy Model vs. {title_data_type} Data (MSE = {mse})', fontsize=16)
    fig.subplots_adjust(top=0.88)
    plt.savefig(f"pysindy-model-rgs-{data_type}.png", bbox_inches='tight')          # Save the plot as a png file


def plot_model_vs_data(data_train, sim_train, data_val, sim_val):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    ax[0].plot(data_train, sim_train, '.', color='tab:blue')
    ax[0].plot([0, 100], [0, 100], "--", color='tab:gray', label="Ideal", linewidth=2)
    ax[0].set_xlabel('Experimental Ball Heights (cm)')
    ax[0].set_ylabel('Model Ball Heights (cm)')
    ax[0].set_title('Model vs. Training Data')
    ax[0].legend()

    ax[1].plot(data_val, sim_val, '.', color='tab:green')
    ax[1].plot([0, 100], [0, 100], "--", color='tab:gray', label="Ideal", linewidth=2)
    ax[1].set_xlabel('Experimental Ball Heights (cm)')
    ax[1].set_ylabel('Model Ball Heights (cm)')
    ax[1].set_title('Model vs. Validation Data')
    ax[1].legend()

    fig.suptitle('PySINDy Model vs. Experimental Data', fontsize=16)
    fig.subplots_adjust(top=0.88)


def plot_expt_data(data_h, input_ff, data_type):
    """ Plot  x[k] vs. u[k] vs. x[k+1] for experimental data """
    if data_type == 'train':
        title_data_type = 'Training'
    elif data_type == 'val':
        title_data_type = 'Validation'
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    
    x_k = data_h[0:data_h.size-2]         # x[k]
    u_k = input_ff[0:data_h.size-2]       # u[k]
    x_kp1 = data_h[1:data_h.size-1]       # x[k+1]

    ax[0].plot(x_k, x_kp1, '.', color='tab:blue')
    ax[0].set_xlabel('x [k] (cm)')
    ax[0].set_ylabel('x [k+1] (cm)')
    ax[0].set_title('Next State vs. Current State')

    ax[1].plot(u_k, x_kp1, '.', color='tab:purple')
    ax[1].set_xlabel('u [k]')
    ax[1].set_ylabel('x [k+1] (cm)')
    ax[1].set_title('Next State vs. Current Input')

    fig.suptitle(f'{title_data_type} Data', fontsize=16)
    fig.subplots_adjust(top=0.88)


# ===== START of main program ===== #

# ** Ensure there are training/validation data sets inside the project directory ** 

# Choose csv files
train_file = 'PySINDy_Data/test-data1.csv'  # location of training data
val_file = 'PySINDy_Data/test-data2.csv'    # location of validation data

# Read training data from csv file
df_time = pd.read_csv(train_file, skiprows=2, dtype=float, usecols=[0])   # Time (sec)
df_ff = pd.read_csv(train_file, skiprows=2, dtype=int, usecols=[1])       # Fan factor
df_h = pd.read_csv(train_file, skiprows=2, dtype=float, usecols=[2])      # Ball height (cm)

# Convert dataframes to arrays
time_arr = np.squeeze(df_time.to_numpy())
fan_factor_arr = np.squeeze(df_ff.to_numpy()) 
height_arr = np.squeeze(df_h.to_numpy())

# Read validation data from csv file
df_time_v = pd.read_csv(val_file, skiprows=2, dtype=float, usecols=[0])   # Time (sec)
df_ff_v = pd.read_csv(val_file, skiprows=2, dtype=int, usecols=[1])       # Fan factor
df_h_v = pd.read_csv(val_file, skiprows=2, dtype=float, usecols=[2])      # Ball height (cm)

# Convert dataframes to arrays
time_arr_val = np.squeeze(df_time_v.to_numpy())
fan_factor_arr_val = np.squeeze(df_ff_v.to_numpy()) 
height_arr_val = np.squeeze(df_h_v.to_numpy())

removeMean = False            # Remove mean from input/output data?
do_3d_data_plot = True        # Show plot of x[k] and u[k] vs. x[k+1]?
do_model_vs_data_plot = True  # Show plot of x_model vs. x_data?

# == *OPTIONAL* : Remove MEAN from data == #
if removeMean:
    fan_factor_arr = fan_factor_arr - np.mean(fan_factor_arr)
    fan_factor_arr_val = fan_factor_arr_val - np.mean(fan_factor_arr_val)
    height_arr = height_arr - np.mean(height_arr)
    height_arr_val = height_arr_val - np.mean(height_arr_val)


# == Create PySINDy object and fit model == #
model = ps.SINDy(
    discrete_time=True,
    differentiation_method=ps.FiniteDifference(order=2),
    feature_library=ps.PolynomialLibrary(degree=3,include_bias=True),
    optimizer=ps.STLSQ(threshold=0.1),              # 'threshold' determines how sparse the chosen basis set should be
    feature_names=["x", "u"],
)

model.fit(x=height_arr, u=fan_factor_arr, t=1)       # 't' is timestep in data

print('\nFeature Library:')
print(model.get_feature_names())

print('Model:')
model.print()

# Create/open a text file to save data
sys.stdout = open('pysindy_model.txt', 'w+')
print('Model:')
model.print()
print('\nFeature Library:')
print(model.get_feature_names())
print(f'\Training Data is Detrended?: {removeMean}')
print('\nPySINDy Parameters:')
pprint.pprint(model.get_params())
sys.stdout.close()

# Reopen the stdout onto the terminal
sys.stdout = os.fdopen(1, 'w', 1)

# == *OPTIONAL* : Plot 3D training and validation data
if do_3d_data_plot:
    # Plot x[k] vs. u[k] vs. x[k+1] for training data
    plot_expt_data(height_arr, fan_factor_arr, 'train')

if do_3d_data_plot:
    # Plot x[k] vs. u[k] vs. x[k+1] for validation data
    plot_expt_data(height_arr_val, fan_factor_arr_val, 'val')

# == Plot model and training/validation data vs. time
# TRAINING DATA
# Simulate the model's response to training input data
x0 = height_arr[0]           # Initial ball height
sim = model.simulate(x0=x0, u=fan_factor_arr, t=time_arr.size)

# Calculate the mean squared error between model and training data
mse_train = round(np.mean(np.square(height_arr - sim)), 2)

# VALIDATION DATA
# Simulate the model's response to validation data
x0 = height_arr_val[0]           # Initial ball height
sim_val = model.simulate(x0=x0, u=fan_factor_arr_val, t=time_arr_val.size)

# Calculate offset between model and val_data
val_offset = np.mean(sim_val - height_arr_val)
print(f'\nModel vs. Val. Data Offset: {round(val_offset, 2)}')
# sim_val = sim_val - val_offset

# Calculate the mean squared error between model and val_data
mse_val = round(np.mean(np.square(height_arr_val - sim_val)), 2)

plot_model_data_vs_time(time_arr, height_arr, sim, fan_factor_arr, mse_train, 'train')
plot_model_data_vs_time(time_arr_val, height_arr_val, sim_val, fan_factor_arr_val, mse_val, 'val')

# == *OPTIONAL* : Plot model vs. training/validation data
if do_model_vs_data_plot:
    plot_model_vs_data(height_arr, sim_clip, height_arr_val, sim_val_clip)
    plt.show() 
