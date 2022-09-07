"""
================================
Title:  main_nonlin_sim.py
Author: Antony Pulikapparambil
Date:   June 2022
================================
"""
# Run this code to test nonlinear MPC and unscented Kalman filter, using a nonlinear
# system model.

import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpc_kalman import nl_mpc, nl_system_model, UnscentedKalmanFilter, SimplexSigmaPoints, h_cv
from datetime import datetime
from time import sleep

matplotlib.rcParams.update(
    {'font.family': 'Times New Roman',
     'font.serif': 'Times New Roman',
     'mathtext.rm': 'Times New Roman',
     'mathtext.fontset': 'stix'})

print("\n===== Ball-in-Tube Nonlinear SISO Model Testing =====")
print('\nConducting MPC test...')

sample_num = 400
sample_time = 0.5  # units: sec
pred_horizon = 2
print(f"Sample Number: {sample_num}")
print(f"MPC Prediction Horizon: {pred_horizon}\n")

# Kalman filter covariance matrices (can tune as needed)
var_n = 9.5     # 'Kalman Q' - lower => trust model
var_e = 0.2     # 'Kalman R' - lower => trust measurement

# Coefficients in cost function J (choose how much to penalize high error or high controller /
# change-in controller output)
Q = 2.26
R = 0.01

P = 'none'       # Set P to string type to ignore terminal cost

# Lower and upper limits of controller (used in optimizer)
umax = 100.0
umin = 0.0

# Create unscented Kalman filter
points = SimplexSigmaPoints(n=1, alpha=0.3)
ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, fx=nl_system_model, hx=h_cv, dt=1., points=points)

sp_offset = 0.1  # 10 percent of the sp (used to increase tube cost as output exceeds 10% of set point)
sqrt_cost = True

# Initialize several zero-arrays below
true_height, meas_height, est_height, height_sp, time_arr, cost,\
    actual_cost, tube_cost = (np.array([0.0] * (sample_num+1)) for y in range(8))

fan_spd_factor = np.array([0] * (sample_num+1))

 # Define height set points (units: cm)
sps = np.array([70, 40, 60, 30]) 
sp_index = 0
height_sp[0] = sps[sp_index]

# Initialize UKF parameters
ukf.x, ukf.Q, ukf.R = meas_height[0], var_n, var_e
est_height[0] = meas_height[0]

 # Calculate initial mpc controller output
uk, _ = nl_mpc(Q, R, est_height[0], pred_horizon, umin, umax, xk_sp=height_sp[0])
fan_spd_factor[0] = np.round(uk, 0)

# Run MPC algorithm for "sample number" of iterations
for i in range(1, sample_num + 1):
    print(f"===== Sample: {i}/{sample_num} =====")
    time_arr[i] = round(i * sample_time, 1)

    # Obtain true height from system model
    true_height[i], _ = nl_system_model(true_height[i-1], fan_spd_factor[i-1])

    # Measurement noise added
    meas_height[i] = true_height[i] + np.random.normal(0, 1.0)

    # Unscented Kalman filter estimates height
    ukf.predict(), ukf.update(meas_height[i])
    est_height[i] = ukf.x

    # Set point assignment
    height_sp[i] = sps[sp_index]

    # Calculate optimal controller output (fan speed factors)
    uk, cost[i] = nl_mpc(Q, R, est_height[i], pred_horizon, umin, umax, xk_sp=height_sp[i], P=P)
    fan_spd_factor[i] = np.round(uk, 0)

    print(f"Fan Speed Factor = {fan_spd_factor[i]}")
    print(f"Meas. Ball Height = {round(meas_height[i], 1)} cm")
    print(f"Est. Ball Height = {round(est_height[i], 1)} cm")
    print(f"Set Height = {height_sp[i]} cm")
    print("============================")

    # Calculate actual cost
    if sqrt_cost:
        actual_cost[i] = np.log(
            (meas_height[i] - height_sp[i]) * Q * (meas_height[i] - height_sp[i])) + np.log(uk * R * uk)
    else:
        actual_cost[i] = (meas_height[i] - height_sp[i]) * Q * (meas_height[i] - height_sp[i]) + uk * R * uk
    
    # Calculate tube cost
    if meas_height[i] > height_sp[i] * (1 + sp_offset):
        tube_cost[i] += abs(meas_height[i] - height_sp[i] * (1 + sp_offset))
    elif meas_height[i] < height_sp[i] * (1 - sp_offset):
        tube_cost[i] += abs(meas_height[i] - height_sp[i] * (1 - sp_offset))

    # Set point change
    if i > 0 and i % 100 == 0:  # set point change frequency (units: samples)
        sp_index += 1
        if sp_index > len(sps) - 1:  # loop around set point array if its end is reached
            sp_index = 0
# ==== End of MPC for-loop ==== #

# MSE b/w set point and estimated height
mse_est_sp = np.mean(np.square(est_height - height_sp))
print(f'Est_h vs. Set_h MSE = {round(mse_est_sp, 2)}')

# MSE b/w set point and true height
mse_true_sp = np.mean(np.square(true_height - height_sp))
print(f'True_h vs. Set_h MSE = {round(mse_true_sp, 2)}')

# Record current date/time (to include in plots)
curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# ==== Plotting the data ====
# Check if data storage folder exists and create if needed

storage_exists = os.path.exists('./nl-mpc-data')
if not storage_exists:
    os.makedirs('./nl-mpc-data')
    sleep(1)

fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex='all', constrained_layout=True)

ax[0].plot(time_arr,meas_height, '.', color='tab:blue', alpha=0.5, label='Measured Height')
ax[0].plot(time_arr, est_height, '-', color='tab:green', alpha=0.5, label='Est. Height')
ax[0].plot(time_arr, height_sp, '-', color='tab:red', alpha=0.5, label='Set Point (MSE = {:.2f})'.format(mse_est_sp))
ax[0].set_ylim([-10, 110])
ax[0].set_title(f'Ball Heights [cm] (UKF_Q = {var_n}, UKF_R = {var_e})')
ax[0].legend()

ax[1].plot(time_arr, fan_spd_factor, '.-', color='tab:red', alpha=0.5)
ax[1].set_ylim([-10, 110]) 
ax[1].set_title(f'Fan Speed Factor (MPC_Q = {Q}, MPC_R = {R}, MPC_N = {pred_horizon})')

ax[2].plot(time_arr, actual_cost, '.', label='$J$', color='tab:red', alpha=0.5)
ax1 = ax[2].twinx()
ax1.plot(time_arr, tube_cost, '.', label='r', color='tab:blue', alpha=0.5)
ax[2].plot(np.nan, '.', label='$r$', color='tab:blue', alpha=0.5)
ax[2].legend(loc=1)
if sqrt_cost:
    ax[2].set_ylabel(r'Actual $\sqrt{COST}$')
else:
    ax[2].set_ylabel(r'Actual Cost')
ax1.set_ylabel('Tube Cost')
ax[2].yaxis.label.set_color('tab:red')
ax1.yaxis.label.set_color('tab:blue')
ax[2].tick_params(axis='y', colors='tab:red')
ax1.tick_params(axis='y', colors='tab:blue')
ax[2].set_title('Costs')

ax[2].set_xlabel('Time (s)')
fig.suptitle(f'BiT Simulation - MPC Test ({curr_datetime})', fontsize=16)

plt.savefig('nl-mpc-data/nl-mpc-model-plot.png', bbox_inches='tight')  # Save the plot as a pdf file
plt.show()
