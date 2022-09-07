"""
================================
Title:  main_lin_sim.py
Author: Antony Pulikapparambil
Date:   May 2022
================================
"""
# Run this code to test PID controllers using linear system model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pid_controller import PositionDiscretePID, VelocityDiscretePID
from datetime import datetime
from time import sleep

print("\n=========== Ball-in-Tube Model Testing ===========")
print('\nConducting PID control test')

sample_num = 200
sample_time = 0.40               # units: sec
print(f"Sample Number: {sample_num}")

# Initialize several zero-arrays below
height, height_sp, time_arr = (np.array([0.0] * sample_num) for y in range(3))
fan_spd_factor = np.array([0] * sample_num)

# Select type of discrete PID controller
pos_or_vel_PID = input("Select PID controller type:\n[1] Position-form\n[2] Velocity-form\n")

# Create a separate controller instance for each tube, to allow for separate PID tuning
if pos_or_vel_PID == '1':
    PID_type = 'Position-form'
    pidController = PositionDiscretePID(sample_num)
else:
    PID_type = 'Velocity-form'
    pidController = VelocityDiscretePID(sample_num)
    print(f"\nPID type: {PID_type}")

# PID parameters
pidController.set_kc(1.000)  # Proportional gain
pidController.set_tau_i(1.500)  # Integral gain
pidController.set_tau_d(0.000)  # Derivative gain
pidController.set_upper_limit(100)  # Controller upper output limit (max fan speed factor)
pidController.set_lower_limit(0)  # Controller lower output limit (min fan speed factor)
pidController.set_ts(sample_time)  # Sampling period (units: seconds)
print("=========== PID values ============")
pidController.print_param()

# Define height set points (units: cm)
for r in range(sample_num):
    if 0 <= r < 50:
        height_sp[r] = 90
    elif 50 <= r < 100:
        height_sp[r] = 50
    elif 100 <= r < 150:
        height_sp[r] = 80
    elif 150 <= r < sample_num:
        height_sp[r] = 40

# STATE AND INPUT MODEL COEFFICIENTS
# ***************
A = 0.85
B = 0.5
# ***************

for i in range(sample_num):
    time_arr[i] = round((i + 1) * sample_time, 1)
    print(
        f"==== Sample: {i + 1}/{sample_num} || Time: {time_arr[i]}/{sample_time * sample_num} s ==== ")

    if i == 0:
        height[i] = 0.0
    else:
        height[i] = round(A*height[i-1] + B*fan_spd_factor[i-1], 1)

    # PID controller generates output using error between current height and set point
    # Update fan speeds based on the controller output
    if PID_type == 'Position-form':
        fan_spd_factor[i] = pidController.update(height[i], height_sp[i], i)
    elif PID_type == 'Velocity-form':
        fan_spd_factor[i] = pidController.update(height[i], height_sp[i], i) + fan_spd_factor[i - 1]
        fan_spd_factor[i] = pidController.saturate_op(fan_spd_factor[i])

    print(f"Fan Speed Factor = {fan_spd_factor[i]}")
    print(f"Ball Height = {height[i]} cm")
    print(f"Set Height = {height_sp[i]} cm")
    print("===============================================")
# ======== End of PID control algorithm for-loop ========

# Calculate the MSEs in ball height, for tubes 3 & 4
error = height - height_sp
mse = round((error @ error) / sample_num, 2)


# ========== SAVING the data ==========
# Record current date/time (to include in data/plots)
curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# Check if data storage folder exists and create if needed
storage_exists = os.path.exists('./pid-data')
if not storage_exists:
    os.makedirs('./pid-data')
    sleep(1)

# Save the tube 4 PID control test data as .csv files
df = pd.DataFrame({"Time (s)": time_arr, "Set Point (cm)": height_sp, "Ball Height (cm)": height,
                   "Fan Factor": fan_spd_factor})
df.to_csv("pid-data/pid-data-model.csv", index=False)

# Prepend the tube 4 PID values and MSE to the csv file
with open("pid-data/pid-data-model.csv", 'r') as rawf:
    data = rawf.read()
with open("pid-data/pid-data-model.csv", 'w') as newf:
    newf.write(f"Model PID Test ({curr_datetime}) \n[A, B] = [{A}, {B}]"
               f"\nPID type: {PID_type} \nKc = {pidController.get_kc()}"
               f"\ntau_i = {pidController.get_tau_i()} \ntau_d = {pidController.get_tau_d()} \nMSE = {mse}\n" + data)


# ========== PLOTTING the data ==========
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex='all')
h, = ax[0].plot(time_arr, height, '.-', color='tab:blue', label='Current Height')
sp, = ax[0].plot(time_arr, height_sp, '-', color='tab:red', label='Set Point Height')
ff, = ax[1].plot(time_arr, fan_spd_factor, '.-', color='tab:green', label='Fan Speed Factor')

ax[0].set_xlabel("Time (s)")                                # Label/color plots and axis labels
ax[0].set_ylabel("Height (cm)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Fan Speed Factor")
ax[0].set_ylim([-10, 110])
ax[0].set_title(f'Model (A = {A}, B = {B}) - Ball Height vs. Time (MSE: {mse})')
ax[1].set_title(f'Model - Fan Speed vs. Time (Kc = {pidController.get_kc()}, '
                f'\u03c4_i = {pidController.get_tau_i()}, \u03c4_d = {pidController.get_tau_d()})')

# Display current date/time on figure title
fig.suptitle(f'BiT Model - {PID_type} PID Control ({curr_datetime})', fontsize=16)

ax[0].legend()                                              # Include a legend
fig.tight_layout()
fig.subplots_adjust(top=0.88)                              # Translate plots downward to make super-title space
plt.savefig("pid-data/pid-model-plot.png", bbox_inches='tight')     # Save the plot as a pdf file
plt.show()
