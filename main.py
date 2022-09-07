"""
================================
Title:  main.py
Author: Antony Pulikapparambil
Date:   May 2022
================================
"""
# Runs control algorithms for the ball-in-tube experiment
# NOTE 1: code requires connection to experiment to run 

import numpy as np
import pandas as pd
# import pyswarms as ps  # import causes warning during runtime (use for adaptive MPC)
import padasip as pa
from helper import BallsInTubesController, plot_pid_control_test, plot_mpc_test, plot_nonlin_mpc, plot_io_test
from pid_controller import PositionDiscretePID, VelocityDiscretePID
from mpc_kalman import mpc, opt_func, myKalmanFilter, nl_mpc, nl_system_model, UnscentedKalmanFilter,\
    SimplexSigmaPoints, h_cv
from time import sleep, perf_counter

# set RabbitMQ server to send and receive signals from

host = '_._._._'    # use IPv4 addr of 64-bit PC that runs RabbitMQ service
cont = BallsInTubesController(host)

# Try-except clause to lower fan speeds and safely exit program if Ctrl+C pressed in command prompt
try:
    sleep(0.1)
    print(" \n=========== Ball-in-Tube Experiment ===========")
    print("Press Ctrl+C to end the process at any time")
    # Prompt user to choose whether to calibrate or use existing level-to-height conversion parameters
    doCalibrate = input("\nWould you like to calibrate the level readings? [y]/[n]: ")
    if doCalibrate == 'y':
        # Calibrate level readings; pass calibration sample number and sample period
        bot_levels, level_gain = cont.calibrate_level(10, 0.5)
        calibr_status = 'Calibrated'
    else:
        cont.lower_fan_speeds()
        # Use parameters from a previous calibration
        bot_levels, level_gain = [13369, 13422, 0, 70153], [0.007719, 0.007682, None, None]
        print("Skipping calibration. Using preset level conversion values:")
        print(f"Bottom Levels(4:1) = {bot_levels}")
        print(f"Level Gains(4:1) = {level_gain}")
        calibr_status = 'Not Calibrated'

    # Let user select mode
    mode = input("\nSelect mode: \n[1] PID Control Test \n[2] Linear MPC Test \n[3] Adaptive MPC Test"
                 "\n[4] Nonlinear MPC Test \n[5] Step Test \n[6] PRBS Test \n[7] RGS Test \n[8] Read Levels"
                 "\n[9] Quit\n")

    #  == 1. PID control test == #
    if mode == '1':
        print('\nConducting PID control test')
        sample_num = 240
        desired_sample_time = 0.50  # units: sec (minimum = ~0.4)
        print(f"Sample Number: {sample_num}")
        print(f"Estimated Time: {sample_num * desired_sample_time} s\n")

        # Initialize several zero-arrays below (NOTE: tubes 1 & 2 excluded since level sensors broken)
        height4, height3, time_arr, curr_t_arr = (np.array([0.0] * sample_num) for y in range(4))
        fan_spd_factor4, fan_speed4, fan_spd_factor3, fan_speed3 = (np.array([0] * sample_num) for y in range(4))
        sample_t_arr = np.array([0.0] * (sample_num - 1))
        height_sp = np.zeros((2, sample_num))

        # Select type of discrete PID controller
        pos_or_vel_PID = input("Select PID controller type:\n[1] Position-form\n[2] Velocity-form\n")

        # Create a separate controller instance for each tube, to allow for separate PID tuning
        if pos_or_vel_PID == '1':
            PID_type = 'Position-form'
            pidController4, pidController3 = (PositionDiscretePID(sample_num) for c in range(2))
        else:
            PID_type = 'Velocity-form'
            pidController4, pidController3 = (VelocityDiscretePID(sample_num) for c in range(2))

        print(f"\nPID type: {PID_type}")
        # Tube 4 - PID parameters
        pidController4.set_kc(0.6)  # Controller gain
        pidController4.set_tau_i(5.00)  # Integral time constant
        pidController4.set_tau_d(0.1)  # Derivative time constant
        pidController4.set_upper_limit(100)  # Controller upper output limit (max fan speed factor)
        pidController4.set_lower_limit(0)  # Controller lower output limit (min fan speed factor)
        pidController4.set_ts(desired_sample_time)  # Sampling period (units: seconds)
        print("============ Tube 4 ============")
        pidController4.print_param()

        # Tube 3 - PID parameters
        pidController3.set_kc(0.6)  # Controller gain
        pidController3.set_tau_i(12.5/2)  # Integral time constant
        pidController3.set_tau_d(0.0)  # Derivative time constant
        pidController3.set_upper_limit(100)  # Controller upper output limit (max fan speed factor)
        pidController3.set_lower_limit(0)  # Controller lower output limit (min fan speed factor)
        pidController3.set_ts(desired_sample_time)  # Sampling period (units: seconds)
        print("============ Tube 3 ============")
        pidController3.print_param()
        sleep(1)

        # Define height set points (units: cm)
        sps = np.array([[70, 70], [40, 40], [60, 60], [30, 30]])
        sp_index = 0
        sp_freq = 60  # Set point change frequency (unit: samples)
        height_sp[:, [0]] = sps[[sp_index]].T

        curr_height = [0, 0, 0, 0]  # [Tube 4, Tube 3, Tube 2, Tube 1]
        start_time = perf_counter()  # Start a counter before conducting PID-control test to measure time elapsed

        # Run the PID control algorithm for "sample number" of iterations
        for i in range(sample_num):
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Record current PID-control test time in array

            # Down-sampling algorithm: ensure samples are only taken during the specified sample times
            sleep_time = float((i + 1) * desired_sample_time) - time_arr[i]
            sleep(max(sleep_time, 0))  # Program sleeps for non-negative amount of time
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Overwrite current PID-control test time in array

            print(
                f"==== Sample: {i + 1}/{sample_num} || Time: {time_arr[i]}/{desired_sample_time * sample_num} s ==== ")

            height_sp[:, [i]] = sps[[sp_index]].T

            curr_levels = [cont.get_level(0), cont.get_level(1), cont.get_level(2), cont.get_level(3)]

            # Convert the ball-levels to ball-heights (distance b/w floor and ball in units of cm) in each tube
            for k in range(4):
                if level_gain[k] is not None:
                    curr_height[k] = round(((bot_levels[k] - curr_levels[k]) * level_gain[k]), 1)
                else:
                    curr_height[k] = None

            # PID controller generates output using error between current height and set point
            # Update fan speeds based on the controller output
            if PID_type == 'Position-form':
                fan_spd_factor4[i] = pidController4.update(curr_height[0], height_sp[0, [i]], i)
                fan_spd_factor3[i] = pidController3.update(curr_height[1], height_sp[1, [i]], i)
            elif PID_type == 'Velocity-form':
                fan_spd_factor4[i] = pidController4.update(curr_height[0], height_sp[0, [i]], i) + fan_spd_factor4[i - 1]
                fan_spd_factor4[i] = pidController4.saturate_op(fan_spd_factor4[i])
                fan_spd_factor3[i] = pidController3.update(curr_height[1], height_sp[1, [i]], i) + fan_spd_factor3[i - 1]
                fan_spd_factor3[i] = pidController3.saturate_op(fan_spd_factor3[i])

            cont.set_fan_speeds(fan_spd_factor4[i], 3)  # Set fan speed for tube 4
            cont.set_fan_speeds(fan_spd_factor3[i], 2)  # Set fan speed for tube 3

            # Measure the fan speeds immediately after applying the fan speed factor
            fan_speeds = cont.get_fan_speeds()
            fan_speed4[i] = fan_speeds[0]  # Record fan speed for tube 4
            fan_speed3[i] = fan_speeds[1]  # Record fan speed for tube 3
            print(f"Fan Speeds (4:1) = {fan_speeds} rpm")

            # Record the current program runtime
            curr_t_arr[i] = perf_counter()
            # Calculate the sample time using the difference in run times between two consecutive iterations
            if i > 0:
                sample_t_arr[i - 1] = round(curr_t_arr[i] - curr_t_arr[i - 1], 2)

            height4[i] = curr_height[0]
            height3[i] = curr_height[1]
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Record current step-test time in array

            print(f"Levels (4:1) = {curr_levels}")
            print(f"Ball Heights (4:1) = {curr_height} cm")
            print(f"Set Heights (4:1) = [{round(float(height_sp[0, [i]]), 1)}, {round(float(height_sp[1, [i]]), 1)}, None, None] cm")
            print("===============================================")

            # Set point change
            if i > 0 and i % sp_freq == 0:  # set point change frequency (units: samples)
                sp_index += 1
                if sp_index > len(sps) - 1:  # loop around set point array if its end is reached
                    sp_index = 0
        # ======== End of PID control algorithm for-loop ========

        sample_time_avg = np.mean(sample_t_arr)  # Calculate average sample time for PID control test
        sample_time_std = np.std(sample_t_arr)  # Calculate standard deviation in sample times
        print(f"Average Sample Time: {round(sample_time_avg, 2)}±{round(sample_time_std, 2)} s")

        # Calculate the MSEs in ball height, for tubes 3 & 4
        error_t4 = height4 - height_sp[0, :]
        mse_t4 = round((error_t4 @ error_t4) / sample_num, 2)
        error_t3 = height3 - height_sp[1, :]
        mse_t3 = round((error_t3 @ error_t3) / sample_num, 2)

        # Lower the fan speeds in all tubes after completing PID test
        cont.lower_fan_speeds()

        # Plot the ball height and fan speed data
        plot_pid_control_test(time_arr, height_sp, height4, fan_spd_factor4, fan_speed4, height3,
                              fan_spd_factor3, fan_speed3, calibr_status, mse_t4, mse_t3, pidController4,
                              pidController4, PID_type)

    # == 2. Linear MPC test and 3. Adaptive MPC test == #
    elif mode == '2' or mode == '3':
        if mode == '2':
            Linear_MPC = True
            Adaptive_MPC = False
            print('\nConducting Linear MPC test')
        else:
            Adaptive_MPC = True
            Linear_MPC = False
            print('\nConducting Adaptive MPC test')

        sample_num = 240
        desired_sample_time = 0.5  # units: sec
        pred_horizon = 3
        print(f"Sample Number: {sample_num}")
        print(f"Estimated Time: {sample_num * desired_sample_time} s")
        print(f"MPC Prediction Horizon: {pred_horizon}\n")

        # ==== RLS algorithm ==== #
        filt4 = pa.filters.FilterRLS(4, mu=0.99)  # 4 is the filter order, mu is the forgetting factor
        filt3 = pa.filters.FilterRLS(4, mu=0.99)  # 4 is the filter order, mu is the forgetting factor

        if Adaptive_MPC:
            import pyswarms as ps
            id_horizon = 15
            print(f"MPC Identification Horizon: {id_horizon}")
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # For A, B optimizer in adaptive MPC

        # Initialize several zero-arrays below (NOTE: tubes 1 & 2 excluded since level sensors broken)
        # NOTE: MPC makes predictions at least 2 samples ahead, so arrays are size 'sample_num + 2'
        time_arr, curr_t_arr, cost, tube_cost, actual_cost, AB_cost = (np.array([0.0] * (sample_num + 2)) for k in
                                                                       range(6))
        fan_speed4, fan_speed3 = (np.array([0] * (sample_num + 2)) for k in range(2))
        sample_t_arr = np.array([0.0] * (sample_num + 1))
        meas_height, est_height, fan_spd_factor, height_sp = (np.zeros((2, sample_num + 2)) for k in range(4))

        # State and input matrices for linear approx of BiT model
        Aoverall = np.array([[0.70, 0],
                             [0, 0.70]])
        Boverall = np.array([[0.5, -0.01],
                             [-0.01, 0.5]])
        # Aoverall = np.concatenate((filt4.w[:2], filt3.w[:2])).reshape(2, 2)  #
        # Boverall = np.concatenate((filt4.w[2:], filt3.w[2:])).reshape(2, 2)
        A0, B0 = Aoverall, Boverall

        sp_offset = 0.1  # 10 percent of the sp (used to increase tube cost as output exceeds 10% of set point)
        sqrt_cost = True

        # Kalman filter covariance matrices (can tune as needed)
        var_n = np.diag([1, 1])  # 'Kalman Q' - lower => trust model    def: 1, 1
        var_e = np.diag([0.5, 0.5])  # 'Kalman R' - lower => trust measurement def: 0.5, 0.5

        # Coefficients in cost function J (choose how much to penalize high error or high controller /
        # change-in controller output)
        Q = np.array([[3000, 500],
                      [500, 3000]])
        R = np.array([[1500, -1000],
                      [-1000, 1500]])

        # P = np.array([[1000, 0],  # Terminal cost
        #               [0, 1000]])
        P = 'none'       # Set P to scalar to ignore terminal cost

        # Lower and upper limits of controller (used in optimizer)
        umax = 100.0
        umin = 0.0

        kf2 = myKalmanFilter(dim_x=2, dim_z=2)  # Create instance of Kalman filter

        # Define height set points (units: cm)
        sps = np.array([[90, 90], [45, 45], [70, 70], [30, 30]])
        sp_index = 0
        sp_freq = 80   # Set point change frequency (unit: samples)
        height_sp[:, [0]] = sps[[sp_index]].T

        # Initialize Kalman filter parameters
        kf2.x, kf2.Q, kf2.R = meas_height[:, [0]], var_n, var_e
        kf2.H, kf2.F, kf2.B = np.array([[1, 0], [0, 1]]), Aoverall, Boverall
        est_height[:, [0]] = meas_height[:, [0]]

        # Calculate initial mpc controller output
        uk, _ = mpc(Q, R, est_height[:, [0]], pred_horizon, umin, umax, xk_sp=height_sp[:, [0]], A=Aoverall, B=Boverall)
        fan_spd_factor[:, [0]] = np.round(uk, 0)

        curr_height = [0, 0, 0, 0]  # [Tube 4, Tube 3, Tube 2, Tube 1]

        # Obtain initial measured heights
        curr_levels = [cont.get_level(0), cont.get_level(1), cont.get_level(2), cont.get_level(3)]

        # Convert the ball-levels to ball-heights (distance b/w floor and ball in units of cm) in each tube
        for k in range(4):
            if level_gain[k] is not None:
                curr_height[k] = round(((bot_levels[k] - curr_levels[k]) * level_gain[k]), 1)
            else:
                curr_height[k] = None

        meas_height[[0], [1]] = curr_height[0]  # Tube 4 IC
        meas_height[[1], [1]] = curr_height[1]  # Tube 3 IC

        start_time = perf_counter()  # Start a counter before conducting MPC test to measure time elapsed

        # Run the MPC algorithm for "sample number" of iterations
        for i in range(1, sample_num + 1):
            curr_t_arr[i - 1] = perf_counter()
            time_arr[i - 1] = round(curr_t_arr[i - 1] - start_time, 2)  # Record current MPC test time in array

            # Down-sampling algorithm: ensure samples are only taken during the specified sample times
            sleep_time = float((i * desired_sample_time) - time_arr[i - 1])
            sleep(max(sleep_time, 0))  # Program sleeps for non-negative amount of time
            curr_t_arr[i - 1] = perf_counter()
            time_arr[i - 1] = round(curr_t_arr[i - 1] - start_time, 2)  # Overwrite current MPC test time in array

            print(
                f"==== Sample: {i}/{sample_num} || Time: {time_arr[i-1]}/{desired_sample_time * sample_num} s ==== ")

            meas_height[[0], [i]] = curr_height[0]
            meas_height[[1], [i]] = curr_height[1]

            # Kalman filter: estimate height given measured height and controller action
            kf2.predict(u=fan_spd_factor[:, [i - 1]]), kf2.update(meas_height[:, [i]])

            est_height[:, [i]] = kf2.x[:, [0]]  # Kalman filter estimation

            height_sp[:, [i]] = sps[[sp_index]].T

            # Calculate optimal controller output (fan speed factors)
            uk, cost[i] = mpc(Q, R, est_height[:, [i]], pred_horizon, umin, umax, xk_sp=height_sp[:, [i]],
                              A=Aoverall, B=Boverall, P=P)

            fan_spd_factor[:, [i]] = np.round(uk, 0)

            cont.set_fan_speeds(fan_spd_factor[[0], [i]], 3)  # Set fan speed for tube 4
            cont.set_fan_speeds(fan_spd_factor[[1], [i]], 2)  # Set fan speed for tube 3

            # Measure the fan speeds immediately after applying the fan speed factor
            fan_speeds = cont.get_fan_speeds()
            fan_speed4[i] = fan_speeds[0]  # Record fan speed for tube 4
            fan_speed3[i] = fan_speeds[1]  # Record fan speed for tube 3

            print(f"Fan Speeds (4:1) = {fan_speeds} rpm")

            curr_levels = [cont.get_level(0), cont.get_level(1), cont.get_level(2), cont.get_level(3)]

            # Convert the ball-levels to ball-heights (distance b/w floor and ball in units of cm) in each tube
            for k in range(4):
                if level_gain[k] is not None:
                    curr_height[k] = round(((bot_levels[k] - curr_levels[k]) * level_gain[k]), 1)
                else:
                    curr_height[k] = None

            # Calculate the sample time using the difference in run times between two consecutive iterations
            if i > 1:
                sample_t_arr[i - 2] = round(curr_t_arr[i - 1] - curr_t_arr[i - 2], 2)

            print(f"Sensor Levels (4:1) = {curr_levels}")
            print(f"Meas. Heights (4:1) = {curr_height} cm")
            print(f"Est. Heights (4:1) = {[round(float(est_height[[0], [i-1]]), 1), round(float(est_height[[1], [i-1]]), 1), None, None]} cm")
            print(f"Set Heights (4:1) = {[round(float(height_sp[[0], [i-1]]), 1), round(float(height_sp[[1], [i-1]]), 1), None, None]} cm")
            print("=================================================")

            if sqrt_cost:
                actual_cost[i + 1] = np.log(
                    (meas_height[:, [i + 1]] - height_sp[:, [i]]).T @ Q @ (meas_height[:, [i + 1]]
                                                                           - height_sp[:, [i]])) + np.log(uk.T @ R @ uk)
            else:
                actual_cost[i + 1] = (meas_height[:, [i + 1]] - height_sp[:, [i]]).T @ Q @ (
                        meas_height[:, [i + 1]] - height_sp[:, [i]]) + uk.T @ R @ uk

            if meas_height[0, i + 1] > height_sp[0, [i]] * (1 + sp_offset):
                tube_cost[i + 1] += abs(meas_height[0, i + 1] - height_sp[0, [i]] * (1 + sp_offset))
            elif meas_height[0, i + 1] < height_sp[0, [i]] * (1 - sp_offset):
                tube_cost[i + 1] += abs(meas_height[0, i + 1] - height_sp[0, [i]] * (1 - sp_offset))
            if meas_height[1, i + 1] > height_sp[1, [i]] * (1 + sp_offset):
                tube_cost[i + 1] += abs(meas_height[1, i + 1] - height_sp[1, [i]] * (1 + sp_offset))
            elif meas_height[1, i + 1] < height_sp[1, [i]] * (1 - sp_offset):
                tube_cost[i + 1] += abs(meas_height[1, i + 1] - height_sp[1, [i]] * (1 - sp_offset))

            # Change set point height
            if i % sp_freq == 0:  # set point change frequency (units: samples)
                sp_index += 1
                if sp_index > len(sps) - 1:  # loop around set point array if its end is reached
                    sp_index = 0

            # Update A and B matrices every "id_horizon" iterations
            if Adaptive_MPC and i % id_horizon == 0:
                # # # ==== PySwarms Adaptive MPC ==== #
                d1 = est_height[:, (i - id_horizon):i]
                d2 = fan_spd_factor[:, (i - id_horizon):i]
                d3 = est_height[:, (i - (id_horizon - 1)):i + 1]
                # d3 = height_sp[:, (i - (id_horizon - 1)):i + 1]

                optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=np.size(Aoverall) + np.size(Boverall),
                                                    options=options)
                AB_cost[int((i - id_horizon) / id_horizon)], pos = optimizer.optimize(opt_func, iters=100, data1=d1,
                                                                                      data2=d2, data3=d3)
                Ap = pos[:np.size(Aoverall)].reshape(2, 2)
                Bp = pos[np.size(Aoverall):].reshape(2, 2)

                # Update A and B matrices (weighted update)
                Aoverall = Aoverall + 0.05 * (Ap - Aoverall)
                Boverall = Boverall + 0.05 * (Bp - Boverall)

                kf2.F, kf2.B = Aoverall, Boverall  # Update Kalman filter parameters

                # ==== RLS algorithm ==== #
                # # Concatenate x_k and u_k
                # myinput = np.concatenate((est_height[[0], [i - 1]], est_height[[1], [i - 1]],
                #                           fan_spd_factor[[0], [i - 1]], fan_spd_factor[[1], [i - 1]]), axis=None)
                # _ = filt4.predict(myinput)  # we don't do anything with the filter output
                # _ = filt3.predict(myinput)  # we don't do anything with the filter output
                #
                # filt4.adapt(est_height[[0], [i]], myinput)  # adapt(observation, regressors)
                # filt3.adapt(est_height[[1], [i]], myinput)
                # Ap = np.concatenate((filt4.w[:2], filt3.w[:2])).reshape(2, 2)  #
                # Bp = np.concatenate((filt4.w[2:], filt3.w[2:])).reshape(2, 2)
                #
                # Aoverall = Aoverall + 1 * (Ap - Aoverall)
                # Boverall = Boverall + 1 * (Bp - Boverall)
                #
                # kf2.F, kf2.B = Aoverall, Boverall  # Update Kalman filter parameters

        # ======== End of MPC algorithm for-loop ========

        sample_time_avg = np.mean(sample_t_arr[:sample_num - 2])  # Calculate average sample time for PID control test
        sample_time_std = np.std(sample_t_arr[:sample_num - 2])  # Calculate standard deviation in sample times
        print(f"Average Sample Time: {round(sample_time_avg, 2)}±{round(sample_time_std, 2)} s")

        # Calculate the MSEs b/w set point and measured height
        mse_meas_sp4 = np.mean(np.square(meas_height[[0], :sample_num - 2] - height_sp[[0], :sample_num - 2]))
        mse_meas_sp3 = np.mean(np.square(meas_height[[1], :sample_num - 2] - height_sp[[1], :sample_num - 2]))

        # Calculate the MSEs b/w set point and estimated height
        mse_est_sp4 = np.mean(np.square(est_height[[0], :sample_num - 2] - height_sp[[0], :sample_num - 2]))
        mse_est_sp3 = np.mean(np.square(est_height[[1], :sample_num - 2] - height_sp[[1], :sample_num - 2]))

        # Calculate the MSEs b/w measured height and estimated height
        mse_est4 = np.mean(np.square(est_height[[0], :sample_num - 2] - meas_height[[0], :sample_num - 2]))
        mse_est3 = np.mean(np.square(est_height[[1], :sample_num - 2] - meas_height[[1], :sample_num - 2]))

        # Save and plot MPC test data
        if Linear_MPC:
            plot_mpc_test(sample_num, time_arr, meas_height, est_height, height_sp, fan_spd_factor, actual_cost,
                          tube_cost,
                          sqrt_cost, mse_meas_sp4, mse_est_sp4, mse_est4, mse_meas_sp3, mse_est_sp3, mse_est3, Aoverall,
                          Boverall, Q, R, pred_horizon, var_n, var_e, test_type='mpc', P=P)

        elif Adaptive_MPC:
            plot_mpc_test(sample_num, time_arr, meas_height, est_height, height_sp, fan_spd_factor, cost, tube_cost,
                          sqrt_cost, mse_meas_sp4, mse_est_sp4, mse_est4, mse_meas_sp3, mse_est_sp3, mse_est3, Aoverall,
                          Boverall, Q, R, pred_horizon, var_n, var_e, test_type='adpt-mpc', A0=A0, B0=B0,
                          id_horiz=id_horizon, P=P)

        # Lower the fan speeds in all tubes after completing PID test
        cont.lower_fan_speeds()

    # == 4. Nonlinear SISO MPC test == #
    elif mode == '4':
        print('\nConducting Nonlinear MPC test')
        sample_num = 240
        desired_sample_time = 0.5  # units: sec
        pred_horizon = 2
        print(f"Sample Number: {sample_num}")
        print(f"MPC Prediction Horizon: {pred_horizon}\n")

        # Kalman filter covariance matrices (can tune as needed)
        var_n = 9.5  # 'Kalman Q' - lower => trust model
        var_e = 0.2  # 'Kalman R' - lower => trust measurement

        # Coefficients in MPC cost function J
        Q = 2.26       # Set point error cost weight
        R = 0.01       # Controller action cost weight
        P = 'none'  # Terminal cost (set to string type to ignore)

        # Lower and upper limits of controller (used in optimizer)
        umax = 100.0
        umin = 0.0

        # Create unscented Kalman filter
        points = SimplexSigmaPoints(n=1, alpha=0.3)
        ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, fx=nl_system_model, hx=h_cv, dt=1., points=points)

        sp_offset = 0.1  # 10 percent of the sp (used to increase tube cost as output exceeds 10% of set point)
        sqrt_cost = True

        # Initialize several zero-arrays below
        true_height, meas_height, est_height, height_sp, time_arr, cost, \
            actual_cost, tube_cost, curr_t_arr = (np.array([0.0] * (sample_num + 1)) for y in range(9))
        sample_t_arr = np.array([0.0] * sample_num)
        fan_spd_factor = np.array([0] * (sample_num + 1))

        # Choose which tube to control (3 or 4)
        tube = 3

        # Define height set points (units: cm)
        sps = np.array([70, 40, 60, 30])
        sp_freq = 60        # Set point change freq (units: samples)
        sp_index = 0
        height_sp[0] = sps[sp_index]

        # Initialize UKF parameters
        ukf.x, ukf.Q, ukf.R = meas_height[0], var_n, var_e
        est_height[0] = meas_height[0]

        # *Alternative to KF: Moving median filter
        # moving_window = [meas_height[0]]

        # Calculate initial mpc controller output
        uk, _ = nl_mpc(Q, R, est_height[0], pred_horizon, umin, umax, xk_sp=height_sp[0])
        fan_spd_factor[0] = np.round(uk, 0)

        start_time = perf_counter()  # Start a counter before conducting MPC test to measure time elapsed

        curr_height = [0, 0, 0, 0]  # [Tube 4, Tube 3, Tube 2, Tube 1]

        # Run nonlinear MPC algorithm for "sample number" of iterations
        for i in range(1, sample_num + 1):
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Record current MPC test time in array

            # Down-sampling algorithm: ensure samples are only taken during the specified sample times
            sleep_time = float((i * desired_sample_time) - time_arr[i])
            sleep(max(sleep_time, 0))  # Program sleeps for non-negative amount of time
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Overwrite current MPC test time in array

            # Calculate the sample time using the difference in run times between two consecutive iterations
            if i > 0:
                sample_t_arr[i - 1] = round(curr_t_arr[i] - curr_t_arr[i - 1], 2)

            print(f"===== Sample: {i}/{sample_num} =====")

            # Obtain height measurements
            curr_levels = [cont.get_level(0), cont.get_level(1), cont.get_level(2), cont.get_level(3)]
            # Convert the ball-levels to ball-heights (distance b/w floor and ball in units of cm) in each tube
            for k in range(4):
                if level_gain[k] is not None:
                    curr_height[k] = round(((bot_levels[k] - curr_levels[k]) * level_gain[k]), 1)
                else:
                    curr_height[k] = None
            if tube == 4:
                meas_height[i] = curr_height[0]    # tube 4 height
            else:
                meas_height[i] = curr_height[1]     # tube 3 height

            # Unscented Kalman filter estimates height
            ukf.predict(), ukf.update(meas_height[i])
            est_height[i] = ukf.x

            # **ALTERNATIVE to KF: Moving median filter/estimation
            # if i < 10:
            #     moving_window.append(meas_height[i])
            #     est_height[i] = meas_height[i]
            # else:
            #     moving_window.append(meas_height[i])
            #     est_height[i] = np.median(moving_window)
            #     del moving_window[0]

            # Set point assignment
            height_sp[i] = sps[sp_index]

            # Calculate optimal controller output (fan speed factors)
            uk, cost[i] = nl_mpc(Q, R, est_height[i], pred_horizon, umin, umax, xk_sp=height_sp[i], P=P)
            fan_spd_factor[i] = np.round(uk, 0)
            if tube == 4:
                cont.set_fan_speeds(fan_spd_factor[i], 3)  # tube 4 fan speed
            else:
                cont.set_fan_speeds(fan_spd_factor[i], 2)  # tube 3 fan speed

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
            if i > 0 and i % sp_freq == 0:  # set point change frequency (units: samples)
                sp_index += 1
                if sp_index > len(sps) - 1:  # loop around set point array if its end is reached
                    sp_index = 0
        # ==== End of MPC for-loop ==== #

        sample_time_avg = np.mean(sample_t_arr[1:])  # Calculate average sample time for PID control test
        sample_time_std = np.std(sample_t_arr[1:])  # Calculate standard deviation in sample times
        print(f"Avg. Sample Time: {round(sample_time_avg, 2)}±{round(sample_time_std, 2)} s")

        # Calculate the MSEs b/w set point and estimated height
        mse_est_sp = np.mean(np.square(est_height - height_sp))
        print(f'MSE = {round(mse_est_sp, 2)}')

        cont.lower_fan_speeds()

        plot_nonlin_mpc(time_arr, meas_height, est_height, height_sp, fan_spd_factor, actual_cost, tube_cost, sqrt_cost,
                        mse_est_sp, var_n, var_e, Q, R, pred_horizon, tube)

    # == 5. Step test == #
    elif mode == '5':
        print('\nConducting step test')

        sample_num = 90
        desired_sample_time = 0.5  # Units: sec
        print(f"Sample Number: {sample_num}")
        print(f"Estimated Time: {desired_sample_time * sample_num} s\n")

        # Initialize several zero-arrays below (NOTE: tubes 1 & 2 excluded since level sensors broken)
        height4, height3, time_arr, curr_t_arr = (np.array([0.0] * sample_num) for y in range(4))
        fan_spd_factor4, fan_speed4, fan_spd_factor3, fan_speed3 = (np.array([0] * sample_num) for y in range(4))
        sample_t_arr = np.array([0.0] * (sample_num - 1))

        # Initialize an empty array for the current ball height
        curr_height = [0, 0, 0, 0]  # [Tube 4, Tube 3, Tube 2, Tube 1]

        # Define fan-speed step (to move balls from bottom of tubes to top)
        for r in range(sample_num):
            if 0 <= r < 1:
                fan_spd_factor4[r] = 100
                fan_spd_factor3[r] = 100
            if 1 <= r < 10:
                fan_spd_factor4[r] = 39
                fan_spd_factor3[r] = 39
            elif 10 <= r < 30:
                fan_spd_factor4[r] = 45
                fan_spd_factor3[r] = 45
            elif 30 <= r < 50:
                fan_spd_factor4[r] = 30
                fan_spd_factor3[r] = 30
            elif 50 <= r < 70:
                fan_spd_factor4[r] = 41
                fan_spd_factor3[r] = 41
            elif 70 <= r < 80:
                fan_spd_factor4[r] = 37
                fan_spd_factor3[r] = 37
            elif 80 <= r < sample_num:
                fan_spd_factor4[r] = 47
                fan_spd_factor3[r] = 47

        start_time = perf_counter()  # Start a counter before conducting step-test to measure time elapsed

        # Run the step-test and collect fan-speed and ball-height data for "sample number" of iterations
        for i in range(sample_num):
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Record current step test time in array

            # Down-sampling algorithm: ensure samples are only taken during the specified sample times
            sleep_time = float((i + 1) * desired_sample_time) - time_arr[i]
            sleep(max(sleep_time, 0))  # Program sleeps for non-negative amount of time
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Overwrite current step test time in array

            print(
                f"==== Sample: {i + 1}/{sample_num} || Time: {time_arr[i]}/{desired_sample_time * sample_num} s ==== ")

            # Set the fan speeds using the fan speed factors
            cont.set_fan_speeds(fan_spd_factor4[i], 3)
            cont.set_fan_speeds(fan_spd_factor3[i], 2)

            # Measure the fan speeds immediately after applying the fan speed factor
            fan_speeds = cont.get_fan_speeds()
            fan_speed4[i] = fan_speeds[0]
            fan_speed3[i] = fan_speeds[1]

            # Measure the current ball levels (distance b/w sensor and ball in arbitrary units)
            curr_levels = [cont.get_level(0), cont.get_level(1), cont.get_level(2), cont.get_level(3)]

            # Convert the ball-levels to ball-heights (distance b/w floor and ball in units of cm) in each tube
            for k in range(4):
                if level_gain[k] is not None:
                    curr_height[k] = round(((bot_levels[k] - curr_levels[k]) * level_gain[k]), 1)
                else:
                    curr_height[k] = None

            # Calculate the sample time using the difference in run times between two consecutive iterations
            if i > 0:
                sample_t_arr[i - 1] = round(curr_t_arr[i] - curr_t_arr[i - 1], 2)

            # Record the current height for tubes 3 & 4 in an array (tubes 1 & 2 are not currently working)
            height4[i] = curr_height[0]
            height3[i] = curr_height[1]
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Record current step-test time in array

            print(f"Fan Speed Factors (4:1) = [{fan_spd_factor4[i]}, {fan_spd_factor3[i]}, 0, 0]")
            print(f"Fan Speeds (4:1) = {fan_speeds} rpm")
            print(f"Levels (4:1) = {curr_levels}")
            print(f"Heights (4:1) = {curr_height} cm")
            print("===============================================")
        # ======== End of step-test for loop ======== #

        sample_time_avg = np.mean(sample_t_arr)  # Calculate average sample time in step test
        sample_time_std = np.std(sample_t_arr)  # Calculate standard deviation in sample times
        print(f"Average Sample Time: {round(sample_time_avg, 2)}±{round(sample_time_std, 2)} s")

        # Lower the fan speeds in all tubes after completing step test
        cont.lower_fan_speeds()

        # Call a function from "helper.py" to plot the fan-speed and ball height vs. time
        plot_io_test(time_arr, height4, fan_spd_factor4, fan_speed4, height3,
                     fan_spd_factor3, fan_speed3, calibr_status, 'Step')

    # == 6. PRBS test and 7. RGS test == #
    elif mode == '6' or mode == '7':
        if mode == '6':
            id_test = 'PRBS'
        else:
            id_test = 'RGS'
        print(f'\nConducting {id_test} Test')

        # Ensure the PRBS csv files are in the project folder
        # Read the csv files and save as dataframes
        df_time = pd.read_csv(f'BiT_{id_test}_input_43.csv', header=0, dtype=int, usecols=[0])
        df4 = pd.read_csv(f'BiT_{id_test}_input_43.csv', header=0, dtype=int, usecols=[1])
        df3 = pd.read_csv(f'BiT_{id_test}_input_43.csv', header=0, dtype=int, usecols=[2])

        # Calculate sample number and sample time
        desired_time_arr = np.squeeze(df_time.to_numpy())
        sample_num = desired_time_arr.size
        desired_sample_time = float(desired_time_arr[1] - desired_time_arr[0])
        print(f"Sample Number: {sample_num} \nEstimated Time: {desired_sample_time * sample_num} s")

        # Convert fan speed factor dataframes to arrays
        fan_spd_factor4 = np.squeeze(df4.to_numpy())
        fan_spd_factor3 = np.squeeze(df3.to_numpy())

        # Initialize several zero-arrays below (NOTE: tubes 1 & 2 excluded since level sensors broken)
        height4, height3, time_arr, curr_t_arr = (np.array([0.0] * sample_num) for y in range(4))
        fan_speed4, fan_speed3 = (np.array([0] * sample_num) for y in range(2))
        sample_t_arr = np.array([0.0] * (sample_num - 1))

        # Initialize an empty array for the current ball height
        curr_height = [0, 0, 0, 0]  # [Tube 4, Tube 3, Tube 2, Tube 1]

        cont.set_fan_speeds(60, 3)
        cont.set_fan_speeds(fan_spd_factor3[0], 2)
        print("Letting balls settle to start point...")
        start_time = perf_counter()  # Start a counter before conducting PRBS test to measure time elapsed

        # Run the PRBS test and collect fan-speed and ball-height data for "sample number" of iterations
        for i in range(sample_num):
            # Record the current program runtime
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Record current PRBS test time in array

            # Down-sampling algorithm: ensure samples are only taken during the specified sample times
            sleep_time = float((i + 1) * desired_sample_time) - time_arr[i]
            sleep(max(sleep_time, 0))  # Program sleeps for non-negative amount of time
            curr_t_arr[i] = perf_counter()
            time_arr[i] = round(curr_t_arr[i] - start_time, 2)  # Overwrite current PRBS test time in array

            print(
                f"==== Sample: {i + 1}/{sample_num} || Time: {time_arr[i]}/{desired_sample_time * sample_num} s ==== ")

            # Set the fan speeds using the fan speed factors
            cont.set_fan_speeds(fan_spd_factor4[i], 3)
            cont.set_fan_speeds(fan_spd_factor3[i], 2)

            # Measure the fan speeds immediately after applying the fan speed factor
            fan_speeds = cont.get_fan_speeds()
            fan_speed4[i] = fan_speeds[0]
            fan_speed3[i] = fan_speeds[1]

            # Measure the current ball levels (distance b/w sensor and ball in arbitrary units)
            curr_levels = [cont.get_level(0), cont.get_level(1), cont.get_level(2), cont.get_level(3)]

            # Convert the ball-levels to ball-heights (distance b/w floor and ball in units of cm) in each tube
            for k in range(4):
                if level_gain[k] is not None:
                    curr_height[k] = round(((bot_levels[k] - curr_levels[k]) * level_gain[k]), 1)
                else:
                    curr_height[k] = None

            # Calculate the sample time using the difference in run times between two consecutive iterations
            if i > 0:
                sample_t_arr[i - 1] = round(curr_t_arr[i] - curr_t_arr[i - 1], 2)

            # Record the current height for tubes 3 & 4 in an array (tubes 1 & 2 are not currently working)
            height4[i] = curr_height[0]
            height3[i] = curr_height[1]

            print(f"Fan Speed Factors (4:1) = [{fan_spd_factor4[i]}, {fan_spd_factor3[i]}, 0, 0]")
            print(f"Fan Speeds (4:1) = {fan_speeds} rpm")
            print(f"Levels (4:1) = {curr_levels}")
            print(f"Heights (4:1) = {curr_height} cm")
            print("===============================================")
        # ======== End of PRBS test for loop ======== #

        sample_time_avg = np.mean(sample_t_arr)  # Calculate average sample time in PRBS test
        sample_time_std = np.std(sample_t_arr)  # Calculate standard deviation in sample times
        print(f"Average Sample Time: {round(sample_time_avg, 2)}±{round(sample_time_std, 2)} s")

        # Lower the fan speeds in all tubes after completing PRBS test
        cont.lower_fan_speeds()

        # Call a function from "helper.py" to plot the fan-speed and ball height vs. time
        plot_io_test(time_arr, height4, fan_spd_factor4, fan_speed4, height3,
                     fan_spd_factor3, fan_speed3, calibr_status, id_test)

    # == 8. READ LEVELS == #
    elif mode == '8':
        fan_spd = input('\nSelect fan speeds:\n[1] Minimum\n[2] Maximum\n')

        if fan_spd == '1':
            # cont.set_fan_speeds(0, 3)  # Set fan speed for tube 4
            # cont.set_fan_speeds(0, 2)  # Set fan speed for tube 3
            # cont.set_fan_speeds(0, 1)  # Set fan speed for tube 2
            # cont.set_fan_speeds(0, 0)  # Set fan speed for tube 1
            print('Minimum Fan Speeds')
        else:
            cont.set_fan_speeds(100, 3)  # Set fan speed for tube 4
            cont.set_fan_speeds(100, 2)  # Set fan speed for tube 3
            cont.set_fan_speeds(100, 1)  # Set fan speed for tube 2
            cont.set_fan_speeds(100, 0)  # Set fan speed for tube 1
            print('Maximum Fan Speeds')

        i = 1
        while True:
            print(f'\n======= Sample {i} =======')
            curr_levels = [cont.get_level(0), cont.get_level(1), cont.get_level(2), cont.get_level(3)]
            print(f"Levels (4:1) = {curr_levels}")
            print(f'Fan Speeds (4:1) = {cont.get_fan_speeds()} rpm')
            i += 1
            sleep(1)

    # == 9. Quit == #
    else:
        cont.lower_fan_speeds()

    # Disconnect from rabbitMQ and end process after completing any mode
    cont.disconnect_expt()

# If Ctrl+C is pressed, lower the fan speeds, and end the program
except KeyboardInterrupt:
    print("\nEnding process.")
    cont.lower_fan_speeds()
    cont.disconnect_expt()
    raise SystemExit
