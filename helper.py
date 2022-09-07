"""
=================================================================
Title:           helper.py
Author:          Antony Pulikapparambil
Date:            May 2022
Acknowledgement: Armianto Sumitro - BallsInTubesController class
=================================================================
"""
# Contains RabbitMQ client, sensor-to-height calibration, and various plotting methods

# NOTE: this file can not be run - it supplements main.py

import pika
import threading
import matplotlib
import os
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from datetime import datetime

matplotlib.rcParams.update(
    {'font.family': 'Times New Roman',
     'font.serif': 'Times New Roman',
     'mathtext.rm': 'Times New Roman',
     'mathtext.fontset': 'stix'})


class BallsInTubesController:
    # Initiates a new Python controller for Balls in Tubes experiment
    # host is an IP address of the RabbitMQ server, should match with host
    # specified in MATLAB in original computer connected to experiment. 

    def __init__(self, host, exchange_name='ballsInTubes'):
        self.host = host
        self.exchange_name = exchange_name

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host))   # opt. heartbeat param
        channel = self.connection.channel()
        # exchanges are fanout for convenience
        # also since MATLAB RabbitMQ Wrapper uses fanout exchanges
        # fanout just means it publishes to all queues that bind to the exchange
        channel.exchange_declare(exchange=exchange_name,
                                 exchange_type='fanout')
        channel.exchange_declare(exchange=exchange_name + 'Send',
                                 exchange_type='fanout')
                                 # i.e. ballsInTubesSend is the exchange name for incoming signals sent
                                 # from MATLAB in the actual controller
        get_queue = channel.queue_declare(queue='', exclusive=True)
        set_queue = channel.queue_declare(queue='', exclusive=True)
        channel.queue_bind(exchange=exchange_name + 'Send',
                           queue=get_queue.method.queue, )
        channel.queue_bind(exchange=exchange_name,
                           queue=set_queue.method.queue)
        channel.basic_consume(
            queue=get_queue.method.queue, on_message_callback=self.callback, auto_ack=True
        )
        thread = threading.Thread(target=channel.start_consuming)
        thread.start()
        self.channel = channel
        self.fan_speeds = np.array([0, 0, 0, 0])
        self.levels = np.array([0, 0, 0, 0])
        self.event = threading.Event()

    # Message parser
    def callback(self, ch, method, properties, body):
        body = body.decode('utf-8')
        tokens = body.split('\t')
        if tokens[0] == 'fanspeeds':
            fan_speeds = tokens[1].split('  ')
            for val, fan_speed in enumerate(fan_speeds):
                self.fan_speeds[val] = int(fan_speed)
            self.event.set()
            return
        elif tokens[0] == 'level':
            tube = int(tokens[1])
            level = float(tokens[2])
            self.levels[tube] = level
            self.event.set()
            return
        else:
            print(body)
        self.connection.add_callback_threadsafe(self.callback)  # attempt to fix connection error - still occurs though

    def get_fan_speeds(self):
        self.channel.basic_publish(exchange=self.exchange_name,
                                   routing_key='',
                                   body='getFanSpeeds')
        self.wait_event()
        return self.fan_speeds

    # Note: speed is definitely not in percentage
    def set_fan_speeds(self, speed, tube):
        self.channel.basic_publish(exchange=self.exchange_name,
                                   routing_key='',
                                   body='setFanSpeed\t' + str(speed) + '\t' + str(tube))

    # Note: tube indices are 0-indexed
    def get_level(self, tube):
        self.channel.basic_publish(exchange=self.exchange_name,
                                   routing_key='',
                                   body='getLevel\t' + str(tube))
        self.wait_event()
        return self.levels[tube]

    # Declare timeout if no value is returned after 5 seconds
    def wait_event(self):
        if not self.event.wait(5):
            print('[ERROR] Did not receive any message from experiment, make sure MATLAB is running rabbitMQ wrapper in'
                  ' the background')
        self.event.clear()

    def disconnect_expt(self):
        pika.BlockingConnection.close(self.connection)

    # Method for calibrating level readings:
    # NOTE: the sensors return a 'level' which is measured from top of the tube to ball, in arbitrary units...
    # ...however, we wish to convert this 'level' to a 'height' measurement from the bottom of the tube
    # to the ball in units of cm... this requires certain conversion parameters which this method calculates
    # NOTE: Tube 1 sensor is broken -- constant 70 000 reading regardless of ball height
    # NOTE: Tube 2 sensor is broken -- constant 0 reading
    def calibrate_level(self, calibr_sample_no=10, calibr_sample_period=0.5):

        print("Calibrating level readings...")

        line_clear = '\r'
        # Move balls to bottom of tube
        print("Moving balls to bottom of tubes...", end=line_clear)
        self.set_fan_speeds(27, 0)  # Tube 1 of expt
        sleep(0.5)
        self.set_fan_speeds(26, 1)  # Tube 2 of expt
        sleep(0.3)
        self.set_fan_speeds(28, 2)  # Tube 3 of expt
        sleep(0.4)
        self.set_fan_speeds(30, 3)  # Tube 4 of expt
        sleep(4)                    # Give balls time to travel to bottom

        print(f"Calibrating. Estimated wait time: {calibr_sample_no * calibr_sample_period} s", end=line_clear)
        tot_bot_levels4, tot_bot_levels3, tot_bot_levels2, tot_bot_levels1, time_arr = [], [], [], [], []

        # Collect "calibration sample number' of sensor readings
        for i in range(calibr_sample_no):
            tot_bot_levels4.append(self.get_level(0))
            tot_bot_levels3.append(self.get_level(1))
            tot_bot_levels2.append(self.get_level(2))
            tot_bot_levels1.append(self.get_level(3))
            time_arr.append(i * calibr_sample_period)
            sleep(calibr_sample_period)

        # Convert the lists to numpy arrays
        tot_bot_levels4, tot_bot_levels3, tot_bot_levels2, tot_bot_levels1 = list_to_array(
            tot_bot_levels4, tot_bot_levels3, tot_bot_levels2, tot_bot_levels1
        )
        # Take average + std of bottom-of-tube level readings
        avg_bot_levels = array_avg(tot_bot_levels4, tot_bot_levels3, tot_bot_levels2, tot_bot_levels1)
        std_bot_levels = array_std(tot_bot_levels4, tot_bot_levels3, tot_bot_levels2, tot_bot_levels1)
        print(f"Bottom Levels(4:1) = [{avg_bot_levels[0]}±{std_bot_levels[0]} {avg_bot_levels[1]}±{std_bot_levels[1]}"
              f" {avg_bot_levels[2]}±{std_bot_levels[2]} {avg_bot_levels[3]}±{std_bot_levels[3]}]")

        # Move balls to top of tube
        print("Moving balls to top of tubes...", end=line_clear)
        self.set_fan_speeds(55, 0)  # Tube 1 of expt
        sleep(0.5)
        self.set_fan_speeds(54, 1)  # Tube 2 of expt
        sleep(0.5)
        self.set_fan_speeds(60, 2)  # Tube 3 of expt
        sleep(0.5)
        self.set_fan_speeds(60, 3)  # Tube 4 of expt
        sleep(6)                    # Give balls time to travel to top

        print(f"Calibrating. Estimated wait time: {calibr_sample_no * calibr_sample_period} s", end=line_clear)
        tot_top_levels4, tot_top_levels3, tot_top_levels2, tot_top_levels1, time_arr = [], [], [], [], []

        # Collect "calibration sample number' of sensor readings
        for i in range(calibr_sample_no):
            tot_top_levels4.append(self.get_level(0))
            tot_top_levels3.append(self.get_level(1))
            tot_top_levels2.append(self.get_level(2))
            tot_top_levels1.append(self.get_level(3))
            time_arr.append(i * calibr_sample_period)        # Currently, has no purpose -- may be used plot to later
            sleep(calibr_sample_period)

        # Convert the lists to numpy arrays
        tot_top_levels4, tot_top_levels3, tot_top_levels2, tot_top_levels1 = list_to_array(
            tot_top_levels4, tot_top_levels3, tot_top_levels2, tot_top_levels1
        )
        # Take average + std of top-of-tube level readings
        avg_top_levels = array_avg(tot_top_levels4, tot_top_levels3, tot_top_levels2, tot_top_levels1)
        std_top_levels = array_std(tot_top_levels4, tot_top_levels3, tot_top_levels2, tot_top_levels1)
        print(f"Top Levels(4:1) = [{avg_top_levels[0]}±{std_top_levels[0]} {avg_top_levels[1]}±{std_top_levels[1]}"
              f" {avg_top_levels[2]}±{std_top_levels[2]} {avg_top_levels[3]}±{std_top_levels[3]}]")

        avg_diff_levels = avg_bot_levels - avg_top_levels

        # Level gain is a conversion factor for the level-to-height conversion
        level_gain = [divide(100, avg_diff_levels[0]), divide(100, avg_diff_levels[1]),     # 'divide' function checks
                      divide(100, avg_diff_levels[2]), divide(100, avg_diff_levels[3])]     # for division by zero

        # Lower fan speeds
        for i in range(4):
            self.set_fan_speeds(30, i)

        sleep(0.5)
        print(f"Level gains: {level_gain}")
        print('Calibration complete. \n')
        sleep(0.5)

        return avg_bot_levels, level_gain

    def lower_fan_speeds(self):
        """ Lowers the fan speeds of each tube """
        for i in range(4):
            self.set_fan_speeds(0, i)
        print("Successfully lowered fan speeds.")

# ==== End of 'BallsInTubesController" class ==== #


def plot_pid_control_test(time_arr, height_sp, height4, fanfactor4, fanspd4,
                          height3, fanfactor3, fanspd3, calibr_status, mse_t4, mse_t3,
                          pidController3, pidController4, PID_type):
    """ Plots ball-height and set point vs. time and fan-speed and fan-speed-factor vs. time for each tube """

    # Record current date/time (to include in data/plots)
    curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Check if data storage folder exists and create if needed
    storage_exists = os.path.exists('./pid-data')
    if not storage_exists:
        os.makedirs('./pid-data')
        sleep(1)

    # Save the tube 4 PID control test data as .csv files
    df_t4 = pd.DataFrame({"Time (s)": time_arr, "Set Point (cm)": height_sp[0, :], "Ball Height (cm)": height4,
                          "Fan Speed (rpm)": fanspd4, "Fan Factor": fanfactor4})
    df_t4.to_csv("pid-data/pid-data-tube4.csv", index=False)

    # Prepend the tube 4 PID values and MSE to the csv file
    with open("pid-data/pid-data-tube4.csv", 'r') as rawf4:
        data4 = rawf4.read()
    with open("pid-data/pid-data-tube4.csv", 'w') as newf4:
        newf4.write(f"Tube 4 PID ({curr_datetime}) \nPID type: {PID_type} \nKc = {pidController4.get_kc()}"
                    f"\ntau_i = {pidController4.get_tau_i()} \ntau_d = {pidController4.get_tau_d()} \nMSE = {mse_t4}\n" + data4)

    # Save the tube 3 PID control test data as .csv files
    df_t3 = pd.DataFrame({"Time (s)": time_arr, "Set Point (cm)": height_sp[1, :], "Ball Height (cm)": height3,
                          "Fan Speed (rpm)": fanspd3, "Fan Factor": fanfactor3})
    df_t3.to_csv("pid-data/pid-data-tube3.csv", index=False)

    # Prepend the tube 3 PID values and MSE to the csv file
    with open("pid-data/pid-data-tube3.csv", 'r') as rawf3:
        data3 = rawf3.read()
    with open("pid-data/pid-data-tube3.csv", 'w') as newf3:
        newf3.write(f"Tube 3 PID ({curr_datetime}) \nPID type: {PID_type} \nKc = {pidController3.get_kc()}"
                    f"\ntau_i = {pidController3.get_tau_i()} \ntau_d = {pidController3.get_tau_d()} \nMSE = {mse_t3}\n" + data3)

    # == Tube 4 Plots == #
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    twin4_factor = ax[1].twinx()
    h4, = ax[0].plot(time_arr, height4, '.-', color='tab:blue', label='Measured Height')
    sp4, = ax[0].plot(time_arr, height_sp[0, :], '-', color='tab:red', label='Set Point Height')
    fs4, = ax[1].plot(time_arr, fanspd4, '.-', color='tab:purple', label='Measured Fan Speed')
    ff4, = twin4_factor.plot(time_arr, fanfactor4, '.-', color='tab:green', label='Fan Speed Factor')

    ax[0].set_xlabel("Time (s)")                                # Label/color plots and axis labels
    ax[0].set_ylabel("Height (cm)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Fan Speed (rpm)")
    twin4_factor.set_ylabel("Fan Speed Factor")
    ax[0].set_ylim([-10, 110])
    ax[1].yaxis.label.set_color(fs4.get_color())                # Color axis labels
    twin4_factor.yaxis.label.set_color(ff4.get_color())
    ax[0].set_title(f'Tube 4 - Ball Height vs. Time ({calibr_status}, MSE: {mse_t4})')
    ax[1].set_title(f'Tube 4 - Fan Speed vs. Time (Kc = {pidController4.get_kc()}, '
                    f'\u03c4_i = {pidController4.get_tau_i()}, \u03c4_d = {pidController4.get_tau_d()})')

    # Display current date/time on figure title
    fig.suptitle(f'Tube 4 - {PID_type} PID Control ({curr_datetime})', fontsize=16)

    tkw = dict(size=2, width=1.5)                               # Color axis ticks
    ax[1].tick_params(axis='y', colors=fs4.get_color(), **tkw)
    twin4_factor.tick_params(axis='y', colors=ff4.get_color(), **tkw)

    ax[0].legend()                                              # Include legends
    ax[1].legend(handles=[fs4, ff4])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)                              # Translate plots downward to make super-title space
    plt.savefig("pid-data/pid-tube4-plot.png", bbox_inches='tight')     # Save the plot as a pdf file
    plt.show()

    # == Tube 3 Plots == #
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    twin3_factor = ax[1].twinx()
    h3, = ax[0].plot(time_arr, height3, '.-', color='tab:blue', label='Measured Ball Height')
    sp3, = ax[0].plot(time_arr, height_sp[1, :], ',-', color='tab:red', label='Set Point Height')
    fs3, = ax[1].plot(time_arr, fanspd3, '.-', color='tab:purple', label='Measured Fan Speed')
    ff3, = twin3_factor.plot(time_arr, fanfactor3, '.-', color='tab:green', label='Fan Speed Factor')

    ax[0].set_xlabel("Time (s)")                                    # Label/color plots and axis labels
    ax[0].set_ylabel("Height (cm)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Fan Speed (rpm)")
    twin3_factor.set_ylabel("Fan Speed Factor")
    ax[0].set_ylim([-10, 110])
    ax[1].yaxis.label.set_color(fs3.get_color())                    # Color axis labels
    twin3_factor.yaxis.label.set_color(ff3.get_color())
    ax[0].set_title(f'Tube 3 - Ball Height vs. Time ({calibr_status}, MSE: {mse_t3})')
    ax[1].set_title(f'Tube 3 - Fan Speed vs. Time (Kc = {pidController3.get_kc()}, '
                    f'\u03c4_i = {pidController3.get_tau_i()}, \u03c4_d = {pidController3.get_tau_d()})')

    # Display current date/time on figure title
    fig.suptitle(f'Tube 3 - {PID_type} PID Control ({curr_datetime})', fontsize=16)

    tkw = dict(size=2, width=1.5)                                   # Color axis ticks
    ax[1].tick_params(axis='y', colors=fs3.get_color(), **tkw)
    twin3_factor.tick_params(axis='y', colors=ff3.get_color(), **tkw)

    ax[0].legend()                                                  # Include legends
    ax[1].legend(handles=[fs3, ff3])
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)                                   # Translate plots downward to make super-title space
    plt.savefig("pid-data/pid-tube3-plot.png", bbox_inches='tight')          # Save the plot as a pdf file
    plt.show()


def plot_mpc_test(N, time_arr, meas_height, est_height, height_sp, fan_spd_factor, actual_cost, tube_cost, sqrt_cost,
                  mse_meas_sp4, mse_est_sp4, mse_est4, mse_meas_sp3, mse_est_sp3, mse_est3, A, B, Q, R, pred_horiz, KQ,
                  KR, test_type, A0=0, B0=0, id_horiz=0, P=0):

    # Record current date/time (to include in data/plots)
    curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    if test_type == 'mpc':
        title_test_type = 'Linear MPC'
    elif test_type == 'adpt-mpc':
        title_test_type = 'Adaptive MPC'

    # Check if data storage folder exists and create if needed
    storage_exists = os.path.exists(f'./{test_type}-data')
    if not storage_exists:
        os.makedirs(f'./{test_type}-data')
        sleep(1)

    # Save the tube 4 MPC test data as .csv files
    df_t4 = pd.DataFrame({"Time (s)": time_arr[:N], "Set Point (cm)": height_sp[0, :N], "Meas. Ball Height (cm)":
                         meas_height[0, :N], "Est. Ball Height (cm)": np.round(est_height[0, :N], 1), "Fan Factor":
                         np.round(fan_spd_factor[0, :N], 0)})
    df_t4.to_csv(f"{test_type}-data/{test_type}-data-tube4.csv", index=False)

    # Prepend the tube 4 MPC values and MSE to the csv file
    with open(f"{test_type}-data/{test_type}-data-tube4.csv", 'r') as rawf4:
        data4 = rawf4.read()
    with open(f"{test_type}-data/{test_type}-data-tube4.csv", 'w') as newf4:
        newf4.write(f"Tube 4 {title_test_type} ({curr_datetime})")
        if test_type == 'mpc':
            newf4.write(f"\nModel A = [{A[0]};{A[1]}] \nModel B = [{B[0]};{B[1]}]")
        else:
            newf4.write(f"\nModel A_i = [{A0[0]};{A0[1]}] \nModel B_i = [{B0[0]};{B0[1]}]")
            newf4.write(f"\nModel A_f = [{np.round(A[0], 3)};{np.round(A[1], 3)}] "
                        f"\nModel B_f = [{np.round(B[0], 3)};{np.round(B[1], 3)}] "
                        f"\nIdentification Horizon = {id_horiz}")
        newf4.write(f" \nMPC Q = [{Q[0]};{Q[1]}] \nMPC R = [{R[0]};{R[1]}]")
        if type(P) == int:
            newf4.write(f"\nMPC P = None (no terminal cost)")
        else:
            newf4.write(f"\nMPC P = [{P[0]};{P[1]}]")
        newf4.write(f"\nPrediction Horizon = {pred_horiz}"
                    f"\nKalman Q = [{KQ[0]};{KQ[1]}] \nKalman R = [{KR[0]};{KR[1]}]"
                    f"\nMSE b/w Set Point & Meas. Height = {round(mse_meas_sp4, 2)}"
                    f"\nMSE b/w Set Point & Est. Height = {round(mse_est_sp4, 2)}"
                    f"\nMSE b/w Est. Height & Meas. Height = {round(mse_est4, 2)} \n" + data4)

    # Save the tube 3 MPC test data as .csv files
    df_t3 = pd.DataFrame({"Time (s)": time_arr[:N], "Set Point (cm)": height_sp[1, :N], "Meas. Ball Height (cm)":
                         meas_height[1, :N], "Est. Ball Height (cm)": np.round(est_height[1, :N], 1), "Fan Factor":
                         np.round(fan_spd_factor[1, :N], 0)})
    df_t3.to_csv(f"{test_type}-data/{test_type}-data-tube3.csv", index=False)
    # Prepend the tube 3 MPC values and MSE to the csv file
    with open(f"{test_type}-data/{test_type}-data-tube3.csv", 'r') as rawf3:
        data3 = rawf3.read()
    with open(f"{test_type}-data/{test_type}-data-tube3.csv", 'w') as newf3:
        newf3.write(f"Tube 3 {title_test_type} ({curr_datetime})")
        if test_type == 'mpc':
            newf3.write(f"\nModel A = [{A[0]};{A[1]}] \nModel B = [{B[0]};{B[1]}]")
        else:
            newf3.write(f"\nModel A_i = [{A0[0]};{A0[1]}] \nModel B_i = [{B0[0]};{B0[1]}]")
            newf3.write(f"\nModel A_f = [{np.round(A[0], 3)};{np.round(A[1], 3)}] "
                        f"\nModel B_f = [{np.round(B[0], 3)};{np.round(B[1], 3)}] "
                        f"\nIdentification Horizon = {id_horiz}")
        newf3.write(f" \nMPC Q = [{Q[0]};{Q[1]}] \nMPC R = [{R[0]};{R[1]}]")
        if type(P) == int:
            newf3.write(f"\nMPC P = None (no terminal cost)")
        else:
            newf3.write(f"\nMPC P = [{P[0]};{P[1]}]")
        newf3.write(f"\nPrediction Horizon = {pred_horiz}"
                    f"\nKalman Q = [{KQ[0]};{KQ[1]}] \nKalman R = [{KR[0]};{KR[1]}]"
                    f"\nMSE b/w Set Point & Meas. Height = {round(mse_meas_sp3, 2)}"
                    f"\nMSE b/w Set Point & Est. Height = {round(mse_est_sp3, 2)}"
                    f"\nMSE b/w Est. Height & Meas. Height = {round(mse_est3, 2)} \n" + data3)

    # ==== Plotting the data ====
    fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex='all', constrained_layout=True)
    ax[0, 0].plot(time_arr[:N-2], meas_height[0, :N-2], '.', color='tab:blue', alpha=0.5, label='Measured Height')
    ax[0, 0].plot(time_arr[:N-2], height_sp[0, :N-2], '-', color='tab:red', alpha=0.5,
                  label='Set Point (MSE = {:.2f})'.format(mse_meas_sp4))
    ax[0, 0].plot(time_arr[:N-2], est_height[0, :N-2], '-', color='tab:green', alpha=0.5,
                  label='Est. Height (MSE = {:.2f})'.format(mse_est4))
    ax[0, 0].set_ylim([-10, 110])

    ax[0, 1].plot(time_arr[:N-2], meas_height[1, :N-2], '.', color='tab:blue', alpha=0.5, label='Measured Height')
    ax[0, 1].plot(time_arr[:N-2], height_sp[1, :N-2], '-', color='tab:red', alpha=0.5,
                  label='Set Point (MSE = {:.2f})'.format(mse_meas_sp3))
    ax[0, 1].plot(time_arr[:N-2], est_height[1, :N-2], '-', color='tab:green', alpha=0.5,
                  label='Est. Height (MSE = {:.2f})'.format(mse_est3))
    ax[0, 1].set_ylim([-10, 110])

    ax[0, 0].legend(loc='best')
    ax[0, 1].legend(loc='best')

    ax[1, 0].plot(time_arr[:N-2], actual_cost[:N-2], '.', label='$J$', color='tab:red', alpha=0.5)
    ax1 = ax[1, 0].twinx()
    ax1.plot(time_arr[:N-2], tube_cost[:N-2], '.', label='r', color='tab:blue', alpha=0.5)
    ax[1, 0].plot(np.nan, '.', label='$r$', color='tab:blue', alpha=0.5)
    ax[1, 0].legend(loc=1)

    ax[1, 1].plot(time_arr[:N-2], fan_spd_factor[0, :N-2], '.-', label='Tube 4', color='tab:red', alpha=0.5)
    ax[1, 1].plot(time_arr[:N-2], fan_spd_factor[1, :N-2], '.-', label='Tube 3', color='tab:orange', alpha=0.5)
    ax[1, 1].set_ylim([-10, 110])
    ax[1, 1].legend(loc=1)

    ax[0, 0].set_title('Tube 4 Heights (cm)')
    ax[0, 1].set_title('Tube 3 Heights (cm)')
    if sqrt_cost:
        ax[1, 0].set_ylabel(r'Actual $\sqrt{COST}$')
    else:
        ax[1, 0].set_ylabel(r'Actual Cost')
    ax1.set_ylabel('Tube Cost')
    ax[1, 1].set_title('Fan Speed Factors')
    ax[1, 0].set_title('Costs')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 1].set_xlabel('Time (s)')

    fig.suptitle(f'Tubes 3 & 4 - {title_test_type} Test ({curr_datetime})', fontsize=16)
    plt.savefig(f"{test_type}-data/{test_type}-test-plot.png", bbox_inches='tight')  # Save the plot as a pdf file
    plt.show()


def plot_nonlin_mpc(time_arr, meas_height, est_height, height_sp, fan_spd_factor, actual_cost, tube_cost, sqrt_cost,
                    mse_est_sp, var_n, var_e, Q, R, pred_horizon, tube):
    # Record current date/time (to include in plots)
    curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Check if data storage folder exists and create if needed
    storage_exists = os.path.exists('./nonlin-mpc-data')
    if not storage_exists:
        os.makedirs('./nonlin-mpc-data')
        sleep(1)

    # Save the tube 4 PID control test data as .csv files
    df = pd.DataFrame({"Time (s)": time_arr, "Set Height (cm)": height_sp, "Meas. Height (cm)": meas_height,
                        "Est. Height (cm)": est_height, "Fan Factor": fan_spd_factor})
    df.to_csv(f"nonlin-mpc-data/nonlin-mpc-data-t{tube}.csv", index=False)
    df.to_csv(f"nonlin-mpc-data/nonlin-mpc-data-t{tube}.csv", index=False)
    # Prepend the tube 4 PID values and MSE to the csv file
    with open(f"nonlin-mpc-data/nonlin-mpc-data-t{tube}.csv", 'r') as rawf:
        data = rawf.read()
    with open(f"nonlin-mpc-data/nonlin-mpc-data-t{tube}.csv", 'w') as newf:
        newf.write(f"Nonlinear SISO MPC Test - Tube {tube} ({curr_datetime}) \nMPC_Q = {Q} \nMPC_R = {R}"
                   f"\nMPC_N = {pred_horizon} \nUKF_Q = {var_n} \nUKF_R = {var_e} \nMSE = {mse_est_sp}\n" + data)

    # ==== Plotting the data ====
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex='all', constrained_layout=True)

    ax[0].plot(time_arr, meas_height, '.', color='tab:blue', alpha=0.5, label='Measured Height')
    ax[0].plot(time_arr, est_height, '-', color='tab:green', alpha=0.5, label='Est. Height')
    ax[0].plot(time_arr, height_sp, '-', color='tab:red', alpha=0.5,
               label='Set Point (MSE = {:.2f})'.format(mse_est_sp))
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
    fig.suptitle(f'Nonlinear SISO MPC Test - Tube {tube} ({curr_datetime})', fontsize=16)
    plt.savefig(f'nonlin-mpc-data/nonlin-mpc-plot-t{tube}.png', bbox_inches='tight')  # Save the plot as a pdf file
    plt.show()


def plot_io_test(time_arr, height4, fanfactor4, fanspd4, height3, fanfactor3, fanspd3, calibr_status, test_type):
    """ Plots and saves ball height, fan speed, and fan speed factor vs. time for step test, PRBS test, and RGS test """

    # Record current date/time (to include in data/plots)
    curr_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    lower_test_type = test_type.lower()

    # Check if data storage folder exists and create if needed
    storage_exists = os.path.exists(f'./{lower_test_type}-data')
    if not storage_exists:
        os.makedirs(f'./{lower_test_type}-data')
        sleep(1)

    # Save the tube 4 I/O test data as a csv file
    df_t4 = pd.DataFrame({"Time (s)": time_arr, "Fan Factor": fanfactor4, "Fan Speed (rpm)": fanspd4,
                          "Ball Height (cm)": height4})
    df_t4.to_csv(f"{lower_test_type}-data/{lower_test_type}-test-data-tube4.csv", index=False)

    # Prepend the date/time to the csv file
    with open(f"{lower_test_type}-data/{lower_test_type}-test-data-tube4.csv", 'r') as rawf4:
        data4 = rawf4.read()
    with open(f"{lower_test_type}-data/{lower_test_type}-test-data-tube4.csv", 'w') as newf4:
        newf4.write(f"Tube 4 {test_type} Test ({curr_datetime})\n" + data4)

    # Save the tube 3 I/O test data as a csv file
    df_t3 = pd.DataFrame({"Time (s)": time_arr, "Fan Factor": fanfactor3, "Fan Speed (rpm)": fanspd3,
                          "Ball Height (cm)": height3})
    df_t3.to_csv(f"{lower_test_type}-data/{lower_test_type}-test-data-tube3.csv", index=False)

    # Prepend the date/time to the csv file
    with open(f"{lower_test_type}-data/{lower_test_type}-test-data-tube3.csv", 'r') as rawf3:
        data3 = rawf3.read()
    with open(f"{lower_test_type}-data/{lower_test_type}-test-data-tube3.csv", 'w') as newf3:
        newf3.write(f"Tube 3 {test_type} Test ({curr_datetime})\n" + data3)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    twin4_spd, twin4_factor = ax[0].twinx(), ax[0].twinx()          # twinx() gives plots multiple y-axes
    twin3_spd, twin3_factor = ax[1].twinx(), ax[1].twinx()

    twin4_factor.spines["right"].set_position(("axes", 1.1))        # Offset y-axes to display multiple scales
    twin3_factor.spines["right"].set_position(("axes", 1.1))

    h4, = ax[0].plot(time_arr, height4, '.-', color='tab:blue', label='Ball Height')
    fs4, = twin4_spd.plot(time_arr, fanspd4, '.-', color='tab:red', label='Fan Speed')
    ff4, = twin4_factor.plot(time_arr, fanfactor4, '.-', color='tab:green', label='Fan Factor')
    h3, = ax[1].plot(time_arr, height3, '.-', color='tab:blue', label='Ball Height')
    fs3, = twin3_spd.plot(time_arr, fanspd3, '.-', color='tab:red', label='Fan Speed')
    ff3, = twin3_factor.plot(time_arr, fanfactor3, '.-', color='tab:green', label='Fan Factor')

    ax[0].set_xlabel("Time (s)")                # Label/color plots and axis labels
    ax[0].set_ylabel("Ball Height (cm)")
    twin4_spd.set_ylabel("Fan Speed (rpm)")
    twin4_factor.set_ylabel("Fan Speed Factor")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Ball Height (cm)")
    twin3_spd.set_ylabel("Fan Speed (rpm)")
    twin3_factor.set_ylabel("Fan Speed Factor")
    ax[0].yaxis.label.set_color(h4.get_color())
    twin4_spd.yaxis.label.set_color(fs4.get_color())
    twin4_factor.yaxis.label.set_color(ff4.get_color())
    ax[1].yaxis.label.set_color(h3.get_color())
    twin3_spd.yaxis.label.set_color(fs3.get_color())
    twin3_factor.yaxis.label.set_color(ff3.get_color())
    ax[0].set_title(f'Tube 4 - Height and Fan Speed vs. Time (Height: {calibr_status})')
    ax[1].set_title(f'Tube 3 - Height and Fan Speed vs. Time (Height: {calibr_status})')

    # Display current date/time on figure title
    fig.suptitle(f'Tubes 3 & 4 - {test_type} Test ({curr_datetime})', fontsize=16)

    tkw = dict(size=6, width=1.5)                                   # Color axis ticks
    ax[0].tick_params(axis='y', colors=h4.get_color(), **tkw)
    twin4_spd.tick_params(axis='y', colors=fs4.get_color(), **tkw)
    twin4_factor.tick_params(axis='y', colors=ff4.get_color(), **tkw)
    ax[1].tick_params(axis='y', colors=h3.get_color(), **tkw)
    twin3_spd.tick_params(axis='y', colors=fs3.get_color(), **tkw)
    twin3_factor.tick_params(axis='y', colors=ff3.get_color(), **tkw)

    # ax[0].legend(handles=[h4, fs4, ff4])                            # Include legends (turned off for now since
    # ax[1].legend(handles=[h3, fs3, ff3])                              it's been getting in the way of the plots)
    fig.tight_layout()
    # Translate plots downward to make super-title space
    fig.subplots_adjust(top=0.88)
    # Save the plots as a pdf file
    plt.savefig(f"{lower_test_type}-data/{lower_test_type}-test-plots.png", bbox_inches='tight')
    plt.show()


def list_to_array(list4, list3, list2, list1):
    """ Converts 4 lists into 4 numpy arrays """
    array4 = np.array(list4)
    array3 = np.array(list3)
    array2 = np.array(list2)
    array1 = np.array(list1)
    return [array4, array3, array2, array1]


def array_avg(array4, array3, array2, array1):
    """ Calculates the mean of array elements """
    avg4 = int(np.mean(array4))
    avg3 = int(np.mean(array3))
    avg2 = int(np.mean(array2))
    avg1 = int(np.mean(array1))
    return np.array([avg4, avg3, avg2, avg1])


def array_std(array4, array3, array2, array1):
    """ Calculates the standard deviation of array elements """
    std4 = int(np.std(array4))
    std3 = int(np.std(array3))
    std2 = int(np.std(array2))
    std1 = int(np.std(array1))
    return std4, std3, std2, std1


def divide(num, den):
    """ Divides two numbers, with divide-by-zero check """
    if den != 0.0:
        quotient = round(num/den, 6)
    else:
        quotient = None
    return quotient
