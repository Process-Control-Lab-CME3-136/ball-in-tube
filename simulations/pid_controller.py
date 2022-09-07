"""
================================
Title:  pid_controller.py
Author: Antony Pulikapparambil
Date:   May 2022
================================
"""
# Contains two discrete-time PID controllers (position- and velocity-form)

# NOTE: this file can not be run - it supplements main.py, main_lin_sim.py,
# and main_nonlin_sim.py

import numpy as np

# ===================================
# Superclass: discrete PID controller
# ===================================


class DiscretePID:
    def __init__(self, sample_num):
        self.Kc = 0.000
        self.tau_i = 0.000
        self.tau_d = 0.000
        self.Ts = 0.000
        self.N = sample_num

        self.P_output = 0.000
        self.I_output = 0.000
        self.D_output = 0.000
        self.lower_limit = 0
        self.upper_limit = 0

        self.raw_val_arr = np.array([0.000] * self.N)
        self.filter_val_arr = np.array([0.000] * self.N)
        self.time_arr = np.array([0.000] * self.N)
        self.set_point_arr = np.array([0.000] * self.N)
        self.pid_op_arr = np.array([0.000] * self.N)

        self.filter_order = 0
        self.cutoff_freq = 0.0
        self.fs = 0.0
        self.saturated = False

    def set_kc(self, Kc):
        self.Kc = Kc

    def get_kc(self):
        return self.Kc

    def set_tau_i(self, tau_i):
        self.tau_i = tau_i

    def get_tau_i(self):
        return self.tau_i

    def set_tau_d(self, tau_d):
        self.tau_d = tau_d

    def get_tau_d(self):
        return self.tau_d

    def set_ts(self, Ts):
        self.Ts = Ts
        self.fs = 1 / Ts

    def get_ts(self):
        return self.Ts

    def set_upper_limit(self, upper_limit):
        self.upper_limit = upper_limit

    def get_upper_limit(self):
        return self.upper_limit

    def set_lower_limit(self, lower_limit):
        self.lower_limit = lower_limit

    def get_lower_limit(self):
        return self.lower_limit

    def set_filter_order(self, filter_order):
        self.filter_order = filter_order

    def set_cutoff_freq(self, cutoff_freq):
        self.cutoff_freq = cutoff_freq

    def set_set_point_arr(self, sp):
        self.set_point_arr = sp[0:self.N-1]

    def set_pid_output_arr(self, u):
        self.pid_op_arr = u[0:self.N-1]

    def set_time_arr(self, t):
        self.time_arr = t[0:self.N-1]

    def print_param(self):
        print(f"Kc = {self.Kc}")
        print(f"\u03c4_i = {self.tau_i}")
        print(f"\u03c4_d = {self.tau_d}")
        print(f"Ts = {self.Ts} s")

# ==================================================
# Subclass of discrete PID controller: position-form
# ==================================================


class PositionDiscretePID(DiscretePID):
    def __init__(self, sample_num):
        DiscretePID.__init__(self, sample_num)
        self.PID_output = 0.000
        self.error = 0.000
        self.integ_err = 0.000
        self.deriv_err = 0.000
        self.prev_val = 0.000

    def update(self, current_val, set_point, i):
        """ Generate controller output from current state and set point """
        self.raw_val_arr[i] = current_val                   # "Raw" data before filtering
        # self.time_arr[i] = i * self.Ts                      # Time array (to test filtering)

        self.error = set_point - current_val            # Calculate error

        self.P_output = self.Kc * self.error            # PROPORTIONAL output

        # Integral anti-windup algorithm: clamping
        if not self.saturated:
            self.integ_err += self.error
            self.I_output = self.Kc * self.integ_err * self.Ts / self.tau_i      # INTEGRAL output
        else:
            self.I_output = 0

        # Derivative anti-kick algorithm: negative derivative of position
        # DERIVATIVE output
        self.D_output = -self.Kc * self.tau_d * (current_val - self.prev_val) / self.Ts
        self.prev_val = current_val

        self.PID_output = self.P_output + self.I_output + self.D_output              # TOTAL PID output

        clip_pid_op = np.clip(self.PID_output, self.lower_limit, self.upper_limit)   # Clip total output

        # Determine if saturation occurred
        if self.PID_output == clip_pid_op:
            self.saturated = False
        else:
            self.saturated = True

        return clip_pid_op

# ==================================================
# Subclass of discrete PID controller: velocity-form
# ==================================================


class VelocityDiscretePID(DiscretePID):
    def __init__(self, sample_num):
        DiscretePID.__init__(self, sample_num)
        self.delta_PID_OP = 0.000
        self.curr_err = 0.000
        self.prev1_err = 0.000
        self.prev2_err = 0.000
        self.prev1_val = 0.000
        self.prev2_val = 0.000

    def update(self, current_val, set_point, i):
        """ Generate controller output from current state and set point """
        self.raw_val_arr[i] = current_val  # "Raw" data before filtering
        self.time_arr[i] = i * self.Ts     # Time array (to test filtering)

        self.curr_err = set_point - current_val  # Current error

        self.P_output = self.Kc * (self.curr_err - self.prev1_err)  # PROPORTIONAL output

        # Do NOT need integral anti-windup algorithm since sum of errors is NOT being computed
        self.I_output = self.Kc * self.curr_err * self.Ts / self.tau_i  # INTEGRAL output

        # Anti-kick derivative algorithm: compute negative derivative of position
        # DERIVATIVE output
        self.D_output = -self.Kc * self.tau_d * (current_val - 2 * self.prev1_val + self.prev2_val) / self.Ts

        self.delta_PID_OP = self.P_output + self.I_output + self.D_output  # TOTAL PID output

        self.prev2_err = self.prev1_err     # Previous-previous error
        self.prev1_err = self.curr_err      # Previous error

        self.prev2_val = self.prev1_val     # Previous-previous value
        self.prev1_val = current_val        # Previous value

        return self.delta_PID_OP

    def saturate_op(self, pid_output):
        clip_pid_op = np.clip(pid_output, self.lower_limit, self.upper_limit)  # Clip total output
        if pid_output == clip_pid_op:                           # Determine if saturation occurred
            self.saturated = False
        else:
            self.saturated = True
        return clip_pid_op
