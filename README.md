# Ball-in-Tube Simulation and Identifiers: Python
Python control algorithms, simulations, and system identifiers for the ball-in-tube experiment, developed by [_Antony Pulikapparambil_](https://github.com/antonypuli) and [_Oguzhan Dogru_](https://github.com/oguzhan-dogru). Contact <dogru@ualberta.ca> for the details/datasets.

Files:
* **main.py** : Runs PID/MPC/Kalman-filter control loops for the ball-in-tube experiment. Requires connection to lab experiment to be run.
* **helper.py** : Contains RabbitMQ client, calibration algorithm, and plotting functions.
* **simulations/main_lin_sim.py** : Runs a linear ARX model simulation with PID controllers.
* **simulations/main_nonlin_sim.py** : Runs a nonlinear model simulation with a nonlinear MPC and unscented Kalman filter.
* **simulations/mpc_kalman.py** : Contains system models, MPCs, and Kalman filters.
* **simulations/pid_controller.py** : Contains discrete-time PID controllers.
* **system identifiers/pysindy-id.py** : Derives a nonlinear system model, given input-output data. Uses the [PySINDy](https://arxiv.org/pdf/2111.08481.pdf) (Python sparse identification of nonlinear dynamics) algorithm to obtain a model, via sparse regression.

The main.py file will send and receive signals from ballsInTubes experiment using Python remotely. The pipeline utilizes the open source
messaging broker RabbitMQ. [_RabbitMQ-Matlab-Client_](https://github.com/ragavsathish/RabbitMQ-Matlab-Client) used 
in the 32-bit MATLAB was written by _ragavsathish_. This was then adapted to a Python-MATLAB connection by _Armianto Sumitro_.


## Lab Usage (_main.py_)
1. Start the RabbitMQ service in the computer with it installed.
2. Turn on the experiment and run MATLAB.
3. Set up a virtual environment and place the _main.py_, _helper.py_, _pid_controller.py_, and _mpc_kalman.py_ files.
4. Ensure the required libraries are installed in your current Python environment. This can be
verified/completed by typing `pip install <library>` in the terminal or, if you're using PyCharm, just add the libraries to your virtual environment.
5. Run the _main.py_ file.
6. The program will prompt you to choose whether to calibrate the device. This calibration takes approximately 20
seconds and optimizes the sensor-level to ball-height conversion. The calibration uses the average sensor readings for approximately 10 samples, at both the bottom and top of the tubes, and returns the standard deviation of the data. Enter `y` to calibrate the device or `n` (or any other keystroke) to skip the calibration and use the preset conversion parameters.
7. The user will then have the choice of conducting either a _PID control test_, _Linear MPC test_, _Adaptive MPC test_, _Nonlinear MPC test,  _Step test_, _PRBS test_, or _RGS test_.    
    * The **_PID control test_** prompts the user to select the type of series PID controller: _position-form_ or _velocity-form_. The measured ball height set heights are fed into the controller, which then calculates the required fan speeds, to minimize the set point error (difference b/w set height and ball height). The mean squared error (MSE) between the measured height and set point is also returned. The PID parameters (_K<sub>c</sub>_ , _tau<sub>i</sub>_ , _tau<sub>d</sub>_) are currently manually tuned, however, once a model for the system is derived, numerical tuning methods can be applied.
    * The **_Linear MPC test_** uses a linear first-order 2-input/2-output ARX model, to modulate the fan speeds, and move the balls to their set heights. Note that since the experiment is nonlinear but the model is linear, the MPC performance is mediocre. The MPC also uses a _Kalman filter_ to estimate the ball heights, using a combination of the linear ARX model and sensor measurements.
    * The **_Adaptive MPC test_** uses a _time-varying_ linear 2-input/2-output ARX model, to control the fans and achieve the ball height set points. The A and B state/input matrices are modified between every identification horizon, using an optimizer algorithm, to minimize the error between the model prediction. This model optimization uses either particle swarm or recursive least squares. A _Kalman filter_ is used for height estimations.
    * The **_Nonlinear MPC test_** uses a SISO nonlinear system model to adjust the fan speeds and control the ball heights. Currently, this test can only control a single tube at a time. The nonlinear MPC is paired to an _Unscented Kalman filter_ to estimate the ball heights.      
    * The **_Step test_** works by lowering all the fan speeds to zero. Then, for the tubes of interest, the fan speeds briefly jump to raise the balls from the tube floor and then these fans are slightly lowered to get the balls to _float_ near the bottom of the tubes. Then the fan speed gets stepped up and the balls to float and settle near the top of the tubes. The fan speeds and ball heights are recorded in arrays to be plotted as functions of time. The plots/data could then be analyzed to obtain a FOPTD model of the system, under the current combination of inputs.
    * The **_PRBS test_** generates a _pseudo-random binary sequence (PRBS)_ of fan speed factor values, by reading a `csv` file, generated in MATLAB. This input sequence was designed, using prior step test data, to excite the system persistently enough to determine its state-space model parameters.
    * The **_RGS test_** generates a _random gaussian sequence (RGS)_ of fan speed speed factor values, using the data from another `csv` file, from MATLAB. This is an alterative to the PRBS, for a system identification signal. However, unlike a PRBS, an RGS allows the system to be excited by periodic multi-level step inputs and sinusoidal inputs, instead.
8. Refer to the methods documentation below for the list of some methods, from the _helper.py_ file, that you can call.


## Methods
| Helper Methods                              | Description                                                                                                                                                                                                                                                                                                                                                                   |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `get_fan_speeds()`                          | Returns an array of fan speeds (units: rpm) from the experiment by calling `getFanSpeeds` in the experiment connected computer.<br/>Output: [`tube4_fanspd`, `tube3_fanspd`, `tube2_fanspd`, `tube1_fanspd`] (type: _int_)                                                                                                                                                    |
| `set_fan_speed(speed, tube)`                | Calls the `setFanSpeed` function in computer `142.244.38.74`. The argument is not the actual fan speed, but an integer in the effective range of 0 to 100. Refer to _Additional Notes (A.)_ below on how to select the desired tube. (No output)                                                                                                                              |
| `get_level(tube)`                           | Returns the ultrasound sensor level-reading, for a single tube, by calling `getLevel#` in the experiment connected computer. Note that `getLevel#` is originally 1-indexed, but `get_level(tube)` is 0-indexed. Again, refer to _Additional Notes (A.)_ below on how to select the desired tube.<br/>Output: `tube_level` (type: _integer_)                                   |
| `calibrate_level(sample_no, sample_period)` | Calibrates the level to height conversion parameters, as explained in the _Usage (C.5)_ section, above. <br/>Output: `bot_levels`, `level_gains`, where <br/>_bot_levels_ = [`tube4_botlvl`, `tube3_botlvl`, `tube2_botlvl`, `tube1_botlvl`] (type: _int_) and, <br/>_level_gains_ = [`tube4_lvlgains`, `tube3_lvlgains`, `tube2_lvlgains`, `tube1_lvlgains`] (type: _float_) |
| `lower_fan_speeds()`                        | Lowers all fan speeds to the allowed minimum of ~1400 rpm. (No output)                                                                                                                                                                                                                                                                                                        |

## Experiment-Specific Notes

### A.) Fan Speed Factor vs. Fan Speed
The `set_fan_speeds(.)` method takes a fan-speed-factor argument (effective range of 0 to 100) that corresponds to the
actual fan speeds as follows. Note that the fan-speed-factor must be an _integer_, else it will be ignored by the
function. Note, the fan strength varies from tube to tube. Note that tube 3 has non-linear fan-speed to height
characteristics (the ball experiences higher acceleration near the top of the tube).

| Fan Speed Factor (dimensionless) | Fan Speed (rpm) |                                  Comment                                   |
|:--------------------------------:|:---------------:|:--------------------------------------------------------------------------:|
|                0                 |      ~1400      |                          Gives minimum fan speed                           |
|               100                |      ~3400      |                          Gives maximum fan speed                           |
|                -1                |     ~1400+      |       Begins increasing from minimum as numbers become more negative       |
|               101                |     ~1400+      | Wraps around effective rpm range and begins increasing with larger numbers |

### B.) Sensor Level vs. Ball Height
The `get_level(.)` method returns the ultrasound level reading. This is measured between the sensor (at top of tube)
and the ball, in arbitrary units. Note that the level sensors are currently broken for tubes 1 & 2: tube 1 gives
a constant reading of ~70 000 and tube 2 gives a constant reading of 0. Note that tube 4 has noticeable sensor noise
near the top of the tube, which can also result in a high standard deviation (std) during calibration of top levels. If
a high std is returned, for any level readings, consider redoing the calibration or skipping it and using the
pre-set conversion values, instead.

| Sensor Level Reading (dimensionless) | Ball Height (cm) |                Comment                 |
|:------------------------------------:|:----------------:|:--------------------------------------:|
|               ~13 000                |        0         | Bottom of tube (furthest from sensor)  |
|                 ~400                 |       100        |    Top of tube (closest to sensor)     |

## Related Applications/Publications
[1] **Velocity PID**: Dogru, O., Velswamy, K., Ibrahim, F., Wu, Y., Sundaramoorthy, A. S., Huang, B., ... & Bell, N. (2022). Reinforcement learning approach to autonomous PID tuning. Computers & Chemical Engineering, 161, 107760.<br/>
[2] **MPCs and Kalman filter**: Dogru, O., Chiplunkar, R., & Huang, B. (2022). Skew Filtering for Online State Estimation and Control.<br/>
[3] **Kalman filter**: Dogru, O., Chiplunkar, R., & Huang, B. (2021). Reinforcement learning with constrained uncertain reward function through particle filtering. IEEE Transactions on Industrial Electronics, 69(7), 7491-7499.
