'''

FAULTY SENSOR INPERPOLATION USING KALMAN FILTER

AUTHOR: ABIJITH J KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import numpy as np

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% INITIALISATION

alpha = 0.8
process_noise_var = 0.36
meas_noise_var = 1

signal = utils.gen_ar1(alpha, process_noise_var)
measurements = signal + np.random.randn(len(signal))

idx = 40

# %% KALMAN FILTERING

update = np.zeros(idx)
update_var = np.zeros(idx)
prediction = np.zeros(idx)
prediction_var = np.zeros(idx)

update[0] = measurements[0]
prediction[0] = measurements[0]
update_var[0] = 1
prediction_var[0] = 1

for i in range(1, idx):
    update[i], update_var[i], prediction[i], prediction_var[i] = \
        utils.kf(measurements[i], meas_noise_var, process_noise_var, update[i-1], update_var[i-1], alpha)

# %% COMPARISON

interp3, bmse3 = utils.wiener_interpolator3(signal, idx, alpha)

# %% PLOTS

os.makedirs('./results', exist_ok=True)
path = './results/'

# Signal plots
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_signal(np.arange(0,idx), prediction, ax=ax, plot_colour='green',
    legend_label=r'TRUE')
utils.plot_signal(np.arange(0,idx), update, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED', xaxis_label=r'n', xlimits=[0,40],
    title_text=r'KALMAN FILTERED SIGNAL', save=path+'kalman1')

# Error plots
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_signal(np.arange(0,idx), update_var, ax=ax, plot_colour='green',
    legend_label=r'TRUE')
utils.plot_signal(np.arange(0,idx), prediction_var, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED', xaxis_label=r'n', xlimits=[0,40], ylimits=[0,1],
    title_text=r'VARIANCE OF ERROR', save=path+'kalman2')

# Comparisons
plt.figure(figsize=(12,6))
ax = plt.gca()

ax.scatter(idx, interp3, label=r'CAUSAL WIENER INTERPOLATION w/ ERROR: %.2f' %(bmse3))
ax.scatter(idx, prediction[idx-1], label=r'KALMAN FILTER w/ ERROR: %.2f' %(update_var[idx-1]))
ax.legend(loc='upper left', frameon=True, framealpha=0.8, facecolor='white')
utils.plot_signal(np.arange(0,100), signal, ax=ax, plot_colour='blue',
    legend_label=r'TRUE SIGNAL', title_text=r'INTERPOLATION AT %d' %(idx), save=path+'comparison1')

# %% COMPARISON

idx = 10
for i in range(1, idx):
    update[i], update_var[i], prediction[i], prediction_var[i] = \
        utils.kf(measurements[i], meas_noise_var, process_noise_var, update[i-1], update_var[i-1], alpha)

interp3, bmse3 = utils.wiener_interpolator3(signal, idx, alpha)

# %% PLOTS

plt.figure(figsize=(12,6))
ax = plt.gca()

ax.scatter(idx, interp3, label=r'CAUSAL WIENER INTERPOLATION w/ ERROR: %.2f' %(bmse3))
ax.scatter(idx, prediction[idx-1], label=r'KALMAN FILTER w/ ERROR: %.2f' %(update_var[idx-1]))
ax.legend(loc='upper left', frameon=True, framealpha=0.8, facecolor='white')
utils.plot_signal(np.arange(0,100), signal, ax=ax, plot_colour='blue',
    legend_label=r'TRUE SIGNAL', title_text=r'INTERPOLATION AT %d' %(idx), save=path+'comparison2')
