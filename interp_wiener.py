'''

FAULTY SENSOR INPERPOLATION USING WIENER FILTER

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

NUM_REALISATIONS = 1000

ALPHAS = np.arange(0.1,1,0.1)
noise_var = 0.36
idx = 40

TMSE_1 = np.zeros(len(ALPHAS))
TMSE_2 = np.zeros(len(ALPHAS))
TMSE_3 = np.zeros(len(ALPHAS))

BMSE_1 = np.zeros(len(ALPHAS))
BMSE_2 = np.zeros(len(ALPHAS))
BMSE_3 = np.zeros(len(ALPHAS))

# %% MONTE-CARLO ANALYSIS

for alpha_itr, alpha in enumerate(ALPHAS):
    TMSE_ALPHA_1 = np.zeros(NUM_REALISATIONS)
    TMSE_ALPHA_2 = np.zeros(NUM_REALISATIONS)
    TMSE_ALPHA_3 = np.zeros(NUM_REALISATIONS)

    BMSE_ALPHA_1 = np.zeros(NUM_REALISATIONS)
    BMSE_ALPHA_2 = np.zeros(NUM_REALISATIONS)
    BMSE_ALPHA_3 = np.zeros(NUM_REALISATIONS)

    for itr in range(NUM_REALISATIONS):
        signal = utils.gen_ar1(alpha, noise_var)
        measurements = np.delete(signal, idx)

        interp1, BMSE_ALPHA_1[itr] = utils.wiener_interpolator1(measurements, \
            idx, alpha)
        interp2, BMSE_ALPHA_2[itr] = utils.wiener_interpolator2(measurements, \
            idx, alpha)
        interp3, BMSE_ALPHA_3[itr] = utils.wiener_interpolator3(measurements, \
            idx, alpha)

        TMSE_ALPHA_1[itr] = (signal[idx] - interp1) ** 2
        TMSE_ALPHA_2[itr] = (signal[idx] - interp2) ** 2
        TMSE_ALPHA_3[itr] = (signal[idx] - interp3) ** 2

    TMSE_1[alpha_itr] = np.mean(TMSE_ALPHA_1)
    TMSE_2[alpha_itr] = np.mean(TMSE_ALPHA_2)
    TMSE_3[alpha_itr] = np.mean(TMSE_ALPHA_3)
    
    BMSE_1[alpha_itr] = np.mean(BMSE_ALPHA_1)
    BMSE_2[alpha_itr] = np.mean(BMSE_ALPHA_2)
    BMSE_3[alpha_itr] = np.mean(BMSE_ALPHA_3)

# %% PLOT SCORES

os.makedirs('./results', exist_ok=True)
path = './results/'

# Wiener interpolator
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_signal(ALPHAS, TMSE_1, ax=ax, plot_colour='blue', legend_label=r'TMSE',
    xlimits=[0,1], ylimits=[0,1])
utils.plot_signal(ALPHAS, BMSE_1, ax=ax, plot_colour='green', legend_label=r'BMSE',
    xlimits=[0,1], ylimits=[0,1], xaxis_label=r'$\alpha$',
    title_text=r'WIENER INTERPOLATION', save=path+'wiener1')

# Two-point average interpolator
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_signal(ALPHAS, TMSE_2, ax=ax, plot_colour='blue', legend_label=r'TMSE',
    xlimits=[0,1], ylimits=[0,1])
utils.plot_signal(ALPHAS, BMSE_2, ax=ax, plot_colour='green', legend_label=r'BMSE',
    xlimits=[0,1], ylimits=[0,1], xaxis_label=r'$\alpha$',
    title_text=r'TWO-POINT AVERAGE INTERPOLATION', save=path+'wiener2')

# Causal Wiener interpolator
plt.figure(figsize=(12,6))
ax = plt.gca()

utils.plot_signal(ALPHAS, TMSE_3, ax=ax, plot_colour='blue', legend_label=r'TMSE',
    xlimits=[0,1], ylimits=[0,1])
utils.plot_signal(ALPHAS, BMSE_3, ax=ax, plot_colour='green', legend_label=r'BMSE',
    xlimits=[0,1], ylimits=[0,1], xaxis_label=r'$\alpha$',
    title_text=r'CAUSAL WIENER INTERPOLATION', save=path+'wiener3')

# %%
