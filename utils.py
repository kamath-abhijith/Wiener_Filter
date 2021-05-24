'''

TOOLS FOR FAULTY SENSOR INTERPOLATION
USING WIENER AND KALMAN FILTER

AUTHOR: ABIJITH J KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import numpy as np

from matplotlib import pyplot as plt

# %% SIGNALS

def gen_ar1(alpha, noise_var, num_points=100):
    '''
    Generate auto-regressive 1 sequence with weight alpha

    :param alpha: weight
    :param noise_var: variance of AWGN
    :optional points: length of the sequence
    :optional init: inital value

    :return: ar1 sequence

    '''

    x = np.zeros(num_points)
    x[0] = np.sqrt(noise_var/(1-alpha**2))*np.random.randn()
    for itr in range(1,num_points):
        x[itr] = alpha*x[itr-1] + np.sqrt(noise_var)*np.random.randn()

    return x

# %% OPERATORS

def acorr_matrix(num_points, idx, alpha):
    '''
    Returns the full autocorrelation matrix for
    full Wiener filter
    '''
    
    acorr_mtx = np.zeros((num_points-1, num_points-1))

    for i in range(num_points - 1):
        for j in range(num_points - 1):
            if np.abs(i - j) < idx:
                acorr_mtx[i, j] = alpha ** np.abs(i - j)
            else:
                acorr_mtx[i, j] = alpha ** np.abs(i - j + 1)

    return acorr_mtx

def acorr_sequence(num_points, idx, alpha):
    '''
    Returns the full autocorrelation sequence for
    full Wiener filter
    '''

    acorr_seq = np.zeros(num_points-1)

    for i in range(num_points - 1):
        if i < idx:
            acorr_seq[i] = alpha ** (idx - i)
        else:
            acorr_seq[i] = alpha ** (i - idx + 1)

    return acorr_seq


# %% INTERPOLATORS

def wiener_interpolator1(signal, idx, alpha, noise_var=0.36):
    '''
    Wiener interpolator for one point in AR1 signal

    :param signal: input signal
    :param idx: index to perform interpolation
    :param alpha: weight of AR1 signal
    :param noise_var: variance of awgn on signal

    :return interp: interpolated point
    :return bmse: bayesian mse

    '''

    num_points = len(signal) + 1

    acorr_mtx = acorr_matrix(num_points, idx, alpha)
    acorr_seq = acorr_sequence(num_points, idx, alpha)

    weights = np.linalg.pinv(acorr_mtx) @ acorr_seq
    interp = np.dot(weights, signal)

    acorr_seq0 = noise_var / (1-alpha**2)
    bmse = acorr_seq0 - (acorr_seq.T) @ np.linalg.pinv(acorr_mtx) @ acorr_seq

    return interp, bmse

def wiener_interpolator2(signal, idx, alpha, noise_var=0.36):
    '''
    Two-point average interpolator for one point in AR1 signal

    :param signal: input signal
    :param idx: index to interpolate
    :param alpha: weight of the AR1 signal
    :param noise_var: variance of noise on signal

    :return interp: interpolation
    :return bmse: bayesian mse

    '''

    interp = alpha / (1 + alpha**2) * (signal[idx-1] + signal[idx+1])

    r0 = noise_var ** 2 / (1 - alpha ** 2)
    acorr_seq = np.array([alpha*r0, alpha*r0])
    acorr_mtx = np.array([[r0, (alpha**2)*r0], [(alpha**2)*r0, r0]])

    bmse = r0 - (acorr_seq.T) @ np.linalg.pinv(acorr_mtx) @ acorr_seq

    return interp, bmse

def wiener_interpolator3(signal, idx, alpha, noise_var=0.36):
    '''
    Causal Wiener interpolator for one point in AR1 signal

    :param signal: input signal
    :param idx: index to interpolate
    :param alpha: weight of the AR1 signal
    :param noise_var: variance of noise on signal

    :return interp: interpolation
    :return bmse: bayesian mse

    '''

    num_points = len(signal) + 1

    acorr_mtx = acorr_matrix(num_points, idx, alpha)
    acorr_mtx = acorr_mtx[:idx, :idx]

    acorr_seq = acorr_sequence(num_points, idx, alpha)
    acorr_seq = acorr_seq[:idx]

    weights = np.linalg.pinv(acorr_mtx) @ acorr_seq
    interp = np.dot(weights, signal[:idx])

    acorr_seq0 = noise_var / (1-alpha**2)
    bmse = acorr_seq0 - (acorr_seq.T) @ np.linalg.pinv(acorr_mtx) @ acorr_seq
    
    return interp, bmse

def kf(meas, meas_noise_var, process_noise_var, prediction,
    prediction_var, alpha=0.8):
    '''
    Kalman filter update for noisy measurements of AR1 signal

    :param meas: latest measurement
    :param meas_noise_var: variance of measurements
    :param process_noise_var: variance of process
    :param prediction: signal prediction
    :param prediction_var: variance of prediction

    :return update: signal update
    :return update_var: variance of the update
    :return up_pred: updated prediction
    :return up_pred_var: updated variance of prediction

    '''

    up_pred = alpha * prediction
    up_pred_var = alpha**2 * prediction_var + process_noise_var

    kalman_gain = up_pred_var / (meas_noise_var + up_pred_var)

    update = up_pred + kalman_gain * (meas - up_pred)
    update_var = (1 - kalman_gain) * up_pred_var

    return update, update_var, up_pred, up_pred_var

# %% PLOTTING

def plot_signal(x, y, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper left', line_style='-', line_width=None,
    show=False, xlimits=[0,100], ylimits=[-2.5,2.5], save=None):
    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(x, y, linestyle=line_style, linewidth=line_width, color=plot_colour,
        label=legend_label)
    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return