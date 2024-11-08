"""

Requried Paramters

light_curve --> transit light curves to be denoised
time --> observation times

"""

import pywt
import numpy as np
import matplotlib.pyplot as plt

def wavelet_denoising(light_curve, mode):

    # Single-level Discrete Wavelet Transform using Symlet wavelet
    wavelet = pywt.Wavelet('sym4') # Refer PyWavelets doc for available wavelet families
    coeffs = pywt.wavedec(light_curve, wavelet, mode=mode, level=1) # Single-level transform
    
    # Extract the detail coefficients (Dx) for thresholding
    Dx = coeffs[-1]
    
    # Calculate the universal threshold
    sigma = np.median(np.abs(Dx)) / 0.6745 # Estimate of noise level
    N = len(light_curve)
    threshold = sigma * np.sqrt(2 * np.log(N))

    # Perform hard thresholding
    thresholded_Dx = pywt.threshold(Dx, value=threshold, mode='hard')

    # Replace the original detail coefficients with thresholded ones
    coeffs[-1] = thresholded_Dx

    # Reconstruct the denoised signal
    denoised_light_curve = pywt.waverec(coeffs, wavelet, mode=mode)

    return denoised_light_curve

mode = 'reflect' # Refer PyWavelets doc for available modes

denoised_light_curve = wavelet_denoising(light_curve, mode)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 5), layout='constrained')

ax1.plot(time, light_curve, label='Original Light Curve', c='green', alpha=0.5)
ax1.plot(time, denoised_light_curve, label='Denoised Light Curve', c='red', alpha=0.5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Flux')
ax1.legend()

plt.show()