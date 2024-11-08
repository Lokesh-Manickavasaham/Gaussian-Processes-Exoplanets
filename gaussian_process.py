"""

Requried Paramters

flux --> relative flux
time --> time (in BJD_TDB)
airmass --> airmass (for plotting)
flux_unc --> uncertainity in flux (used to (re)compute GP)
fwhm --> full-width at half-maximum of spectral profile (as an additional regressor)
Tc --> actual mid-time from observation (in BJD_TDB)

### Fixed Parameters of Transit Model
period --> orbital period (in days)
e --> eccentricity
w --> longitude of periastron (in degrees)
[u1, u2] --> quadratic limb darkening coefficients

### Initial Guesses for Transit Model Parameters
rp_over_rs --> planet radius (in units of stellar radii)
mid_time --> mid time (in BJD_TDB)
inclination --> orbital inclination (in degrees)
sma_over_rs --> semi-major axis (in units of stellar radii)

metric_time --> kernel scale length for time as a regressor
metric_fwhm --> kernel scale length for fwhm as a regressor
amp --> kernel amplitude

### Gaussian priors for a and i
a_mu --> mean of semi-major axis 
a_std --> standard deviation of semi-major axis 
i_mu --> mean of inclination angle
i_std --> standard deviation of inclination angle

### Bounds for mean hyperparameters
rp_lower --> lower limit for planet radius 
rp_upper --> upper limit for planet radius 
mid_time_lower --> lower limit for mid time
mid_time_upper --> upper limit for mid time
a_lower --> lower limit for semi-major axis
a_upper --> upper limit for semi-major axis
i_lower --> lower limit for inclination angle
i_upper --> upper limit for inclination angle

### Bounds for kernel hyperparameters
amp_lower --> lower limit for kernel amplitude
amp_upper --> upper limit for kernel amplitude
time_metric_lower --> lower limit for scale length for time as a regressor
time_metric_upper --> upper limit for scale length for time as a regressor
fwhm_metric_lower --> lower limit for scale length for fwhm as a regressor
fwhm_metric_upper --> upper limit for scale length for fwhm as a regressor

### PyDE and emcee parameters
npop --> size of the parameter vector population (~ nwalkers in emcee)
ndim --> (generally the length of bounds array) (~ ndim in emcee)
ngen --> no. of generations to run (~ nsteps in emcee)

"""

import emcee
import batman
import george
import numpy as np
from george import kernels
from pyde.de import DiffEvol
import matplotlib.pyplot as plt
from george.modeling import Model
    
global period, e, w, u1, u2
global a_mu, a_std, i_mu, i_std
global rp_lower, rp_upper, mid_time_lower, mid_time_upper, a_lower, a_upper, i_lower, i_upper
global amp_lower, amp_upper, time_metric_lower, time_metric_upper, fwhm_metric_lower, fwhm_metric_upper

def transit_model(rp, t0, a, i, time):
    params = batman.TransitParams()                     
    params.t0 = t0     
    params.per = period
    params.rp = rp 
    params.a = a       
    params.inc = i     
    params.ecc = e 
    params.w = w      
    params.limb_dark = "quadratic"
    params.u = [u1, u2]
    m = batman.TransitModel(params, time, fac=3.0)      
    return m.light_curve(params)

def log_joint_prior(params):
    rp, mid_time, a, i, amp, time_metric, fwhm_metric = params
    uniform_params = np.array([rp, mid_time, a, i, amp, time_metric, fwhm_metric])
    lower_bounds = [rp_lower, mid_time_lower, a_lower, i_lower, amp_lower, time_metric_lower, fwhm_metric_lower]
    upper_bounds = [rp_upper, mid_time_upper, a_upper, i_upper, amp_upper, time_metric_upper, fwhm_metric_upper]
    if np.all(lower_bounds < uniform_params) and np.all(uniform_params < upper_bounds):
        log_prior_a = -0.5 * ((a - a_mu)**2 / a_std**2 + np.log(2*np.pi*(a_std**2)))
        log_prior_i = -0.5 * ((i - i_mu)**2 / i_std**2 + np.log(2*np.pi*(i_std**2)))
        return log_prior_a + log_prior_i
    return -np.inf

def log_joint_likelihood(params, Y):
    rp, mid_time, a, i, amp, time_metric, fwhm_metric = params
    gp.set_parameter_vector([rp, mid_time, a, i, amp, time_metric, fwhm_metric])
    return gp.log_likelihood(Y, quiet=True) 

def log_joint_posterior(params, Y):
    lp = log_joint_prior(params)
    ll = log_joint_likelihood(params, Y)
    if np.isfinite(lp) and np.isfinite(ll):
        return -(lp + ll)
    return np.inf

class TransitModel(Model):
    parameter_names = ("rp", "t0", "a", "i")
    def get_value(self, X):
        time = X[:, 0]
        transit = transit_model(self.rp, self.t0, self.a, self.i, time)
        return transit

bounds = np.vstack([[rp_lower, mid_time_lower, a_lower, i_lower, amp_lower, time_metric_lower, fwhm_metric_lower],
                    [rp_upper, mid_time_upper, a_upper, i_upper, amp_upper, time_metric_upper, fwhm_metric_upper]]).T

X = np.vstack([time, fwhm]).T
Y = flux

# Initialize GP
mean_function = TransitModel(rp=rp_over_rs, t0=mid_time, a=sma_over_rs, i=inclination)
kernel = kernels.Product(kernels.ConstantKernel(log_constant=np.log(amp**2), ndim=2), kernels.ExpSquaredKernel(metric=[metric_time, metric_fwhm], ndim=2))
gp = george.GP(kernel=kernel, fit_kernel=True, mean=mean_function, fit_mean=True)
gp.compute(X, flux_unc)

# Optimize hyperparameters using Differential Evolution
de = DiffEvol(log_joint_posterior, bounds, npop, c=0.5, maximize=False, args=[Y])
de.optimize(ngen)
optimized_params = de.minimum_location

# Marginalize hyperparameters using MCMC starting with the optimized values
pos = optimized_params + 1e-4 * np.random.randn(npop, ndim)
sampler = emcee.EnsembleSampler(npop, ndim, log_joint_posterior, args=[Y], threads=15)
sampler.run_mcmc(pos, ngen, progress=True)
samples = sampler.get_chain(discard=1000, thin=10, flat=True)
log_probs = sampler.get_log_prob(discard=1000, thin=10, flat=True)
max_prob_index = np.argmax(log_probs)
best_params_mcmc = samples[max_prob_index]

# Obtain the uncertainities
percentiles = np.percentile(samples, [16, 50, 84], axis=0)
medians = percentiles[1]
lower_uncertainties = medians - percentiles[0]
upper_uncertainties = percentiles[2] - medians

# Recompute using the marginalized hyperparameters
gp.set_parameter_vector(best_params_mcmc)
gp.compute(X, flux_unc)

# Predict the model
mu, var = gp.predict(Y, X, return_var=True)
sigma = np.sqrt(var)

# Detrend the light curves
transit_flux = mean_function.get_value(X)
detrended_flux = Y / (mu/transit_flux)
std = np.std(detrended_flux - transit_flux)

# For better visualisation
x = 1
offset= 0.0008

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 6), sharex=True)
plt.subplots_adjust(hspace=0)

ax1.plot((time-Tc)*24, Y+0.01, '.', color='navy', label='Observed')
ax1.plot((time-Tc)*24, mu+0.01, ls='--', color='grey', label='GP Prediction', alpha=0.7)
ax1.plot((time-Tc)*24, detrended_flux, '.', markerfacecolor='white', color='black', label='Detrended Flux')
ax1.plot((time-Tc)*24, transit_flux, 'r', label='Mean Transit Model')
ax1.fill_between((time-Tc)*24, mu+0.01 + x*sigma, mu+0.01 - x*sigma, color='blue', alpha=0.2)
ax1.set_ylabel('Relative Flux')
ax1.legend(fontsize='small', frameon=False, loc='upper left')

ax3 = ax1.twinx()
ax3.plot((time-Tc)*24, airmass, '.', color='grey', label='Airmass', alpha=0.05)
ax3.set_ylabel('Airmass')

ax2.plot((time-Tc)*24, detrended_flux - transit_flux, 'k.', alpha=0.8)
ax2.axhline(y=0, color='r', alpha=0.5)
ax2.text((time[0]-Tc)*24 - 0.8, offset, f'$\sigma$ = {1e6*std[-1]:.1f} ppm', fontsize=8, color='k', verticalalignment='center', horizontalalignment='left')
ax2.set_xlabel('Time from mid-transit [hours]')
ax2.set_ylabel('Residuals')

plt.show()