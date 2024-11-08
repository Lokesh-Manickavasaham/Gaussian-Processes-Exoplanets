"""

Requried Paramters

time (array) --> time of observations (in BJD_TDB)
mid_time --> transit mid-time (in same units as time)
period --> orbital period (in days)
a --> semi-major axis (in terms of stellar radius)
i --> inclination (in degrees)
p --> planet radius (in terms of stellar radius)
ldc (array) --> limb darkening coefficients (upto 4)

"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def I(r, ldc):
    n = len(ldc)
    sum = 0
    for i in range(n):
        sum += ldc[i] * (1 - (1 - (r**2))**((i+1)/4))
    return 2*r*(1 - sum)

def I_star_edge(z, p, ldc):
    integral = quad(I, z - p, 1, args=(ldc))
    return integral[0] / (1 - (z-p)**2)

def I_star_center(z, p, ldc):
    integral = quad(I, z - p, z + p, args=(ldc))
    return integral[0] / (4*z*p)

def omega(ldc):
    n = len(ldc)
    c0 = 1 - np.sum(ldc)
    sum = c0 / 4
    for i in range(n): 
        sum += ldc[i] / (i+5)
    return sum

def time_to_z(time, mid_time, period, a, i):
    w = 2 * np.pi / period 
    i = np.deg2rad(i)
    t_rel = np.array(time - mid_time)
    return a * np.sqrt(np.sin(w*t_rel)**2 + (np.cos(i) * np.cos(w*t_rel))**2)

def TransitModel(time, mid_time, period, a, i, p, ldc):
    z = time_to_z(time, mid_time, period, a, i)
    n = len(z)
    omg = omega(ldc)
    flux = np.zeros_like(z)
    for i in range(n):
        if z[i] >= 1+p:
            flux[i] = 1
            # print(f"i = {i}, z= {z[i]:.3f}, 1+p = {1+p:.3f}, z > 1+p, flux={flux[i]:.6f}")
        elif z[i] < 1+p and z[i] > 1-p:
            flux[i] = 1 - ((I_star_edge(z[i], p, ldc) * 7.4 * ((p**2 * np.arccos((z[i]-1)/p)) - ((z[i]-1) * np.sqrt(p**2 - (z[i]-1)**2)))) / 4*omg)
            # print(f"i = {i}, z= {z[i]:.3f}, 1-p = {1-p:.3f}, 1+p = {1+p:.6f}, 1-p < z < 1+p, flux={flux[i]:.6f}")
        elif z[i] <= 1-p:
            flux[i] = 1 - ((I_star_center(z[i], p, ldc) * 7.4 *  p**2) / 4*np.pi*omg)
            # print(f"i = {i}, z= {z[i]:.3f}, 1-p = {1-p:.3f}, z < 1-p, flux={flux[i]:.6f}")
    return flux

transit_flux = TransitModel(time, mid_time, period, a, i, p, ldc)
                            
plt.plot(time-mid_time, transit_flux, ".-")
plt.xlabel("Time from $T_{mid}$ (BJD_TDB)")
plt.ylabel("Relative Flux")
plt.show()
