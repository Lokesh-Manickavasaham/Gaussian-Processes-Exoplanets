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

def TransitModel(time, p, ldc, mid_time, period, a, i):
    z = time_to_z(time, mid_time, period, a, i)
    n = len(z)
    omg = omega(ldc)
    # print(z, z[33])
    flux = np.zeros_like(z)
    for i in range(n):
        if z[i] >= 1+p:
            flux[i] = 1
            # print(f"i = {i}, z= {z[i]:.3f}, 1+p = {1+p:.3f}, z > 1+p, flux={flux[i]:.6f}")
        elif z[i] < 1+p and z[i] > 1-p:
            flux[i] = 1 - ((I_star_edge(z[i], p, ldc) * ((p**2 * np.arccos((z[i]-1)/p)) - ((z[i]-1) * np.sqrt(p**2 - (z[i]-1)**2)))) / 4*omg)
            # print(f"i = {i}, z= {z[i]:.3f}, 1-p = {1-p:.3f}, 1+p = {1+p:.6f}, 1-p < z < 1+p, flux={flux[i]:.6f}")
        elif z[i] <= 1-p:
            flux[i] = 1 - ((I_star_center(z[i], p, ldc) *  p**2) / 4*np.pi*omg) # np.pi was not there
            # print(f"i = {i}, z= {z[i]:.3f}, 1-p = {1-p:.3f}, z < 1-p, flux={flux[i]:.6f}")
    return flux

time = np.array([2460356.13627585, 2460356.13965766, 2460356.14301969,
                2460356.14639466, 2460356.14975997, 2460356.15311847,
                2460356.15648746, 2460356.15986384, 2460356.16323127,
                2460356.16659277, 2460356.16995081, 2460356.17331469,
                2460356.17668787, 2460356.18006517, 2460356.18342323,
                2460356.18680313, 2460356.19017644, 2460356.19353   ,
                2460356.19689122, 2460356.2002423 , 2460356.20361615,
                2460356.20697806, 2460356.21034599, 2460356.21371761,
                2460356.21708981, 2460356.22044473, 2460356.22381273,
                2460356.22716893, 2460356.23053009, 2460356.23390139,
                2460356.23726321, 2460356.24063996, 2460356.24399207,
                2460356.24735558, 2460356.25071706, 2460356.25407384,
                2460356.2574345 , 2460356.26080834, 2460356.2641781 ,
                2460356.26755198, 2460356.27093299, 2460356.27430148,
                2460356.277675  , 2460356.28104541, 2460356.28440135,
                2460356.28776038, 2460356.29112265, 2460356.2944781 ,
                2460356.29784248, 2460356.30182737, 2460356.30516402,
                2460356.30853634, 2460356.31188737])

p = 0.091                                                    # ratio of planet radius to stellar radius
i = 83.8                                                     # inclination (in degrees)
a = 5.53                                                     # ratio of semi-major axis to stellar radius
period = 2.7347656                                           # period of orbit (in days)
ldc = [0.3835939, 0.1003784, 0.4090374, -0.2413474]          # Limb darkening coefficients (upto 4)
mid_time = 2460356.191940856                                 # transit mid-time (in same units as time)

transit_flux = TransitModel(time, p, ldc, mid_time, period, a, i)
                            
plt.plot(time-mid_time, transit_flux, ".-")
plt.show()