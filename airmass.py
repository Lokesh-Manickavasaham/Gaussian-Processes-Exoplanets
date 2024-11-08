
"""

Requried Paramters

folder_path --> folder path containing FITS files 

### Observatory location
latitude --> Observatory latitude (in degrees)
longitude --> Observatory longitude (in degrees)
elevation --> Observatory elevation (in meters)

"""
import os
import numpy as np
import astropy.units as u
import pylightcurve as plc
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

def calculate_airmass(folder_path, observatory):

    tot_airmass = []
    tot_time = []

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.fits')]

    for file in files:
        with fits.open(file) as hdu:

            header = hdu[0].header
            date = header['DATE-OBS'] # or header['DATE-AVG']

            # Fix target position
            ra = plc.Hours(header['RA'])
            dec = plc.Degrees(header['DEC'])
            targ = plc.FixedTarget(ra, dec)

            # Convert observation time to BJD_TDB
            time_obj = Time(date, format='isot', scale='utc')
            time_in_jd = time_obj.jd
            time_in_bjd_tdb = targ.convert_to_bjd_tdb(time_in_jd, 'JD_UTC')
            tot_time.append(time_in_bjd_tdb)

            # Calculate skycoordinates
            star_coord = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=(u.hourangle, u.deg), frame='icrs')

            # Convert observation time to AltAz frame (for airmass calculation)
            altaz_frame = AltAz(obstime=time_obj, location=observatory)
            star_altaz = star_coord.transform_to(altaz_frame)

            # Calculate airmass
            air_mass = star_altaz.secz  # sec(z) is the airmass
            if air_mass < 0:  # sec(z) becomes negative when the object is below the horizon
                air_mass = float('nan')

            tot_airmass.append(air_mass)
    return tot_time, tot_airmass

observatory = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=elevation*u.m)

tot_time, tot_airmass = calculate_airmass(folder_path, observatory)

time = np.take_along_axis(np.array(tot_time), np.argsort(tot_time), axis=0)
airmass = np.take_along_axis(np.array(tot_airmass), np.argsort(tot_time), axis=0)

plt.plot(time, airmass, ".-")
plt.xlabel("Time (BJD_TDB)")
plt.ylabel("Airmass")
plt.show()