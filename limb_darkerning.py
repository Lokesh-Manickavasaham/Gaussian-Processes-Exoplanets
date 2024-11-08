"""

Requried Paramters

ldc (array) --> limb darkening coefficients of non-linear law (4)

"""

import math

def quadratic_limb_darkening(ldc, method):
  if method == 'quadratic':
      u1 = ((12/35)*ldc[0]) + ldc[1] + ((164/105)*ldc[2]) + (2*ldc[3])
      u2 = ((10/21)*ldc[0]) - ((34/63)*ldc[2]) - ldc[3]
      return u1, u2
  else:
      print("Invalid method")

method='quadratic'

u1, u2 = quadratic_limb_darkening(ldc, method)
