import math 

# Limb darkening coefficients of non linear law (c1, c2, c3, c4)
ldc = [0.3369471782720396,
  0.20582268551248764,
  0.41483350298449173,
  -0.25608656100515886]

method='quadratic'

if method == 'quadratic':
    u1 = ((12/35)*ldc[0]) + ldc[1] + ((164/105)*ldc[2]) + (2*ldc[3])
    u2 = ((10/21)*ldc[0]) - ((34/63)*ldc[2]) - ldc[3]
    print(f"u1 = {u1:.5g}, u2 = {u2:.5g}")
else:
    print("Invalid method")
