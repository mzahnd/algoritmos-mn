import numpy as np

# los valores iniciales xk, xk-1 tienen que 
# estar lo suficientemente cerca del 0
# es un poco más lento que N-R

def f(x):
  return x**2-2

xk = 2
xkm1 = 1.2

for n in range(30):
  xk = xk - (f(xk) * (xk-xkm1))/(f(xk) - f(xkm1))

print(xk)
