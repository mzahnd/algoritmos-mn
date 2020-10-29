import numpy as np

# el valor inicial de x tiene que 
# estar lo suficientemente cerca del 0

def f(x):
  return x**2-2

def df(x):
  return 2*x

x = 1.6

for n in range(20):
  x = x - f(x)/df(x)

print(x)

