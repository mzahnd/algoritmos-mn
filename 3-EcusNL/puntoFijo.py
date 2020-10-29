import numpy as np

#El 0 tiene que estar encuadrado en [a,b]
#g(x) pertenece a [a,b] si x pertenece a [a,b]
# la derivada de g(x) en [a,b] es MENOR a 1

def g(x):
  return np.exp(x)
x = 0.2

for n in range(300):
  x = g(x)

print(x)
