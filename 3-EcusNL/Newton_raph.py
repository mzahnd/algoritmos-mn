import numpy as np

# el valor inicial de x tiene que 
# estar lo suficientemente cerca del 0


#buscamos f(x)=0
#ahora mismo se busca la raiz cuadrada de 2
# f(x) = x**2-2

def f(x):
  return x**2-2

def df(x):
  return 2*x

#guess inicial
x = 1.6

for n in range(20):
  x = x - f(x)/df(x)

print(x)

