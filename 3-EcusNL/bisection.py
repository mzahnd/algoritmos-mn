_POWERRAT_TOLERANCE = 0.0001

def func(x):

  #Acá poner la función a evaluar
  #(a la que querés encontrarle el 0)
  return x

def bisection(a,b,tolerance):

  # First interval
  a = 0
  b = 10


  # Expand the interval if f(a)f(b)>0
  while func(a)*func(b) > 0:
    b += 1000

  # Perfom bisection method for finding the root
  for _ in range(10000):
    c = (a + b)/2
    if (b - a) > tolerance:
      if func(a)*func(b) > 0:
        a = c
      else:
        b = c
    else:
      return c
  return c


print(bisection(-1,1, _POWERRAT_TOLERANCE))
