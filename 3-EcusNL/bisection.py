_POWERRAT_TOLERANCE = 0.0001

def func(x):

  #Acá poner la función a evaluar
  #(a la que querés encontrarle el 0)
  return x

def bisection(a,b,tolerance):
  '''
  :param a inicio del intervalo en el que se operará
  :param b final del intervalo en el que se operará
  (ATENCIÓN: lineas 21, 22 patean b hasta que se cumplen
  las hipóteis de Bolzano si es que no se cumplen de entrada.
  Si la función no corta el eje X en x>a hay un loop infinito)
  :param tolerance tolerancia de la bisección (float)
  :return c resultado estimado de la función
  '''

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
