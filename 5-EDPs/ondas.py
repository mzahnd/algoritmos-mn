#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

##################################################
#Ecuación de onda
#
#d^2/dt^2 u = c^2*d^2/dx^2 u     to<t<tf, a<x<b
#
#u(a,t) = c1, u(b,t) = c2 para todo t
#u(x,to) = f(x) para todo x
#
#[x,t,u] = onda(f,g.c1,c2,a,b,to,tf,c,n,m)
#
#Argumentos de entrada:
#=====================
#f 		= función de la condición inical
#g 		= función de la condición inical (derivada)
#c1,c2 	= condiciones de borde
#a,b	= límites de la variable espacial
#to,tf	= tiempos inicial y final
#c		= constante de la ecuación diferencial
#n		= número de intervalos en la variable espacial
#m		= número de intervalos en la variable temporal
#
#Argumentos de salida:
#====================
#x		= grilla en la variable espacial
#t		= grilla en la variable temporal
#u		= resultado

def waveEq(f, g, a, b, t0, tf, c, xDots, tDots):
    """Solve wave equation:  d^2/dt^2 u(x,t) = c^2 * d^2/dx^2 u(x,t)
    
    For t0 < t < tf  and  a < x < b

    Contour conditions:
    u(a,t) = u(b,t) = 0     For t >= 0


    Algorithm refractored from lecture example.

    Arguments:
        f       : Initial condition function. Valid for x in [a;b]
        g       : Derivative of f.            Valid for x in [a;b]
        a, b    : Space interval
        t0, tf  : Time interval
        c       : Wave speed
        xDots   : Grill dots in space interval
        tDots   : Grill dots in time interval

    Returns:
        x, t, u

        x       : Grill for the spatial variable
        t       : Grill for the time variable
        u       : Equation numerical solution

    Raises:
        ArithmeticError.
    """

    class Grill:
        def __init__(self, sizeX, sizeY):
            self.X = sizeX
            self.T = sizeY

    # Delta x
    deltaX = (b-a)/(xDots-1)
    
    # Delta t
    deltaT = (tf-t0)/(tDots-1)

    #Grill
    grill = Grill(np.linspace(a, b, xDots), np.linspace(t0, tf, tDots))
    
    # Numeric integration constants
    r = c * deltaT / deltaX

    if (r > 1):
        raise ArithmeticError('r value ends up being greater than 1. '
                    'Please modify your grill and/or wave speed parameters.')

    rSquared = r * r
    rSquaredOverTwo = rSquared / 2
    oneMinusRSqd = 1 - rSquared
    twiceOneMRS = 2 * oneMinusRSqd
    

    
    u = np.zeros((xDots, tDots))
    
    # Initial condition

    # First row
    u[1:-1, 0] = f(grill.X[1:-1])
    
    # Second row (derivative)
    # NOT Using an approximation for the second derivative (needs to add it)
    #u[1:-1, 1] = u[1:-1, 0] + g(grill.X[1:-1]) * deltaT

    # Using an approximation for the second derivative
    u[1:-1, 1] = oneMinusRSqd * u[1:-1, 0] + deltaT * g(grill.X[1:-1]) \
                + rSquaredOverTwo * (u[2:, 0] + u[0:-2, 0])

    for j in range(2,tDots):
        u[1:-1, j] = twiceOneMRS * u[1:-1, j-1] \
                    + rSquared * (u[0:-2, j-1] + u[2:, j-1]) - u[1:-1, j-2]

    return grill.X,grill.T,u



def testWave():
    """Test for the Wave equation algorithm taken from lecture.
    """
    f = lambda x: (np.sin(np.pi*x)+np.sin(2*np.pi*x))
    g = lambda x: np.zeros(x.size)
    
    a = 0.0
    b = 1.0
    t0 = 0.0
    tf = 1.0
    c = 2.0
    n = 111
    m = int(n*c - 1)

    print('Calculating equation using:')
    print(f'\tx in ({a} ; {b})\n\tt in ({t0}; {tf})')
    print(f'\n\tc = {c}\n\tDots for x = {n}\n\tDots for t = {m}')
    print(f'\tr = {c * ((tf-t0)/(m-1)) / ((b-a)/(n-1))}\n')

    x, t, u = waveEq(f, g, a, b, t0, tf, c, n, m)
    
    print('-> Finished <-')

    # Analytical solution
    ua = np.zeros(u.shape)
    for k in range(x.shape[0]):
        ua[k,:] = np.sin(np.pi * x[k]) * np.cos(2.0 * np.pi * t) \
                    + np.sin(2.0 * np.pi * x[k]) * np.cos(4.0 * np.pi * t)
    print('Error: ', np.max(np.abs(u-ua)))
    
if __name__ == "__main__":
    testWave()
