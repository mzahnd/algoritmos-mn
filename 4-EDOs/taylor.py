# -*- coding: utf-8 -*-

import numpy as np
from numpy import linspace, logspace, diff, zeros
from numpy import cos, sin, exp, log, pi
import matplotlib.pyplot as plt


####################################
# Implementación genérica de Euler
# f(t,x): derivada de x respecto al tiempo
# x0: condición inicial
# t0, tf: tiempo inicial y final
# h: paso de integración
####################################
def euler(f, x0, t0, tf, h):
    """
    Arguments:
        f: Function handler. Must recieve two arguments, t and x. f(t, x).
        x0: Initial value
        t0 / tf: Initial and final time to evaluate
        h: Desiered step

    Returns:
        t, x
        t: Array with the time points used for the function's calculation.
            (X axis on a plot)
            Shape: (n,)
        x: Obtained values at a given t.
            (Y axis on a plot)
            Shape: (n,1)  - One row for each element in t -
    """
    N = int((tf - t0) / h)  # número de puntos
    t = linspace(t0, tf, N + 1)
    n = x0.shape[0]  # dimensión del problema
    x = zeros((n, N + 1))
    x[:, 0] = x0
    for k in range(N):
        x[:, k + 1] = x[:, k] + h * f(t[k], x[:, k])
    return t, x


def taylor(x0, t0, tf, h, *derivatives):
    """
    Generic implementation of taylor of N order
    Arguments:
        x0: Initial value
        t0 / tf: Initial and final time to evaluate
        h: Desiered step
        f: derivatives of the function g, starting from g'
        up to the desired order


    Returns:
        t, x
        t: Array with the time points used for the function's calculation.
            (X axis on a plot)
            Shape: (n,)
        x: Obtained values at a given t.
            (Y axis on a plot)
            Shape: (n,1)  - One row for each element in t -
    """
    N = int((tf - t0) / h)  # número de puntos
    t = linspace(t0, tf, N + 1)
    n = x0.shape[0]  # dimensión del problema
    x = zeros((n, N + 1))
    x[:, 0] = x0
    for k in range(N):
        x[:, k + 1] = x[:, k]
        for index, der in enumerate(derivatives):
            i = index+1
            x[:, k + 1] = x[:, k + 1] + der(t[k], x[:, k]) * (h ** i) / np.math.factorial(i)

    return t, x


# LO SIGUIENTE ES EL EJEMPLO QUE DIÓ EN CLASE USANDO AL FUNCIÓN TAYOR
# SI CORRES LA FUNCIÓN PARA MOSTRAR EL ERROR TENES QUE CAMBIAR AHÍ LOS DATOS
# SEGUN EL GRAOD QUE USES
# SAME PLOTAYLOR

########################
# EJEMPLO
########################
R = 1e3  # Valor de la resistencia
C = 1e-6  # Valor de la capacidad
w = 2.0 * pi * 1000  # frecuencia angular de la señal de entrada
A = 1.0  # amplitud de la señal de entrada
T = 5 * 2 * pi / w  # simulo cinco ciclos

####################################
# Derivada primera de x
def dx(t, x):
    return ((A * cos(w * t) - x) / (R * C))


####################################
# Derivada segunda de x
def d2x(t, x):
    return ((-A * w * sin(w * t) - ((A * cos(w * t) - x) / (R * C))) / (R * C))


####################################
# Plot ejemplo
def plotaylor(h):
    x0 = zeros(1)
    t, xt = taylor(x0, 0, T, h, dx, d2x)
    fig, ax = plt.subplots()
    ax.plot(t, xt[0, :], label='x(t)')
    ax.legend()
    plt.title('Ejercicio')   

    plt.show()


####################################
# Estimación error
def esterrorejemplo(h):

    # i es el grado del polinomio
    i = 2

    x0 = zeros(1)
    t, xt1 = taylor(x0, 0, T, h, dx, d2x)
    t, xt2 = taylor(x0, 0, T, h/2, dx, d2x)

    eet = abs(xt1[0, -1] - xt2[0, -1]) / ((2**i)-1)

    return eet


plotaylor(T / 10000)
print(esterrorejemplo(T / 10000))
