import numpy as np
import matplotlib.pyplot as plt


def heatEquation(C, long, Tmax, Xjumps, Tjumps, calt0, u0t, uNt):
    deltaX = long / N
    deltaT = Tmax / M
    r = deltaT / deltaX ** 2

    solMatrix = np.zeros((Xjumps + 1, Tjumps + 1))

    solMatrix[:, 0] = calt0
    solMatrix[0] = u0t
    solMatrix[Xjumps] = uNt

    for j in range(0, Tjumps):
        for i in range(1, Xjumps):
            solMatrix[i, j + 1] = (1 - 2 * r * C) * solMatrix[i, j] + r * C * (
                        solMatrix[i + 1, j] + solMatrix[i - 1, j])
    return solMatrix



#COSAS PARA ANALIZAR UNA ECUACIÓN DE CALOR

# Paso en x
h = 0.2
# Paso en t
k = 0.02
# Cantidad de pasos en x
N = 5
# longitud del palo
long = h * N
# Cantidad de pasos en t
M = 15
# tiempo total del palo
Tmax = k * M

# Condiciones iniciales
# u(x,0) = 1 - |2x - 1|
u0 = [1 - np.abs(2 * xi - 1) for xi in np.linspace(0, N * h, N + 1)]
# u(0,t) = u(1,t) = 0
u0t = 0
uNt = 0
# Constante de la ecuacion de calor
C = 0.5

# ACORDARSE QUE LA CANTIDAD FINAL DE PUNTOS ES N+1, M+1!

matrix = heatEquation(C, long, Tmax, N, M, u0, u0t, uNt)
print(np.round(matrix, 3))

fig, ax = plt.subplots()
for t in range(M + 1):
    y = matrix[:, t]
    x = np.linspace(0, N * h, N + 1)
    ax.plot(x, y, label=str(t * k))
    ax.legend()
plt.show()



