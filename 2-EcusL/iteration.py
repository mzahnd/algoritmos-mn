import numpy as np


# IMPORTANTE!
# si A es estrictamente diagonal dominante, entonces tanto
# Jacobi como Gauss-Seidel convergen si o si.
# Si la Matriz es simétrica y definida positiva G-S converge
# si o si.
# (Ao segundo es chequeable a ojo, para lo primero me
# hice una funcioncita)


def diagonalyDominantCheck(A):
    isDominant = True
    shape = A.shape[0]
    for i in range(shape):

        #Sumo todos los valores absolutos de los elementos
        #de la fila menos el elemento diagonal
        tempSum = 0
        for j in range(shape):
            if j != i:
                tempSum += abs(A[i, j])

        #Me fijo que la fila respete la dominancia diagonal
        if abs(A[i, i]) < tempSum:
            isDominant = False
            break
    return isDominant


def jacobiIteration(A, b):
    x = np.full(b.shape, 1)
    nextx = np.zeros(b.shape)
    for _ in range(100):
        for i in range(x.shape[0]):
            tempSum = 0
            for j in range(A.shape[1]):
                if j != i:
                    tempSum += A[i, j] * x[j]

            nextx[i] = (b[i] - tempSum) / A[i, i]

        x = nextx

    return x


def GaussSeidelIteration(A, b):
    x = np.full(b.shape, 1)
    nextx = np.zeros(b.shape)
    for _ in range(100):
        for i in range(x.shape[0]):
            tempSum1 = 0
            for j in range(i):
                tempSum1 += A[i, j] * nextx[j]
            tempSum2 = 0
            for j in range(i + 1, x.shape[0]):
                tempSum2 += A[i, j] * x[j]

            nextx[i] = (b[i] - tempSum1 - tempSum2) / A[i, i]
        x = nextx

    return x

print(jacobiIteration(np.array([[2, 1], [1, 3]]), np.array([[5], [6]])))
print(GaussSeidelIteration(np.array([[2, 1], [1, 3]]), np.array([[5], [6]])))
