import numpy as np


def linealSolverBackwards(A, b):
    """Solves a system of equations given a upper triangular matrix.
    Arguments:
        A: Triangular Matrix
        b: System's solutions
    Raises:
        RuntimeError if A is not upper triangular
    Returns:
        A numpy array with the system's X vector.

        :param A: Lower triangular matrix.
        :param b: System's solutions
        :return: The vector solution for the equation system
    """
    if np.allclose(A, np.triu(A)):
        n = len(b)
        x = np.zeros((n, 1))

        for k in reversed(range(0, n)):

            tempSum = []
            for number in range(k + 1, n):
                tempSum.append(-1 * A[k][number] * x[number])

            tempSum = sum(tempSum)
            x[k] = (b[k] + tempSum) / A[k][k]

        return x

    else:
        raise RuntimeError("Matrix A is not upper triangular.")


def QRdec(A):
    '''
    :param A: matriz a descomponer en Q, R. Puede ser rectangular.
    '''
    numberRowsA = A.shape[0]
    numberColsA = A.shape[1]

    #Q es cuadrada
    Q = np.zeros((numberRowsA, numberRowsA))
    #R tiene la msima forma que la matriz original
    R = np.zeros(A.shape)


    # Aplico Gram-Schmidt como lo hace en el video
    # para ir llenando a Q y voy completando R al
    # mismo tiempo

    for j in range(numberColsA):
        Q[:, j] = A[:, j]
        for col in range(j):
            dotProd = np.dot(A[:, j], Q[:, col])
            R[col][j] = dotProd
            Q[:, j] = Q[:, j] - dotProd * Q[:, col]

        colNorm = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / colNorm
        R[j, j] = colNorm

    # LLeno las columnas de Q que faltan con numeros aleatorios
    # y las ortonormalizo respecto de las otras columnas

    for j in range(j+1, numberRowsA):
        Q[:, j] = np.random.rand(numberRowsA)
        for col in range(j):
            dotProd = np.dot(Q[:, j], Q[:, col])
            Q[:, j] = Q[:, j] - dotProd * Q[:, col]
        colNorm = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / colNorm

    return Q, R


def leastsqQR(A, b):
    '''
    :param A: matriz de coeficientes del sistema de ecuaciones
    :param b: vector solución (IMPORTANTE: b.shape[1] = 1, es vector columna)
    '''

    Q, R = QRdec(A)
    R_1 = R[:A.shape[1], :]
    Q_T = np.transpose(Q)
    Q1_T = Q_T[:A.shape[1], :]
    Q2_T = Q_T[A.shape[1]:, :]

    errorCuad = np.linalg.norm(np.dot(Q2_T, b))**2

    # Para minimizar la diferencia entre A*x
    # y b se llega al sistema R_1 * x = (Q_1)^T * b
    # error residual (error cuadrático): (norma(Q_2T*b))^2

    # VV tener en cuenta que err siempre va printearse como > 0
    # por redondeo, aunque los valores de entrada sean exactos
    # En ese caso no va a ser error "posta" (o sea residuo de
    # cuadrados mínimos)

    print("Error cuadrático: {}".format(errorCuad))
    x = linealSolverBackwards(R_1, np.dot(Q1_T, b))

    return x

print(leastsqQR(np.array([[1,2],[3,4],[5,6]]), np.array([[5],[11], [17]])))



