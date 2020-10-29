#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Descomposición en valores singulares

Utiliza numpy para el cálculo de autovalores y autovectores
"""

import numpy as np
import random as rnd

def svd(matrix):
    """

    Returns:
        U, S, V.T

        U: U matrix
        S: Sigma matrix
        V.T: V matrix already transposed
    """
    if (type(matrix) is not np.ndarray):
        raise RuntimeError("Expected numpy matrix.")

    # Eigenvalues of A^T*A
    Dt, V = np.linalg.eig( np.matmul(matrix.T, matrix) )
    
    # Sort them before going on
    Dt, V = _sortMatrices(Dt, V)

    # Calculate each sigma.
    # This will help us calculate each column on the U matrix
    sigmas = np.sqrt(Dt)
    
    # Generate U
    maxSize = max(np.shape(matrix))
    U = np.zeros((maxSize, maxSize))

    for pos in range(np.shape(sigmas)[0]):
        U[:,pos] = np.matmul(matrix, V[:,pos]) / sigmas[pos]

    currentCol = 0
    while currentCol < maxSize:
        # Fill empty columns with numbers
        if (np.allclose(U[:,currentCol], np.zeros((maxSize, 1)))):
            rnd.seed()
            for row in range(maxSize):
                U[row,currentCol] = rnd.random()
            
            U = _matrixOrthonormalization(U, column=currentCol)


        currentCol += 1

    # Generate Sigma matrix
    S = np.zeros(( np.shape(U)[1], np.shape(V)[0] ))
    
    for rowEtCol in range(0, len(sigmas), 1):
        S[rowEtCol][rowEtCol] = sigmas[rowEtCol]

    return U, S, V.T


def pesudoInverse(matrix):
    """Calculate the Moore-Penrose pseudo-inverse of a matrix.

    Uses SVD to achieve it.

    Arguments:
        matrix: Numpy matrix to calculate its pseudo-inverse.

    Returns:
        Numpy matrix A+ (the pseudo-inverse).
    """

    # Calculate the SVD matrices
    U, S, Vt = svd(matrix)

    # A+ = V * S+ * U.T => The sigma (S) matrix shape needs to be inverted.
    pseudoSigma = S.T
    sigmaShape = np.shape(pseudoSigma)

    # Recalculate Sigma as Sigma+ (each value != 0 is now 1/value)
    for row in range(0, sigmaShape[0]):
        for col in range(0, sigmaShape[1]):
            # pylint: disable=E1136  # pylint/issues/3139
            if pseudoSigma[row][col] != 0:
                pseudoSigma[row][col] = 1 / pseudoSigma[row][col]

    # Return A+, being A+ = V * S+ * U.T
    return np.matmul(np.matmul(Vt.T, pseudoSigma), U.T)

def _sortMatrices(matrixA, matrixB):
    ascendingOrder = np.argsort(matrixA)

    sortedA = np.zeros(np.shape(matrixA))
    sortedB = np.zeros(np.shape(matrixB))

    current = 0
    for i in ascendingOrder[::-1]:
        sortedA[current] = matrixA[i]
        sortedB[:,current] = matrixB[:,i]
        current += 1
    
    return sortedA, sortedB


def _norm(v):
    """Calculate the norm of a vector

    Arguments:
        v: ndarray with vector shape: (1,n) , (n,1) or (n,)

    Returns:
        Floating point number with the norm.
    """
    if type(v) is not np.ndarray:
        raise RuntimeError('Please provide a numpy array.')
    elif np.ndim(v) > 2:
        raise RuntimeError('Too many dimensions!')
    elif np.ndim(v) == 2 and np.shape(v)[0] != 1 and np.shape(v)[1] != 1:
        raise RuntimeError('This is not a numpy vector.')
    cumsum = 0
    
    flatv = v.flatten()
    
    for element in flatv:
        cumsum += element**2

    return np.sqrt(cumsum)


def _projectOperator(v, u):
    return np.inner(v, u)*u


def _matrixOrthonormalization(matrix, column=0):
    """Orthogonalize a given column of a matrix using Gram-Schmidt.

    matrix: Matrix to perform algorithm over
    column: Column to orthonormalize. Set to 0 to perform over the whole
            matrix.
    """

    # Projections
    columnsToTheLeft = np.shape(matrix[0,:])[0] - abs(column)

    if columnsToTheLeft < 0:
        raise RuntimeError('There are no columns at the left of this matrix.')

    for currentColumn in range(0, columnsToTheLeft, 1):
        matrix[:,column] = matrix[:,column] - \
            _projectOperator(matrix[:,column].T, matrix[:,currentColumn])
    

    # Normalize resulting vector
    matrix[:,column] = matrix[:,column]/_norm(matrix[:,column])

    return matrix


def test():
    matrix = np.array([[12, 4], [3, 2], [6, 2]])
    U, S, Vt = svd(matrix)

    # Debug
    print('U: \n', U, '\n')
    print('U.T*U: \n', np.matmul(U.T, U), '\n')

    print('S: \n', S, '\n')

    print('V.T: \n', Vt, '\n')

    print('U*S*V.T = \n', np.matmul(np.matmul(U, S), Vt))

    print('A: \n', matrix)


    print('V*S*U.T = \n', pesudoInverse(matrix), '\n')

    print('Numpy solution: \n', np.linalg.pinv(matrix))

if __name__ == "__main__":
    print("Executed as stand-alone script. Running test function.\n")
    test()
