#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def linealSolverForward(A, b):
    """Solves a system of equations given a lower triangular matrix.

    Arguments:
        A: Lower triangular matrix. Shape: (n, n)
        b: System's solutions. Shape: (n, 1)

    Raises:
        RuntimeError if A is not lower triangular

    Returns:
        A numpy array with the system's X vector. Shape: (n, 1)
    """
    # First we check if the matrix is a lower triangle
    if np.allclose(A, np.tril(A)):
        n = len(b)
        x = np.zeros((n, 1))

        # we apply the forward formula for every element of x
        for k in range(0, n):
            tempSum = []
            for number in range(0, k):
                tempSum.append(-1 * A[k][number] * x[number])
            tempSum = sum(tempSum)
            x[k] = (b[k] + tempSum) / A[k][k]

        return x

    else:
        raise RuntimeError("Matrix A is not lower triangular.")

def linealSolverBackwards(A, b):
    """Solves a system of equations given a upper triangular matrix.

    Arguments:
        A: Upper triangular matrix. Shape: (n,n)
        b: System's solutions. Shape: (n, 1)

    Raises:
        RuntimeError if A is not upper triangular

    Returns:
        A numpy array with the system's X vector. Shape: (n,1)
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


def cholesky(matrix):
    """Performs the cholesky descomposition into two matrices.

    Arguments:
        matrix: Hermitian and definite symmetric matrix to perfom Cholesky's
        descomposition.
            Must have shape (n,n)

    Raises:
        RuntimeError when an invalid matrix is given (ie. non Hermitian or
        non definite symmetric).

    Returns:
        Two numpy arrays.
            Both with shape (n,n)
    """

    # check if matrix is Cholesky compatible
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("Matrix is not square.")

    # Is it symmetric?
    if not np.allclose(matrix, np.transpose(matrix)):
        raise RuntimeError("Matrix is not symmetric.")
    else:
        size = matrix.shape[0]

    # Now the algorithm itself

    lower = np.zeros(matrix.shape)
    
    # We iterate over the triangle from left to right and top to bottom
    for i in range(size):
        for j in range(i + 1):  
           
            # The element belongs to the diagonal:
            if i == j:
                tosqrt = matrix[i][i] - np.sum(lower[i][:i]**2)
                if tosqrt <= 0:
                    raise RuntimeError("Matrix is not definite symmetric.")
                else:
                    lower[i][j] = np.sqrt(tosqrt)
            
            # The element *does not* belong to the diagonal
            else:
                sumatoria = []
                for z in range(j):
                    sumatoria.append(lower[i][z] * lower[j][z])
                sumatoria = sum(sumatoria)
                lower[i][j] = (matrix[i][j] - sumatoria) / lower[j][j]

    upper = np.matrix.transpose(lower)

    return lower, upper


def test_cholesky():
    """Test cholesky.
    
    This function will stop the code execution when it finds an error.

    Returns:
        None
    """

    # ====== Cholesky ======
    # Definite positive matrices
    # This matrices eigenvalues are > 0
    #
    # Eigenvalues sign has been verified using numpy's linalg.eig
    positiveMatrices = (np.array([[2, 3], [3, 6]]), np.array([[1, 2], [2, 5]]),
                        np.array([[57, 40, 7], [40, 78, 6], [7, 6, 13]]),
                        np.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7],
                                  [8, 1, 7, 25]]),
                        np.array([[34, 12, 0, 0], [12, 41, 0, 0], [0, 0, 1, 0],
                                  [0, 0, 0, 1]]))

    # Not definite positive matrices
    # This matrices eigenvalues are < 0
    #
    # Eigenvalues sign has been verified using numpy's linalg.eig
    notPositiveMatrices = (np.array([[4, 2],
                                     [2, -1]]), np.array([[-1, 1], [1, 1]]),
                           np.array([[3, 8, 9], [8, -5, 4], [9, 4, 0]]),
                           np.array([[57, 40, 77, 30], [40, 78, 61, 69],
                                     [77, 61, 13, 59], [30, 69, 59, 81]]))

    # Non Hermitian matrices.
    # As their values are reals, this is the same as 'non symmetric matrices'.
    notHermitian = (np.array([[1, 3], [9, 7]]),
                    np.array([[1, 5, 7], [2, 4, 9], [0, 3, 0]]),
                    np.array([[5, 7, 5, 2], [9, 2, 6, 2], [40, 54, 78, 84],
                              [10, 43, 21, 19]]))
    
    print("="*60, "\nTesting cholesky() function...")

    # Definite positive matrices
    for testMatrix in positiveMatrices:
        print('-'*50)
        print(f'Testing definite positive matrix: \n{testMatrix}')
        try:
            A, B = cholesky(testMatrix)
            comparison = np.allclose(np.matmul(A, B), testMatrix)
            assert comparison, \
                "The last tested matrix did not pass the test."
            print("-> Pass <-")
        except RuntimeError:
            assert False, "The last tested matrix did not pass the test."

    # Not definite positive matrices
    # This matrices should raise and exception (as these aren't valid)
    for testMatrix in notPositiveMatrices:
        print('-'*50)
        print(f'Testing not definite positive matrix: \n{testMatrix}')
        try:
            cholesky(testMatrix)
            assert False, "The last tested matrix did not pass the test."
        except RuntimeError:
            assert True
            print("-> Pass <-")

    # Not Hermitian matrices
    # This matrices should raise and exception (as these aren't valid)
    for testMatrix in notHermitian:
        print('-'*50)
        print(f'Testing not Hermitian matrix: \n{testMatrix}')
        try:
            cholesky(testMatrix)
            assert False, "The last tested matrix did not pass the test."
        except RuntimeError:
            assert True
            print("-> Pass <-")


def leastsq(A, b):
    """Solves a least squares problem.
    Arguments:
        A: Numpy Matrix
        b: Numpy array with the points to approximate.
    Raises:
        RuntimeError if b is not a nx1 vector or the size of A (which is
        square) is different than n, also if either A or b are not arrays.
    Returns:
        A numpy array with the system's approximation
    """

    if type(A) != np.ndarray or type(b) != np.ndarray:
        raise RuntimeError("Input error! One of the leastq arguments is not a "
                           + "numpy array")

    if b.shape[1] != 1 or A.shape[0] != b.shape[0]:
        print('b', A.shape[1], b.shape[0])
        raise RuntimeError("b is not a nx1 vector or the size of A is "
                           + "different than n.")

    # D is A^t*A and E is A^t*b
    # D x = E
    D = np.matmul(np.matrix.transpose(A), A)
    E = np.matmul(np.matrix.transpose(A), b)

    # Separates the lower and upper part of the cholesky decomposition of D
    # lowD uppD x = E
    lowD, uppD = cholesky(D)

    # W is equal to uppD x, and thus is the solution for LowD W = E
    W = linealSolverForward(lowD, E)
    x = linealSolverBackwards(uppD, W)
    return x


if __name__ == "__main__":
    #print("Executed as stand-alone script. Running test function.\n")
    #test_cholesky()
    print(leastsq(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[5], [11], [17]])))