#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def isPositiveDefinite(matrix, debug=False):
    """
    Taken from: https://stackoverflow.com/a/44287862
    """

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing symmetry of the matrix: \n', matrix)
        print()

    # Check if it is symmetric
    if isSymmetric(matrix, debug):
        # Using cholesky to check the check if it's positive deffinite
        try:
            np.linalg.cholesky(matrix)
            print(u'[\u2713] Matrix is positive definite.')
            return True
        except np.linalg.LinAlgError:
            print('[X] Matrix is not positive definite.')
            return False
    else:
        return False

def isSymmetric(matrix, debug=False):

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing symmetry of the matrix: \n', matrix)
        print()

    if np.allclose(matrix, matrix.T):
        print(u'[\u2713] Matrix is symmetric.')
        return True
    else:
        print('[X] Matrix is NOT symmetric.')
        return False

def diagDominant(matrix, debug=False):
    """Print if a matrix is diagonal dominant

    Taken from: https://stackoverflow.com/a/43074838
    """

     # '\u2713' = ✓ (unicode tick character)

    if debug:
        print('Analysing matrix: \n', matrix)
        print()

    D = np.diag(np.abs(matrix)) # Find diagonal coefficients

    S = np.sum(np.abs(matrix), axis=1) - D # Find row sum without diagonal
    
    if np.all(D > S):
        print(u'[\u2713] Matrix is strictly diagonally dominant.')
        return True
    else:
        print('[X] Matrix is NOT strictly diagonally dominant')
        return False

def useJacobiGaussSeidel(matrix, debug=False):
    """Given an nxn matrix, tells if Gauss-Seidel or Jacobi could be used.

    Lineal system of equations: Ax = b
    
    Arguments:
        matrix: Should be matrix A of the system.

    Returns:
        True if either Jacobi or Gauss-Seidel could be used    
    """
    matrixNumDim = np.ndim(matrix)
    if (matrixNumDim != 2):
        raise RuntimeError("I'm not prepared for this. </3")

    matrixShape = np.shape(matrix)
    if (matrixShape[0] != matrixShape[1]):
        print('[X] Matrix is not squared')
        return False
    
    jacobi = diagDominant(matrix, debug)
    posDef = isPositiveDefinite(matrix, debug)

    if posDef:
        print(u'[\u2713] Gauss-Seidel can be used.')
        print('[X] Jacobi can not be used')
        return True

    elif jacobi:
        print(u'[\u2713] Gauss-Seidel can be used.')
        print(u'[\u2713] Jacobi can be used')
        return True

    else:
        print('[X] Gauss-Seidel can not be used.')
        print('[X] Jacobi can not be used')
        return False
        


if __name__ == "__main__":
    # Strictly diagonally dominant
    I = np.array([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
    print('Testing: \n', I,'\n')
    diagDominant(I)

    # Diagonally dominant (not strictly)
    I = np.array([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
    print('Testing: \n', I,'\n')
    diagDominant(I)

    # Not diagonally dominant
    I = np.array([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
    print('Testing: \n', I,'\n')
    diagDominant(I)

    # Positive definite
    I = np.eye(4)
    print('Testing: \n', I,'\n')
    isPositiveDefinite(I)

    I = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    print('Testing: \n', I,'\n')
    isPositiveDefinite(I)

    # NOT positive definite
    I = np.array([[1, 2], [2, 1]])
    print('Testing: \n', I,'\n')
    isPositiveDefinite(I)

    print('\nJacobi/Gauss-Seidel ?')
    
    # Strictly diagonally dominant and not definite positive.
    I = np.array([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
    print('Testing: \n', I,'\n')
    useJacobiGaussSeidel(I)

    # Positive definite and diagonally dominant
    I = np.eye(4)
    print('Testing: \n', I,'\n')
    useJacobiGaussSeidel(I)