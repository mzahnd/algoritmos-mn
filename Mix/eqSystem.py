#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Which alg can you use to solve a system of equations, and least squares.

Ax = b

With A matrix; x and b vectors
"""

import numpy as np

import scipy.linalg as sclin
import scipy

import sys

def pureDiagonal(matrix, debug=False):
    """Matrix is pure diagonal"""

    if (np.ndim(matrix) != 2):
        raise RuntimeError("I'm not prepared for this. </3")

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing matrix: \n', matrix)
        print()

    if isSquared(matrix, debug=True) and \
            np.allclose(matrix, np.diag(np.diag(matrix))):
        print(u'[\u2713] Matrix is pure diagonal.')
        return True
    else:
        print('[X] Matrix is not pure diagonal.')
        return False


def upperTriangular(matrix, debug=False):
    """Matrix is an upper one"""

    if (np.ndim(matrix) != 2):
        raise RuntimeError("I'm not prepared for this. </3")

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing matrix: \n', matrix)
        print()

    if np.allclose(matrix, np.triu(matrix)):
        print(u'[\u2713] Matrix is upper triangular.')
        return True
    else:
        print('[X] Matrix is not upper triangular.')
        return False


def lowerTriangular(matrix, debug=False):
    """Matrix is an upper one"""

    if (np.ndim(matrix) != 2):
        raise RuntimeError("I'm not prepared for this. </3")

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing matrix: \n', matrix)
        print()

    if np.allclose(matrix, np.tril(matrix)):
        print(u'[\u2713] Matrix is lower triangular.')
        return True
    else:
        print('[X] Matrix is not lower triangular.')
        return False
    

def checkLU(matrix, debug=False):
    if (np.ndim(matrix) != 2):
        raise RuntimeError("I'm not prepared for this. </3")

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing LU for matrix: \n', matrix)
        print()

    if not isSquared(matrix, debug):
        print('[X] Matrix can not be decomposed using LU.')
        return False

    matrixShape = np.shape(matrix)

    try:
        pMatrix, lMatrix, uMatrix = sclin.lu(matrix, permute_l=False, 
                                        check_finite=True)
        if (np.allclose(matrix - np.matmul(np.matmul(pMatrix, lMatrix),
                                uMatrix), np.zeros(matrixShape))):
            print(u'[\u2713] Matrix is decomposable using LU.')

            if (np.allclose(pMatrix, np.zeros(matrixShape))):
                print('\tNo P matrix is requiered for decomposing.')
                
            else:
                print(u'\t[!!!] P matrix is requiered for decomposing.')

            return True
        else:
            print('[X] Matrix is not decomposable using LU.')
            return False
    except:
        print('[X] Matrix is not decomposable using LU. ', 
        '(Catched exception: )', sys.exc_info()[0])
        return False


def isSymmetric(matrix, debug=False):

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing symmetry of the matrix: \n', matrix)
        print()

    if isSquared(matrix, debug) and np.allclose(matrix, matrix.T):
        print(u'[\u2713] Matrix is symmetric.')
        return True
    else:
        print('[X] Matrix is not symmetric.')
        return False
        

def isPositiveDefinite(matrix, debug=False):
    """
    Taken from: https://stackoverflow.com/a/44287862
    """

    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing if matrix is positive definite: \n', matrix)
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

def isSquared(matrix, debug=True):
    matrixShape = np.shape(matrix)
    if (matrixShape[0] != matrixShape[1]):
        if debug:
            print('[X] Matrix is not squared')
        return False
    else:
        if debug:
            print(u'[\u2713] Matrix is squared.')
        return True

def checkCholesky(matrix, debug=False):
    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing Cholesky for matrix: \n', matrix)
        print()

    # isPositiveDefinite verifies cholesky :)
    if isPositiveDefinite(matrix, debug=False):
        print(u'[\u2713] Matrix is decomposable using Cholesky.')
        return True
    else:
        print('[X] Matrix is not decomposable using Cholesky.')
        return False


def checkQR(matrix, debug=False):
    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing QR for matrix: \n', matrix)
        print()

    try:
        qMatrix, rMatrix = sclin.qr(matrix)
        if np.allclose(matrix, np.matmul(qMatrix, rMatrix)):
            print(u'[\u2713] Matrix is decomposable using QR.')
            return True
        else:
            print('[X] Matrix is not decomposable using QR.')
            return False
    except sclin.LinAlgError:
        print('[X] Matrix is not decomposable using QR.')
        return False


def checkSVD(matrix, debug=False):
    # '\u2713' = ✓ (unicode tick character)
    if debug:
        print('Analysing SVD for matrix: \n', matrix)
        print()

    try:
        uMatrix, sMatrix, vhMatrix = sclin.svd(matrix, full_matrices=True,
                                                check_finite=True)
        
        # Diagonalize sigma to perform multiplication
        sigma = np.zeros(( np.shape(uMatrix)[1], np.shape(vhMatrix)[0] ))
    
        for rowEtCol in range(0, len(sMatrix), 1):
            sigma[rowEtCol][rowEtCol] = sMatrix[rowEtCol]

        if debug:
            print('U = \n', uMatrix)
            print('S = \n', sigma)
            print('V = \n', vhMatrix)

        svdProduct = np.matmul(uMatrix, np.matmul(sigma, vhMatrix))

        if debug:
            print('U*U.T = \n', np.matmul(uMatrix, uMatrix.T))
            print('U*S*V = \n', svdProduct)

        if np.allclose(matrix, svdProduct):
            print(u'[\u2713] Matrix is decomposable using SVD.')
            return True
        else:
            print('[X] Matrix is not decomposable using SVD.')
            return False
    except sclin.LinAlgError:
        print('[X] Matrix is not decomposable using SVD.')
        return False

# print(u'[\u2713] Matrix is.')
# print('[X] Matrix is not.')

def solveLinealEq(matrix, debug=False):
    print('Analysing matrix: \n', matrix, '\n')

    if pureDiagonal(matrix, debug):
        print('--->', end=' ')
        print("You're a lucky one. Divide each b_i by a_ii and get your x_i", 
            end=' ')
        print('<---')
    elif upperTriangular(matrix, debug):
        print('--->', end=' ')
        print('This system must be solved using backwards substitution.', end=' ')
        print('<---')
    elif lowerTriangular(matrix, debug):
        print('--->', end=' ')
        print('This system must be solved using forward substitution.', end=' ')
        print('<---')
    else:
        if checkCholesky(matrix, debug):
            print('--->', end=' ')
            print('This system is solvable using Cholesky.', end=' ')
            print('<---')
        #elif checkLU(matrix, debug):
        if checkLU(matrix, debug):
            print('--->', end=' ')
            print('This system is solvable using LU decomposition.', end=' ')
            print('<---')
        #else:
        if checkQR(matrix, debug):
            print('--->', end=' ')
            print('This system is solvable using QR decomposition.', end=' ')
            print('<---')

        matrix = np.array([[12, 4], [3, 2], [6, 2]])
        if checkSVD(matrix, debug):
            print('--->', end=' ')
            print('This system is solvable using SVD decomposition.', end=' ')
            print('<---')
    print()


def tests():
    print('='*10, 'Matrix tests', '='*10)

    A = np.eye(5)
    solveLinealEq(A, debug=False)

    A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    solveLinealEq(A, debug=False)

    A = np.array([[12, 4], [3, 2], [6, 2]])
    solveLinealEq(A, debug=False)

if __name__ == "__main__":
    tests()