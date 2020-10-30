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
    #First we check if the matrix is a lower triangle
    if np.allclose(A, np.tril(A)):
        n = len(b)
        x = np.zeros((n, 1))

        #we apply the forward formula for every element of x 
        for k in range(0, n):
            tempSum = []
            for number in range(0, k):
                tempSum.append(-1 * A[k][number] * x[number])
            tempSum = sum(tempSum)
            x[k] = (b[k] + tempSum) / A[k][k]

        return x
        
    else:
        raise RuntimeError("Matrix A is not lower triangular.")


def test_linealSolverForward():
    """Test linealSolverForward.

    This function will stop the code execution when it finds an error.

    Returns:
        None
    """

    # ====== Forward and backwards substitution ======
    # Each tuple has tree elements, by index:
    # 0: matrix A (lower triangular)
    # 1: matrix b (System solution)
    # 2: matrix c (Expected answer)
    #
    # System's solutions were verified using WolframAlpha.

    lowTriangMatrices = (
        (np.array([[8, 0, 0], [2, 3, 0], [4, 7, 1]]),
         np.array([[8], [5], [0]]),
         np.array([[1], [1], [-11]])),
        (np.array([[8, 0, 0], [2, 3, 0], [4, 7, 1]]),
         np.array([[5], [1], [-8]]),
         np.array([[5 / 8], [-1 / 12], [-119 / 12]])),
        (np.array([[5, 0, 0], [76, 63, 0], [47, 77, 31]]),
         np.array([[69], [10], [4]]),
         np.array([[69 / 5], [-742 / 45], [28127 / 1395]])),
        (np.array([[44, 0, 0, 0], [17, 10, 0, 0],
                   [65, 43, 49, 0], [75, 5, 81, 76]]),
         np.array([[66], [74], [8], [22]]),
         np.array([[3 / 2], [97 / 20], [-5961 / 980], [9747 / 1960]])),
    )

    print("="*60, "\nTesting linearSolverForward() function.")
    for system in lowTriangMatrices:
        try:
            print('-'*50)
            print("Testing system:\n"
                  + f"A =\n{system[0]}\n"
                  + f"b =\n{system[1]}\n"
                  + f"Expected answer =\n{system[2]}")

            ans = linealSolverForward(system[0], system[1])
            comparison = np.allclose(ans, system[2])
            assert \
                    comparison, \
                    "The last tested system did not pass the test."
            print("-> Pass <-")
        except RuntimeError:
            assert False, "The last tested system did not pass the test."

if __name__ == "__main__":
    print("Executed as stand-alone script. Running test function.\n")
    test_linealSolverForward()