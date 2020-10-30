#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


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

def test_linealSolverBackwards():
    """Test linealSolverBackwards.

    This function will stop the code execution when it finds an error.

    Returns:
        None
    """
    # ====== Backwards substitution ======
    # Each tuple has tree elements, by index:
    # 0: matrix A (lower triangular)
    # 1: matrix b (System solution)
    # 2: matrix c (Expected answer)
    #
    # System's solutions were verified using WolframAlpha.


    upperTriangMatrices = (
        (np.array([[12, 5, 6], [0, 1, -4], [0, 0, 9]]),
         np.array([[-37], [2], [9]]),
         np.array([[-73 / 12], [6], [1]])),
        (np.array([[-8, 35, 65], [0, 40, -64], [0, 0, 64]]),
         np.array([[20], [12], [43]]),
         np.array([[4595 / 512], [11 / 8], [43 / 64]])),
        (np.array([[-72, 8, 64, 91], [0, 70, -90, -27],
                   [0, 0, -39, -22], [0, 0, 0, -22] ]),
         np.array([ [-40], [18], [-47], [-2] ]),
         np.array([ [3499 / 1848], [3555 / 2002], [15 / 13], [1 / 11] ]))
    )

    print("="*60, "\nTesting linearSolverBackwards() function.")
    for system in upperTriangMatrices:
        try:
            print('-'*50)
            print("Testing system:\n"
                  + f"A =\n{system[0]}\n"
                  + f"b =\n{system[1]}\n"
                  + f"Expected answer =\n{system[2]}")

            ans = linealSolverBackwards(system[0], system[1])
            comparison = np.allclose(ans, system[2])
            assert \
                    comparison, \
                    "The last tested system did not pass the test."

            print("-> Pass <-")
        except RuntimeError:
            assert False, "The last tested system did not pass the test."

if __name__ == "__main__":
    print("Executed as stand-alone script. Running test function.\n")
    test_linealSolverBackwards()