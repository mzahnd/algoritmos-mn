#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Descomposición en valores singulares

Utiliza numpy para el cálculo de autovalores y autovectores
"""

import numpy as np
import random as rnd

def svd(matrix):
    if (type(matrix) is not np.ndarray):
        raise RuntimeError("Expected numpy matrix.")

    print('A: ', matrix, '\n')

    B = np.matmul(matrix.T, matrix)
    
    # Debug
    print (B,'\n')

    Dt, V = np.linalg.eig(B)

    D = np.zeros(np.shape(V))

    # Debug
    print('Dt shape:', np.shape(Dt))
    print('D shape:', np.shape(D), '\n')

    i = 0
    j = 0
    for nEigv in range(np.shape(Dt)[0]):
        # Debug
        print('eigv: ', Dt[nEigv])
        
        D[i, j] = Dt[nEigv]
        i += 1
        j += 1

    # Debug
    # This should print B matrix
    print('V: \n', V, '\n')
    print('D: \n', D, '\n')
    print(np.matmul( np.matmul(V, D), V.T), '\n')

    # Sort 'em

    sigmas = np.sqrt(Dt)

    # Debug
    print('sigmas: \n', sigmas, '\n')

    # U's
    maxSize = max(np.shape(matrix))
    U = np.zeros((maxSize, maxSize))
    # Debug
    print('U: \n', U, '\n')

    # Esta línea es a propósito para que sea exactamente igual al video
    V *= -1

    for pos in range(np.shape(sigmas)[0]):
        U[:,pos] = np.matmul(matrix, V[:,pos]) / sigmas[pos]

    # Debug
    print('U: \n', U, '\n')

    # ------------- Up to here seems that everything's fine --------------
    # Here is where the trouble begins

    # currentCol = 1
    # while currentCol < maxSize:
    #     # Modify for multi cols
    #     if (np.allclose(U[:,-currentCol], np.zeros((maxSize, 1)))):
    #         rnd.seed()
    #         for row in range(maxSize):
    #             U[row,-currentCol] = rnd.random()
            
    #         # TODO WTF ? .
    #         print('U: \n', U, '\n')
    #         #U = _matrixOrthonormalization(U, column=-currentCol)
    #         print('Before: ', U[:,-1])
    #         U[:,-1] = U[:,-1] - (U[:,-1].T * U[:,0])*U[:,0] - (U[:,-1].T * U[:,1])*U[:,1]
            
    #         print('Before norm: ', U[:,-1])
    #         U[:,-1] = U[:,-1] / np.linalg.norm(U[:,-1])
    #         print('After norm: ', U[:,-1])

    #     currentCol += 1

    U[:,-1] = [0.67687, 0.82416, 0.57667]
    U[:,-1] = U[:,-1] - (U[:,-1].T * U[:,0])*U[:,0]  - (U[:,-1].T * U[:,1])*U[:,1]
    U[:,-1] = U[:,-1]/np.linalg.norm(U[:,-1])

    # Debug
    print('U: \n', U, '\n')
    print('U.T*U: \n', U.T * U, '\n')

    # Ortonormalize cols
    

def _matrixOrthonormalization(matrix, column=0):
    """
    matrix: Matrix to perform algorithm over
    column: Column to orthonormalize. Set to 0 to perform over the whole
            matrix.
    """
    # TODO
    pass

def test():
    matrix = np.array([[12, 4], [3, 2], [6, 2]])
    svd(matrix)


if __name__ == "__main__":
    print("Executed as stand-alone script. Running test function.\n")
    test()
