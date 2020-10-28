import numpy as np

def LUdec(matrix):

    n = matrix.shape[0]
    U = matrix
    L = np.eye(n)

    for k in range(n-1):
        for l in range(k+1, n):
            m = U[l, k] / U[k, k];
            U[l, k] = 0;
            U[l, k + 1:] = U[l, k+1:n] - m * U[k, k+1:]
            L[l, k] = m;

    return L, U


L, U = LUdec(np.array([[3, 5], [6, 7]]))
print(L, U)