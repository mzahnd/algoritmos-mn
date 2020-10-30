import numpy as np

def isFullRank(matrix):
    """Given a numpy (n,m) array prints on screen if it's full rank or not.
    
    As a plus, it also prints if it's singular.

    Arguments:
        matrix: (n,m) numpy array .

    Returns:
        Nothing

    Raises:
        RuntimeError: when matrix dimension is not 2.
    """

    # '\u2713' = ✓ (unicode tick character)

    print('Analysing matrix: \n', matrix)
    
    matrixNumDim = np.ndim(matrix)
    if (matrixNumDim != 2):
        raise RuntimeError("I'm not prepared for this. </3")

    matrixShape = np.shape(matrix)
    matrixRange = np.linalg.matrix_rank(matrix)

    print()

    if min(matrixShape) == matrixRange:
        print(u"[\u2713] Matrix is full rank.")
        if matrixShape[0] == matrixShape[1]:
            print(u"[\u2713] This matrix is squeared, so it's nonsingular.")
        else:
            print("[X] This matrix is not squeared, so it's singular.")
    else:
        print("[X] Matrix is NOT full rank.")

    print()


if __name__ == "__main__":
    I = np.eye(4) # Full rank matrix
    isFullRank(I)

    I[-1, -1] = 0 # Make it rank deficient
    isFullRank(I)
    
    I = np.array([[1,2,3],[4,5,6]]) # Full rank non-squared matrix
    isFullRank(I)