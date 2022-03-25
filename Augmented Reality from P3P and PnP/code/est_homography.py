import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    H = []
    A = []
    # for all points, append them to A and reshape A for a 8x9 shape 2d array
    for i in range(X.shape[0]):
        ax = [ -X[i,0], -X[i,1], -1, 0, 0, 0, X[i,0]*Y[i,0], X[i,1]*Y[i,0], Y[i,0]]
        ay = [ 0, 0, 0, -X[i,0], -X[i,1], -1, X[i,0]*Y[i,1], X[i,1]*Y[i,1], Y[i,1]]
        A += ax
        A += ay
    A = np.array(A)
    A = np.reshape(A, (-1,9))

    # SVD and reshape for H
    [U, S, V] = np.linalg.svd(A)
    H = np.reshape(V[-1,:], (-1, 3))
    ##### STUDENT CODE END #####

    return H
