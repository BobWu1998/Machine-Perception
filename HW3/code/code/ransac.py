from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        '''
        OUTPUT:
            inliers: an array that consists of index of inliers (sample_indices + other_inlier_indices)
        '''
        def skew(x):
            return np.array([[0, -x[2], x[1]],
                             [x[2], 0, -x[0]],
                             [-x[1], x[0], 0]])

        # get essential matrix
        E = least_squares_estimation(X1[sample_indices, :], X2[sample_indices, :])

        # iterate for inliers
        test_x1, test_x2 = X1[test_indices, :], X2[test_indices, :]
        e3 = np.array([0, 0, 1])
        inliers = sample_indices # add sample_indices to inliers first

        # determine inliers and append the index
        for i in range(test_x1.shape[0]):
            x1, x2 = test_x1[i,:].T, test_x2[i,:].T # the element in X1[test_indices, :]

            d_2_1 = (x2.T @ E @ x1 )**2 / (np.linalg.norm(skew(e3) @ E @ x1))**2
            d_1_2 = (x1.T @ E.T @ x2 )**2 / (np.linalg.norm(skew(e3) @ E.T @ x2))**2
            residual = d_2_1 + d_1_2

            if residual < eps:
                inliers = np.append(inliers, test_indices[i])

        """ END YOUR CODE
        """
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers

    