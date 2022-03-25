import numpy as np

def least_squares_estimation(X1, X2):
  """ YOUR CODE HERE
  """

  N = X1.shape[0]
  A = np.zeros((N,9))

  # get A matrix
  A[:, :3] = X1[:,0].reshape((-1,1)) * X2
  A[:, 3:6] = X1[:,1].reshape((-1,1)) * X2
  A[:, 6:] = X1[:,2].reshape((-1,1)) * X2

  # calculate and format E matrix
  [U, S, Vh] = np.linalg.svd(A)
  E_stacked = Vh.T[:, -1] # last col of V, or E_stacked = Vh[-1, :] # last row of Vh
  E = E_stacked.reshape((3,3)).T # need to transpose because 1st 3 elem is the 1st col

  # project E for S = diag(1, 1, 0)
  [U, S, Vh] = np.linalg.svd(E)
  E = U @ np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]) @ Vh

  """ END YOUR CODE
  """
  return E
