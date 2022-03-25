# from matplotlib.pyplot import axis
import numpy as np
# Y = np.arange(9).reshape((-1,3))
# Y = np.vstack((Y, np.array([[2,2,3],
#                            [2,3,4]])))
# Y = Y.T
# X = np.arange(9).reshape((-1,3))
# X = np.vstack((X, np.array([[0,0,3],
#                            [1,2,4]])))
# X = X.T
# print(Y)

# """
#     Solve Procrustes: Y = RX + t

#     Input:
#         X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
#         Y: Nx3 numpy array of N points in world coordinate
#     Returns:
#         R: 3x3 numpy array describing camera orientation in the world (R_wc)
#         t: (3,) numpy array describing camera translation in the world (t_wc)

#     """


# Test of Procrustes
# Y_bar = np.average(Y,axis=1).reshape((-1,1))
# print(Y_bar)
# Y = Y - Y_bar
# print(Y)

# X_bar = np.average(X,axis=1).reshape((-1,1))
# X = X - X_bar

# print(X @ Y.T)

# [U, S, Vt] = np.linalg.svd(X @ Y.T)

# R = Vt.T @ \
#         np.array([[1, 0, 0],
#                   [0, 1, 0],
#                   [0, 0, np.linalg.det(Vt.T@U.T)]]) @ U.T

# T = Y_bar - R @ X_bar
# print(T)

# p1 = np.array([1,2])
# # print(np.vstack((np.transpose(p1), np.array([1]))))
# print((np.append(p1,1)).reshape((-1,1)))



# Test of P3P
p1 = np.array([1,1])
f = 1
j1 = 1/np.sqrt((p1[0]**2, p1[1]**2, f**2)*(np.append(p1,f)))
print(j1)