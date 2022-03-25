import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    N = pixels.shape[0]
    Pw = np.zeros(((pixels.shape[0], 3)))

    # calculate for the homography matrix
    extrinsic_matrix = np.zeros((3,3))
    extrinsic_matrix[:,0], extrinsic_matrix[:,1], extrinsic_matrix[:,2] = R_wc[:,0], R_wc[:,1], t_wc

    H = K @ extrinsic_matrix

    H_inv = np.linalg.inv(H)
    for i in range(N):
        pixel = np.hstack((pixels[i], 1))#np.linalg.norm(H[:,0]) #H[-1][-1] # need to multiply the lambda
    
        depth = -t_wc[-1]/(R_wc @ np.linalg.inv(K) @ pixel)[-1]#(R_wc.T @ t_wc)[-1] - (R_wc.T @ np.linalg.inv(K) @ pixel)[-1] 

        Pw[i] = depth * R_wc @ np.linalg.inv(K) @ pixel + t_wc#depth * (R_wc.T @ np.linalg.inv(K) @ pixel) - (R_wc.T @ t_wc)

    ##### STUDENT CODE END #####
    return Pw
