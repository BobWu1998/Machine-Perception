from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation

    # calculate transformation from world to camera (normalized)
    H = est_homography(Pw[:,:-1], Pc)
    H = H/H[-1,-1]

    H_prime = np.linalg.inv(K) @ H

    h1_prime, h2_prime, h3_prime = H_prime[:,0], H_prime[:,1], H_prime[:,2]

    # calculate (h1', h2', h1' cross h2')
    h1_h2_cross = np.zeros_like(H_prime)
    h1_h2_cross[:,0] = h1_prime
    h1_h2_cross[:,1] = h2_prime
    h1_h2_cross[:,2] = np.cross(h1_prime, h2_prime)

    [U, S, V] = np.linalg.svd(h1_h2_cross)
    R = U @ \
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, np.linalg.det(U@V)]]) @ V
    
    t = h3_prime/np.linalg.norm(h1_prime)

    R = np.transpose(R) # transform from Rwc to Rc
    t = -R @ t # tranform from twc to tcw

    ##### STUDENT CODE END #####

    return R, t
