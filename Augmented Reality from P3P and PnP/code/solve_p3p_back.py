from re import S
import numpy as np
from sympy import Q

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####
    


    P1, P2, P3 = Pw[1], Pw[2], Pw[3]
    f = K[0,0]
    p1, p2, p3 = Pc[1], Pc[2], Pc[3]
    print("p1, p2, p3:", p1, p2, p3)



    a, b, c = np.linalg.norm(P2-P3), np.linalg.norm(P1-P3), np.linalg.norm(P1-P2)
    # print("a, b, c:", a, b, c)
    # a, b, c = np.linalg.norm(p2-p3), np.linalg.norm(p1-p3), np.linalg.norm(p1-p2)
    # print("a, b, c:", a, b, c)

    j1 = 1/np.sqrt(p1[0]**2 + p1[1]**2 + f**2)*(np.append(p1,f))
    j2 = 1/np.sqrt(p2[0]**2 + p2[1]**2 + f**2)*(np.append(p2,f))
    j3 = 1/np.sqrt(p3[0]**2 + p3[1]**2 + f**2)*(np.append(p3,f)) # unit vector in 1 by 3

    print("j1, j2, j3:", j1, j2, j3)

    cos_alpha, cos_beta, cos_gamma = j2 @ j3, j1 @ j3, j1 @ j2

    A4 = ((a**2-c**2)/b**2-1)**2 - (4*c**2/b**2 * cos_alpha**2)

    A3 = 4*( (a**2-c**2)/b**2 * (1-(a**2-c**2)/b**2) * cos_beta \
             - (1 - (a**2+c**2)/b**2) * cos_alpha * cos_gamma \
             + 2 * c**2/b**2 * cos_alpha**2 * cos_beta )

    A2 = 2*( ((a**2-c**2)/b**2)**2 - 1 + 2 * ((a**2-c**2)/b**2)**2 * cos_beta**2 \
             + 2*(b**2-c**2)/b**2 * cos_alpha**2 \
             - 4*(a**2+c**2)/b**2 * cos_alpha * cos_beta * cos_gamma \
             + 2*(b**2-a**2)/b**2 * cos_gamma**2 )

    A1 = 4*( -(a**2-c**2)/b**2 * (1 + (a**2-c**2)/b**2) * cos_beta  \
             + 2*a**2/b**2 * cos_gamma**2 * cos_beta \
             - (1 - (a**2+c**2)/b**2) * cos_alpha * cos_gamma )
    
    A0 = (1 + (a**2-c**2)/b**2)**2 - 4*a**2/b**2 * cos_gamma**2

    coeff = [A4, A3, A2, A1, A0]
    print('coeff:',coeff)
    roots = np.roots(coeff)
    print('roots:',roots)

    # get rid of all imaginary roots
    real_roots = []
    for v in roots:
        if not np.iscomplex(v):
            real_roots.append(np.real(v))

    print("real_roots:", real_roots)
    # real_roots[1] = real_roots[0]

    # Select the roots that make all distance positive
    for v in real_roots:
        u = ( (-1 + (a**2-c**2)/b**2 ) * v**2 - 2*(a**2-c**2)/b**2 * cos_beta * v + 1 + (a**2-c**2)/b**2 ) \
            / (2*(cos_gamma - v * cos_alpha))
        print("u,v:\n", u,v)
        s1_square = c**2 / (1 + u**2 - 2*u*cos_gamma)
        if s1_square > 0 and u > 0 and v > 0:
            s1 = np.sqrt(s1_square)
            s2 = u*s1
            s3 = v*s1

    print('s1, s2, s3:', s1, s2, s3)
    Pc_3d = np.zeros_like(Pw[1:4],dtype=float)
    Pc_3d[0] = s1 * j1
    Pc_3d[1] = s2 * j2
    Pc_3d[2] = s3 * j3
    # print("Pc_3d:", Pc_3d)
    # print("Pw[1:4]:", Pw[1:4])

    ##### STUDENT CODE END #####
    R,t = Procrustes(Pc_3d, Pw[1:4])

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    # Y is A, X is B
    X, Y = np.transpose(X), np.transpose(Y)

    Y_bar = np.average(Y,axis=1).reshape((-1,1))
    Y = Y - Y_bar

    X_bar = np.average(X,axis=1).reshape((-1,1))
    X = X - X_bar

    [U, S, Vt] = np.linalg.svd(X @ Y.T) #(X @ Y.T)

    R = Vt.T @ \
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, np.linalg.det(Vt.T@U.T)]]) @ U.T

    t = Y_bar - R @ X_bar
    t = t.flatten()
    ##### STUDENT CODE END #####
    return R, t
