import numpy as np
from Triangulation import Triangulation as trgl

def e_estimation(F, K):
    """
    Input F: Fundamental Matrix
    Input K: Intrisic Matrix
    Output E: Essential Matrix, E = np.transpose(np.inv(K)) * F * K
    """
    K_t = np.transpose(K)
    E = np.dot(np.dot(K_t, F), K)
    u, sigma, v_t = np.linalg.svd(E)
    sigma[2] = 0
    #sigma = np.array([[sigma[0], 0, 0], [0, sigma[1], 0], [0,0,0]])
    sigma = np.array([[1, 0, 0], [0, 1, 0], [0,0,0]])
    E = np.dot(np.dot(u, sigma), v_t)
    return E

def cal_possible_RT(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, SIGMA, V_t = np.linalg.svd(E)
    T = U[:, 2]
    RTs = []
    RTs.append([ T, np.dot(np.dot(U, W), V_t)])
    RTs.append([-T, np.dot(np.dot(U, W), V_t)])
    RTs.append([ T, np.dot(np.dot(U, W.transpose()), V_t)])
    RTs.append([-T, np.dot(np.dot(U, W.transpose()), V_t)])
    for idx, [T, R] in enumerate(RTs):
        if np.linalg.det(R) < -0.999:
            RTs[idx][0] = -T
            RTs[idx][1] = -R
    return RTs

def cheirality_check(K, R, T, kp1s, kp2s):
    valid_count = 0
    #print("len of matches",len(matches))
    points3D = trgl.linear_triangulation(kp1s, 
                                         kp2s,
                                         K, R, T)
    matches = [0] * len(kp1s)
    for idx, [X, Y, Z] in enumerate(points3D):
        Z = R[2].dot((np.array([X, Y, Z])).reshape((3,1))) + T[2]
        if Z[0] > 0.0001:
                valid_count += 1
                matches[idx] = 1
    return valid_count * 1.0 / len(matches), matches

if __name__ == "__main__":
    cheirality_check(0,0,0,0,0)