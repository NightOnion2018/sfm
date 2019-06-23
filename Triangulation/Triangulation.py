import numpy as np
import cv2
import scipy
from scipy.optimize import root

def linear_triangulation(kp1, kp2, K, R, T):
    """
    input kp1: list of key points
    input kp2: list of key points
    input matches: list of match DMatch
    output points3D: list of 3D points in word coordinate
    """
    points3D = []
    P = np.column_stack([np.eye(3), np.zeros((3,1))])
    P = np.dot(K, P)
    P_p = np.column_stack([R, T])
    P_p = np.dot(K, P_p)
    for p1, p2 in zip(kp1, kp2):
        u, v = p1
        u_p, v_p = p2
        A = np.ndarray([4, 4], dtype = np.float)
        A[0][0] = v * P[2][0] - P[1][0]
        A[0][1] = v * P[2][1] - P[1][1]
        A[0][2] = v * P[2][2] - P[1][2]
        A[0][3] = v * P[2][3] - P[1][3]

        A[1][0] = u * P[2][0] - P[0][0]
        A[1][1] = u * P[2][1] - P[0][1]
        A[1][2] = u * P[2][2] - P[0][2]
        A[1][3] = u * P[2][3] - P[0][3]

        A[2][0] = v_p * P_p[2][0] - P_p[1][0]
        A[2][1] = v_p * P_p[2][1] - P_p[1][1]
        A[2][2] = v_p * P_p[2][2] - P_p[1][2]
        A[2][3] = v_p * P_p[2][3] - P_p[1][3]

        A[3][0] = u_p * P_p[2][0] - P_p[0][0]
        A[3][1] = u_p * P_p[2][1] - P_p[0][1]
        A[3][2] = u_p * P_p[2][2] - P_p[0][2]
        A[3][3] = u_p * P_p[2][3] - P_p[0][3]
        U, SIGMA, V_t = np.linalg.svd(A)
        V = V_t.transpose()
        #print("V=",V)
        X_h = V[:,3]
        X = X_h[:3] / X_h[3]
        points3D.append(X)
    return points3D

def point_normalize(U):
    return U * 1.0 / U[-1]

def nonlinear_triangulation(kp1, kp2, K, R, T, matches):
    """
    Here We Suppose the 1st Camera Coordination is the refence coordination
    input kp1: list of key points of first  camera
    input kp2: list of key points of second camera
    input K  : intrisic matrix of second camera
    input R  : the rotation matrix of camera 2 related to camera 1
    input C  : the word coordinate of camera 1 related to camera 2
    input matches: a list of DMatch between kp1 and kp2
    output points3D: a list of 3D points after triangulation
    """
    def make_target(p1_2d, p2_2d):
        def func(x):
            point_3d = np.array([x[0], x[1], x[2], 1])
            RT   = np.column_stack([np.eye(3), np.zeros((3,1))])
            P    = np.dot(K, RT)
            RT_p = np.column_stack([R, T])
            P_p  = np.dot(K, RT_p)
            U    = np.dot(P  , np.array([x[0], x[1], x[2], 1]))
            U    = point_normalize(U)[: -1]
            U_p  = np.dot(P_p, np.array([x[0], x[1], x[2], 1]))
            U_p  = point_normalize(U_p)[: -1]
            # f
            f = [np.sum(np.square(p1_2d - U) + np.square(p2_2d - U_p)),
                 0,
                 0]
            # df ???
            def jacob(K, RC, p_2d, point_3d):
                f_x, f_y, c_x, c_y = K[0][0], K[1][1], K[0][2], K[1][2]
                r_0, r_1, r_2 = RC[0], RC[1], RC[2]
                u, v = p_2d[0], p_2d[1]
                A = f_x * r_0.dot(point_3d) + c_x * r_2.dot(point_3d)
                dA_dx = f_x * r_0[0] + c_x * r_2[0]
                dA_dy = f_x * r_0[1] + c_x * r_2[1]
                dA_dz = f_x * r_0[2] + c_x * r_2[2]
                B = r_2.dot(point_3d)
                dB_dx = r_2[0]
                dB_dy = r_2[1]
                dB_dz = r_2[2]
                C = f_y * r_1.dot(point_3d) + c_y * r_2.dot(point_3d)
                dC_dx = f_y * r_1[0] + c_y * r_2[0]
                dC_dy = f_y * r_1[1] + c_y * r_2[1]
                dC_dz = f_y * r_1[2] + c_y * r_2[2]
                df_dx = 2 * (u - A / B) * (-(dA_dx * B - A * dB_dx) / pow(B, 2)) + 2 * (v - C / B) * (-(dC_dx * B - C * dB_dx) / pow(B, 2))
                df_dy = 2 * (u - A / B) * (-(dA_dy * B - A * dB_dy) / pow(B, 2)) + 2 * (v - C / B) * (-(dC_dy * B - C * dB_dy) / pow(B, 2))
                df_dz = 2 * (u - A / B) * (-(dA_dz * B - A * dB_dz) / pow(B, 2)) + 2 * (v - C / B) * (-(dC_dz * B - C * dB_dz) / pow(B, 2))
                return [df_dx, df_dy, df_dz]
            # for 1st cam
            [df_dx_1, df_dy_1, df_dz_1] = jacob(K, RT, p1_2d, point_3d)
            # for 2nd cam
            [df_dx_2, df_dy_2, df_dz_2] = jacob(K, RT_p, p2_2d, point_3d)

            df_dx = df_dx_1 + df_dx_2
            df_dy = df_dy_1 + df_dy_2
            df_dz = df_dz_1 + df_dz_2

            df = np.array([[df_dx, df_dy, df_dz],
                           [0, 0, 0],
                           [0, 0, 0]
                          ])
            return f, df
        return func

    ret = []
    points3D_0 = linear_triangulation(kp1, kp2, K, R, T, matches)
    for match, point3D_0 in zip(matches, points3D_0):
        p1_idx = match.queryIdx
        p2_idx = match.trainIdx
        p_1, p_2 = kp1[p1_idx], kp2[p2_idx]
        u, v = p_1.pt
        u_p, v_p = p_2.pt
        func = make_target(np.array([u, v]),
                           np.array([u_p, v_p]))
        # do unit test here
        def unit_test_func(dx, dy, dz):
            return
            x0, y0, z0 = 10., 20, 3.0
            x, y, z = x0, y0, z0
            f_0, df_0 = func([x, y, z])
            print([x, y, z], f_0[0])
            x, y, z = x0 + dx, y0 + dy, z0 + dz
            f_1, df_1 = func([x, y, z])
            print([x, y, z], f_1[0])
            x, y, z = x0 + dx / 2, y0 + dy / 2, z0 + dz / 2
            f_2, df_2 = func([x, y, z])
            print([x, y, z], f_2[0])
            if dx > 0.000001:
                print("Estimated Dx:", df_2[0][0])
                print("True      Dx:", (f_1[0] - f_0[0]) / dx)
            if dy > 0.000001:
                print("Estimated Dy:", df_2[0][1])
                print("True      Dy:", (f_1[0] - f_0[0]) / dy)
            if dz > 0.000001:
                print("Estimated Dz:", df_2[0][2])
                print("True      Dz:", (f_1[0] - f_0[0]) / dz)
        #unit_test_func(0.001, 0, 0)
        #unit_test_func(0.0001, 0, 0)
        #unit_test_func(0.00001, 0, 0)
        #unit_test_func(0, 0.001, 0)
        #unit_test_func(0, 0.0001, 0)
        #unit_test_func(0, 0.00001, 0)
        #unit_test_func(0., 0, 0.01)
        #unit_test_func(0., 0, 0.001)
        #unit_test_func(0., 0, 0.0001)
        #print("-"*80)
        #print("Linear Optimized:", point3D_0)
        point3D_0[0] += 4
        point3D_0[1] += 0
        point3D_0[2] += 0
        #print("Initial Value(Distorted) for Non-Linear Optimizer:", point3D_0)
        sol = root(func, 
                   [point3D_0[0], point3D_0[1], point3D_0[2]], 
                   jac=True, method='lm')
        #print("Non-Linear Optimized:", sol.x)
        ret.append(sol.x)
    return ret

if __name__ == "__main__":
    X, Y, Z = 200, 200, 350
    kp1, kp2 = [], []
    K = np.array([[Z, 0, 0], [0, Z, 0], [0, 0, 1]])

    u, v = X, Y
    p = cv2.KeyPoint()
    p.pt = (u, v)
    kp1.append(p)
    
    matches = []
    match = cv2.DMatch()
    match.queryIdx = 0
    match.trainIdx = 0
    matches.append(match)

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # rotation using x-axis
    theta_x = 100
    cos = np.cos(np.pi * theta_x / 180)
    sin = np.sin(np.pi * theta_x / 180)
    R_x = np.array([[1,      0,    0],
                    [0,    cos, -sin], 
                    [0,    sin,  cos]])
    # rotation using y-axis
    theta_y = 200
    cos = np.cos(np.pi * theta_y / 180)
    sin = np.sin(np.pi * theta_y / 180)
    R_y = np.array([[ cos,   0,  sin],
                    [   0,   1,    0], 
                    [-sin,   0,  cos]])
    # rotation using z-axis
    theta_z = 45
    cos = np.cos(np.pi * theta_z / 180)
    sin = np.sin(np.pi * theta_z / 180)
    R_z = np.array([[cos, -sin, 0], 
                    [sin,  cos, 0],
                    [0,    0,   1]])
    R = np.dot(R, R_z)
    R = np.dot(R, R_y)
    R = np.dot(R, R_x)
    T = np.array([[00], [00], [1000]])

    pt_3d = np.dot(np.column_stack([R, T]), np.array([[X], [Y], [Z], [1]]))
    pt_h = np.dot(K, pt_3d)
    u, v = pt_h[0] / pt_h[2], pt_h[1] / pt_h[2]
    p = cv2.KeyPoint()
    p.pt = (u, v)
    kp2.append(p)

    print("Word 3D Coordinate",(X, Y, Z))
    print("Word 2D Coordinate",kp1[0].pt)
    print("Camera 3D Coordinate",list(pt_3d))
    print("Camera 2D Coordinate",kp2[0].pt)
    point3Ds = linear_triangulation(kp1, kp2, K, R, T, matches)
    for point in point3Ds:
        print("Linear Triangulation Recover 3D Coordinate:",point)
    point3Ds = nonlinear_triangulation(kp1, kp2, K, R, T, matches)
    for point in point3Ds:
        print("Non-Linear Triangulation Recover 3D Coordinate:",point)
