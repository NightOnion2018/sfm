import cv2
import numpy as np
import random

def print_matrix(mat):
    r, c = mat.shape
    print('[', end = '')
    for ri in range(r):
        for ci in range(c):
            s = "%f" % mat[ri, ci]
            if s[0] != '-':
                s = ' ' + s
            s = s + ' '
            print(s, end='')
        if ri == r - 1:
            print(']')
        else:
            print('')
        print(' ', end='')

def genenrate_CUBE3D_Points(org, length, step):
    X0, Y0, Z0 = org
    points = []
    for x in range(X0, X0 + length, step):
        for y in range(Y0, Y0 + length, step):
            points.append([x, y, Z0])
    #        points.append([x, y, Z0 + length])
    for x in range(X0, X0+length, step):
        for z in range(Z0, Z0 + length, step):
            points.append([x, Y0, z])
    #        points.append([x, Y0 + length, z])
    for y in range(Y0, Y0 + length, step):
        for z in range(Z0, Z0 + length, step):
    #        points.append([X0, y, z])
            points.append([X0 + length, y, z])
    theta_z = 90
    cos = np.cos(np.pi * theta_z / 180)
    sin = np.sin(np.pi * theta_z / 180)
    R_z = np.array([[cos, -sin, 0], 
                    [sin,  cos, 0],
                    [0,    0,   1]])
    for idx, [x, y , z] in enumerate(points):
        pp = R_z.dot(np.array([x, y, z]).reshape(3,1))
        x1, y1, z1 = pp[0], pp[1], pp[2]
        points[idx] = [x1, y1, z1]
    return points

def generate_CUBE_kpt_match():
    points3D = genenrate_CUBE3D_Points((1000, 1000, 5000), 80000, 4000)
    matches = [cv2.DMatch() for _ in range(len(points3D))]
    for idx in range(len(matches)):
        matches[idx].queryIdx = idx
        matches[idx].trainIdx = idx
    mismatches = []
    f_x = 1080
    f_y = 1080
    c_x = 646
    c_y = 376
    cam1 = Util_Camera(f_x, f_y, c_x, c_y)
    cam2 = Util_Camera(f_x, f_y, c_x, c_y,
                        theta_x= 45,
                        theta_y= 00,
                        theta_z= 90,
                        t_x=5000,
                        t_y=5000,
                        t_z=5000)
    kpts1 = cam1.gen_keypoint(points3D)
    kpts2 = cam2.gen_keypoint(points3D)
    return kpts1, kpts2, matches, cam1, cam2, points3D, mismatches

def random_sample3D(nums = 10, rng = range(2000, 3000)):
    res = []
    rng = range(0, 1000)
    for _ in range(nums):
        x = random.sample(rng, 1)
        y = random.sample(rng, 1)
        z = random.sample(rng, 1)
        z = [2000]
        res.append([x[0] * 1.0, y[0] * 1.0, z[0] * 1.0])
    return res


class Util_Camera(object):
    def __init__(self, 
                f_x = 0,
                f_y = 0,
                c_x = 0,
                c_y = 0,
                theta_x = 0,
                theta_y = 0,
                theta_z = 0,
                t_x = 0,
                t_y = 0,
                t_z = 0):
        """
        input t_x/y/z: means the word's orign's coordinate in camera's coordinate
        """
        self.c_x = c_x
        self.c_y = c_y
        self.f_x = f_x
        self.f_y = f_y
        self.K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]) * 1.0
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 1.0
        # rotation using x-axis
        cos = np.cos(np.pi * theta_x / 180)
        sin = np.sin(np.pi * theta_x / 180)
        R_x = np.array([[1,      0,    0],
                        [0,    cos, -sin], 
                        [0,    sin,  cos]])
                        # rotation using y-axis
        cos = np.cos(np.pi * theta_y / 180)
        sin = np.sin(np.pi * theta_y / 180)
        R_y = np.array([[ cos,   0,  sin],
                        [   0,   1,    0], 
                        [-sin,   0,  cos]])
        # rotation using z-axis
        cos = np.cos(np.pi * theta_z / 180)
        sin = np.sin(np.pi * theta_z / 180)
        R_z = np.array([[cos, -sin, 0], 
                        [sin,  cos, 0],
                        [0,    0,   1]])
        R = np.dot(R, R_z)
        R = np.dot(R, R_y)
        R = np.dot(R, R_x)
        self.R = R
        self.T = np.array([t_x, t_y, t_z]) * 1.0

    def gen_keypoint(self, points3D): 
        res = [] 
        points2D, mask = self.project(points3D)
        for point2D in points2D:
            kpt = cv2.KeyPoint()
            kpt.pt = point2D
            res.append(kpt)
        return [res, mask]

    def project(self, points3D):
        res = []
        R, T, K = self.R, self.T, self.K
        RT = np.column_stack([R, T])
        mask = [1] * len(points3D)
        for idx, point3D in enumerate(points3D):
            X_h = np.array([point3D[0], point3D[1], point3D[2], 1]).reshape(4, 1)
            x = RT.dot(X_h)
            u, v, w = np.dot(K, x)
            u, v = u / w, v / w
            if w <= 0:
                mask[idx] = 0
            res.append((np.int32(u), np.int32(v)))
            #res.append((np.round(u), np.round(v)))
            #res.append((u, v))
        return [res, mask]


class Two_View_System(object):
    def __init__(self, cam1, cam2):
        self.cam1 = cam1
        self.cam2 = cam2
        t_x, t_y, t_z = self.cam2.T[0], self.cam2.T[1], self.cam2.T[2]
        temp_T = np.array([[0, -t_z, t_y], [t_z, 0, -t_x], [-t_y, t_x, 0]])
        self.E = np.cross(self.cam2.T, self.cam2.R)
        #print(self.E)
        self.E = np.dot(temp_T, self.cam2.R)
        #print(self.E)
        self.F = np.dot(np.dot(np.transpose(np.linalg.inv(cam2.K)), self.E), np.linalg.inv(cam1.K))
        self.points3D = None
        self.kpts1 = None
        self.kpts2 = None
    
    def set_points3D(self, points3D):
        self.points3D = points3D
        self.kpts1 = self.cam1.gen_keypoint(points3D)
        self.kpts2 = self.cam2.gen_keypoint(points3D)
    
    def validate_E(self):
        cam2 = self.cam2
        R, T = cam2.R, cam2.T
        RT = np.column_stack([R, T])
        nonzero_cnt = 0
        for point3D in self.points3D:
            X_h = np.array([point3D[0], point3D[1], point3D[2], 1]).reshape(4, 1)
            X_2 = RT.dot(X_h) * 1.0
            X_1 = X_h[:-1] * 1.0
            res = np.dot(X_2.reshape(1,3), np.dot(self.E, X_1))
            if res > 0.000001:
                nonzero_cnt += 1
        print("Number of Nonzero res in E Validation:", nonzero_cnt)

    def validate_F(self):
        nonzero_cnt = 0
        for kpt1, ktp2 in zip(self.kpts1, self.kpts2):
            x1, y1 = kpt1.pt
            x2, y2 = ktp2.pt
            X_p = np.array([x2, y2, 1])
            X = np.array([x1, y1, 1])
            res = np.dot(np.dot(X_p, self.F), X.reshape(3,1))
            if res > 0.000001:
                nonzero_cnt += 1
                #print("Non-Zero",res)
        print("Number of Nonzero res in F Validation:", nonzero_cnt)


def gen_kpt_matches(num_kpts = 200, inliar_percent = 0.8):
    inliar_percent = inliar_percent
    points3D = random_sample3D(num_kpts, rng = range(1000, 10000))
    matches = [cv2.DMatch() for _ in range(len(points3D))]
    inliars = random.sample(range(len(matches)), 
                            int(inliar_percent * len(matches)))
    for idx, match in enumerate(matches):
        match.queryIdx = idx
        match.trainIdx = random.choice(range(len(matches)))
    for idx in inliars:
        matches[idx].queryIdx = idx
        matches[idx].trainIdx = idx
    mismatches = []
    for idx in range(len(matches)):
        if matches[idx].trainIdx != idx:
            mismatches.append(idx)
    f_x = 1080 * 1.5
    f_y = 1080 * 1.5
    c_x = 0
    c_y = 0
    cam1 = Util_Camera(f_x, f_y, c_x, c_y)
    cam2 = Util_Camera(f_x, f_y, c_x, c_y,
                        theta_x= 45,
                        theta_y= 30,
                        theta_z= 90,
                        t_x=100,
                        t_y=200,
                        t_z=30)
    kpts1 = cam1.gen_keypoint(points3D)
    kpts2 = cam2.gen_keypoint(points3D)
    return kpts1, kpts2, matches, cam1, cam2, points3D, mismatches


if __name__ == "__main__":
    inliar_percent = 0.80
    points3D = random_sample3D(200)
    matches = [cv2.DMatch()] * len(points3D)
    inliars = random.sample(range(len(matches)), 
                            int(inliar_percent * len(matches)))
    for idx, match in enumerate(matches):
        match.queryIdx = idx
        match.trainIdx = random.choice(range(len(matches)))
    for idx in inliars:
        matches[idx].queryIdx = idx
        matches[idx].trainIdx = idx
    f_x = 200
    f_y = 200
    c_x = 0
    c_y = 0
    cam1 = Util_Camera(f_x, f_y, c_x, c_y)
    cam2 = Util_Camera(f_x, f_y, c_x, c_y,
                        theta_x= 0,
                        theta_y= 0,
                        theta_z= 90,
                        t_x= 150,
                        t_y=0,
                        t_z=0)
    kpts1 = cam1.gen_keypoint(points3D)
    kpts2 = cam2.gen_keypoint(points3D)
