import collections
import os
import sys
import numpy as np
import random
import cv2

sys.path.append("/Users/patrickji/workspace/visual_code/SctructureFromMotion")
import util

class CameraPose(object):
    def __init__(self, cam, cam_id):
        self.cam = cam
        self.kp_landmark = {}
        self.cam_id = cam_id
        self.name = "gen_images/gen_"+str(self.cam_id)+".jpg"

    def projectLandmarks(self, landmarks):
        kpts1, mask = self.cam.gen_keypoint(landmarks)
        matches = [0] * len(landmarks)
        #if sum(mask) != len(mask):
        #    print("Some 3D landmarks in "+str(self.cam_id)+" is not in front of camera")
        for idx_landmark, landmark in enumerate(landmarks):
            u, v = kpts1[idx_landmark].pt
            u = int(u)
            v = int(v)
            if u >= 0 and v >= 0 and u < 2 * self.cam.c_x and v < 2 * self.cam.c_y and mask[idx_landmark]:
            #if u >= 0 and v >= 0 and u < 2 * self.cam.c_x and v < 2 * self.cam.c_y:
                self.kp_landmark[(self.cam_id, idx_landmark)] = (u, v)
                matches[idx_landmark] = 1
        return matches

    def gen_landmarks(self, num_landmarks):
        c_x = int(self.cam.c_x)
        c_y = int(self.cam.c_y)
        fl = int(self.cam.f_x)
        landmarks = []
        for i in range(num_landmarks):
            u = random.randint(1, 2 * c_x - 1)
            v = random.randint(1, 2 * c_y - 1)
            w = random.randint(fl + 2, fl * 15)
            x, y, z = (u - c_x) * w / fl, (v - c_y) * w / fl, w
            landmarks.append([x, y, z])
        return landmarks

class Landmark(object):
    def __init__(self, num_landmarks, cam_pos):
        self.num_landmarks = num_landmarks
        self.landmarks = cam_pos.gen_landmarks(num_landmarks)
        self.landmark_counter = collections.Counter()

    def updateCounter(self, mathes):
        for idx, v in enumerate(mathes):
            if v: self.landmark_counter[v] += 1
    
    def get_Landmarks(self):
        return self.landmarks


def output(outstream, campos_list, landmark):
    landmarks = landmark.get_Landmarks()
    FOCAL_LEN = campos_list[0].cam.f_x
    c_x = campos_list[0].cam.c_x
    c_y = campos_list[0].cam.c_y
    outstream.write(str(FOCAL_LEN)+" "+str(c_x)+ " " + str(c_y)+" # F, cx, cy\n")
    outstream.write(str(len(campos_list)) + " # number of image poses\n")
    outstream.write(str(len(landmarks)) + " # number of landmards\n")
    for i in range(len(landmarks)):
        outstream.write(str(landmarks[i][0]) + " ")
        outstream.write(str(landmarks[i][1]) + " ")
        outstream.write(str(landmarks[i][2]) + " ")
        outstream.write("# landmark["+str(i)+']\n')
    landmark_vertex_offset = len(campos_list)
    for cam_idx, campos in enumerate(campos_list):
        outstream.write(str(len(campos.kp_landmark))+"\n")
        for (im_idx, landmark_idx), (u, v) in campos.kp_landmark.items():
            outstream.write(str(im_idx)+' '+str(landmark_idx + landmark_vertex_offset) + " ")
            outstream.write(str(u) + " " + str(v) + " \n")

    for im_idx, im in enumerate(campos_list):
        outstream.write(im.name+"\n")

    for cam_idx, cam_pos in enumerate(campos_list):
        cam = cam_pos.cam
        K = cam.K
        RT = np.column_stack([cam.R, cam.T])
        P = np.dot(K, RT)
        P = RT
        for i in range(3):
            for j in range(4):
                outstream.write(str(P[i][j])+" ")
            outstream.write('\n')
        outstream.write('\n')


def cv2_validation(campos_list):
    campos0 = campos_list[0]
    campos1 = campos_list[1]
    landmark_map = collections.defaultdict(list)
    for key, val in campos0.kp_landmark.items():
        cam_id, lnd_id = key
        u, v = val
        landmark_map[lnd_id].append(tuple(val))
    for key, val in campos1.kp_landmark.items():
        cam_id, lnd_id = key
        u, v = val
        landmark_map[lnd_id].append(tuple(val))

    src = []
    dst = []
    for lnd_id, vals in landmark_map.items():
        if len(vals) == 2:
            src.append(vals[0])
            dst.append(vals[1])
    
    src = np.float32([pt for pt in src])
    dst = np.float32([pt for pt in dst])
    E, mask = cv2.findEssentialMat(src, dst, 
                                   cameraMatrix = campos0.cam.K,
                                   method=cv2.FM_RANSAC, 
                                   prob = 0.99, 
                                   threshold = 1.0)
    _, local_R, local_T, mask = cv2.recoverPose(E, src, dst, 
                                                cameraMatrix = campos0.cam.K, 
                                                mask = None)
    print("local_R = \n",local_R)
    print("local_T = \n",local_T)


if __name__ == "__main__":
    num_cams = 10
    num_landmarks = 1000
    f_x = 1077
    f_y = 1077
    c_x = 2736
    c_y = 1824
    campos_list = []
    os.system("rm -rf gen_images; mkdir gen_images ")
    landmark = None
    for i in range(num_cams):
        theta_x = 0 + 10 * i
        theta_y = 0 + 0 * i
        theta_z = 0 + 0 * i
        t_x = 50 * i
        t_y = 100 * i
        t_z = 50 * i
        cam = util.Util_Camera(f_x = f_x,
                                f_y= f_y,
                                c_x= c_x,
                                c_y= c_y,
                                theta_x= theta_x,
                                theta_y= theta_y,
                                theta_z= theta_z,
                                t_x=t_x,
                                t_y=t_y,
                                t_z=t_z
                                )
        cam_pos = CameraPose(cam, i)
        if i == 0: 
            landmark = Landmark(num_landmarks, cam_pos)
        matches = cam_pos.projectLandmarks(landmark.get_Landmarks())
        landmark.updateCounter(matches)
        if sum(matches) * 1.0 / len(matches) < 0.2:
            print("Camera_"+str(i)+" has too few points:",sum(matches))
        campos_list.append(cam_pos)
        os.system("touch "+cam_pos.name)

    f = open("/Users/patrickji/workspace/visual_code/SctructureFromMotion/SFM_example/sfm_gen.g2o", 'w')
    output(f, campos_list, landmark)
    f.close()
    cv2_validation(campos_list)
