import collections
import cv2
from cv2 import DescriptorMatcher
import numpy as np
import glob
import copy
import sys
from util import print_matrix


class Landmark(object):
    def __init__(self, x=0, y=0, z=0):
        self.pt = np.array([x, y, z], dtype=np.float)
        self.seen = 0

    def get_avg_pos(self):
        x = self.pt[0] / (self.seen - 1)
        y = self.pt[1] / (self.seen - 1)
        z = self.pt[2] / (self.seen - 1)
        return [x,y,z]
    
    def set_avg_pos(self):
        self.pt = np.array(self.get_avg_pos())

class ImagePose(object):
    def __init__(self):
        self.im = None
        self.desc = None
        self.kp = []
        self.T = None
        self.P = None
        self.F = None

        self.kp_kp = {}  # map kp_idx to {im_idx, kp_idx}
        self.kp_landmark = collections.defaultdict(int)  # map kp_idx to landmark_idx

    def set_kp_kp(self, kp_idx, im_idx, kp_idx2):
        if kp_idx not in self.kp_kp: 
            self.kp_kp[kp_idx] = {}
        self.kp_kp[kp_idx][im_idx] = kp_idx2

    def kp_match_exist(self, kp_idx, im_idx):
        return im_idx in self.kp_kp[kp_idx]

    def set_kp_landmark(self, kp_idx, lndmrk_idx):
        self.kp_landmark[kp_idx] = lndmrk_idx

    def kp_landmark_exist(self, kp_idx):
        return kp_idx in self.kp_landmark

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

class SFM(object):
    def __init__(self, dataset, downsample=1, num_images=-1, use_dummy=False):
        self.dataset = dataset
        self.downsample = downsample
        self.feature = cv2.AKAZE_create()
        self.matcher = cv2.DescriptorMatcher().create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
        self.images = []
        self.landmarks = []

        # initialization
        self.image_names = glob.glob(self.dataset)
        if num_images != -1:
            skip = 20
            #self.image_names = self.image_names[0:skip*num_images:skip]
            self.image_names = self.image_names[0:num_images]
        self.image_names.sort(key=lambda x: len(x))
        self.load_K()


    def load_K(self):
        K_txt = "/".join(self.dataset.split('/')[:-1]+['K.txt'])
        f = open(K_txt, 'r')
        K = np.zeros((3, 3))
        r = 0
        for line in f:
            for c, v in enumerate(line.split()):
                K[r][c] = float(v)
            r += 1
            if r >= 3:
                break
        f.close()
        self.K = K
        #self.K[0, 2] = cv2.imread(self.image_names[0]).shape[1] / 2
        #self.K[2, 2] = cv2.imread(self.image_names[0]).shape[0] / 2
        self.K = K / self.downsample
        self.FOCAL_LENGTH = (self.K[0][0] + self.K[1][1]) / 2
        self.K[0][0] = self.FOCAL_LENGTH
        self.K[1][1] = self.FOCAL_LENGTH
        self.K[2][2] = 1

    def output(self, outstream):
        outstream.write(str((self.K[0][0] + self.K[1][1]) / 2)+" "+str(self.K[0][2])+ " " + str(self.K[1][2])+" # F, cx, cy\n")
        outstream.write(str(len(self.image_names)) + " # number of image poses\n")
        outstream.write(str(len(self.landmarks)) + " # number of landmards\n")
        for i in range(len(self.landmarks)):
            outstream.write(str(self.landmarks[i].pt[0]) + " ")
            outstream.write(str(self.landmarks[i].pt[1]) + " ")
            outstream.write(str(self.landmarks[i].pt[2]) + " ")
            outstream.write("# landmark["+str(i)+'].pt\n')
        landmark_vertex_offset = len(self.image_names)
        for im_idx, im in enumerate(self.images):
            outstream.write(str(len(im.kp_landmark)) + " # Num of edges between image pose["+str(im_idx)+"] and landmarks\n")
            for (kp_idx, landmark_idx) in im.kp_landmark.items():
                outstream.write(str(im_idx)+' '+str(landmark_idx + landmark_vertex_offset) + " ")
                outstream.write(str(im.kp[kp_idx].pt[0]) + " " + str(im.kp[kp_idx].pt[1]) + " \n")

        for im_idx, im in enumerate(self.image_names):
            outstream.write(im+"\n")

        for im_idx, im in enumerate(self.images):
            for ri in range(3):
                for ci in range(4):
                    outstream.write(str(im.T[ri, ci])+" ")
                outstream.write('\n')

    def feature_extraction(self):
        for image_name in self.image_names:
            a = ImagePose()
            im = cv2.imread(image_name)
            assert(im is not None)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if self.downsample > 1:
                new_size = tuple([int(v / self.downsample) for v in im.shape])
                im = cv2.resize(im, new_size[::-1])
            a.im = im
            a.kp, a.desc = self.feature.detectAndCompute(a.im, None)
            print("Finish Image",image_name)
            self.images.append(a)

    def feature_matching(self, logging = False, view_match = False):
        for i, im_i in enumerate(self.images[:-1]):
            for j, im_j in enumerate(self.images[i + 1: i + 2], i + 1):
                i_kp, j_kp = [], []
                src, dst = [], []
                # do knnMatch first and ransac later
                matches = self.matcher.knnMatch(im_i.desc, im_j.desc,2)
                for match in matches:
                    if match[0].distance >= 0.75 * match[1].distance:
                        continue
                    kp_i_idx, kp_j_idx = match[0].queryIdx, match[0].trainIdx
                    src.append(im_i.kp[kp_i_idx].pt)
                    dst.append(im_j.kp[kp_j_idx].pt)
                    i_kp.append(kp_i_idx)
                    j_kp.append(kp_j_idx)
                # do fundamentalMat RANSAC 
                src = np.float32([pt for pt in src])
                dst = np.float32([pt for pt in dst])
                F, masks = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, ransacReprojThreshold = 3, confidence = 0.99)
                im_j.F = F
                print("Feature Matching",i,'vs.',j,'=',np.sum(masks),'/',src.shape[0])
                if logging:
                    print("Feature Matching",i,'vs.',j,'=',np.sum(masks),'/',src.shape[0])
                if masks is None:
                    print("Cannot Find Enough Feature Matching Between "+str(i)+" and "+str(j))
                    continue
                for k, mask_v in enumerate(masks):
                    if not mask_v: continue
                    im_i.set_kp_kp(i_kp[k], j, j_kp[k])
                    im_j.set_kp_kp(j_kp[k], i, i_kp[k])
                if view_match:
                    win_name = str(i) + " VS. " + str(j)
                    cv2.namedWindow(win_name)
                    im1 = copy.deepcopy(im_i.im)
                    im2 = copy.deepcopy(im_j.im)
                    canvas = np.column_stack([im1, im2])
                    offset = im2.shape[1]
                    for k, mask_v in enumerate(masks):
                        if not mask_v: continue
                        kp_i_loc = tuple(np.int32(src[k]))
                        kp_j_loc = list(np.int32(dst[k]))
                        kp_j_loc[0] += offset
                        color = [0] * 3
                        color[k % 3] = 255
                        cv2.line(canvas, kp_i_loc, tuple(kp_j_loc), tuple(color), 1)
                    #new_size = tuple([int(v / self.downsample) for v in canvas.shape])
                    new_size = tuple([int(v / 4) for v in canvas.shape])
                    canvas = cv2.resize(canvas, (new_size[1], new_size[0]))
                    cv2.imshow(win_name, canvas)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    def validate_P(self):
        for img_idx, im in enumerate(self.images):
            print("------",img_idx,'-------')
            for kp_idx, lnd_mrk_idx in im.kp_landmark.items():
                pt = im.kp[kp_idx].pt
                xyz = self.landmarks[lnd_mrk_idx].pt
                XYZ_1 = np.array([xyz[0], xyz[1], xyz[2], 1])
                uvw = im.P.dot(XYZ_1.reshape(4,1))
                u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
                print('im',img_idx,'kp',kp_idx,'landmark',lnd_mrk_idx)
                print('    ',pt)
                print('    ',(u[0], v[0]))

    def pose_recover(self, logging = False):
        if 0:
            ret, landmark_kp_mapping = read_g2o('/Users/patrickji/workspace/visual_code/SctructureFromMotion/SFM_example/sfm_gen.g2o')
            self.images = []
            self.image_names = []
            self.FOCAL_LENGTH = ret['f_x']
            c_x, c_y = ret['c_x'], ret['c_y']
            f_x = ret['f_x']
            self.K = np.array([[f_x, 0, c_x], [0, f_x, c_y], [0, 0, 1]])
            for i in range(ret['num_pos']):
                pos = ImagePose()
                self.images.append(pos)
                for j in range(1000):
                    pos.kp.append(None)
                self.image_names.append(ret["image_"+str(i)+"_name"])
            for key, vals in landmark_kp_mapping.items():
                landmark_id = key
                vals.sort()
                for i1 in range(len(vals) - 1):
                    for i2 in range(i1 + 1, len(vals)):
                        im1_id, kpt1, (u1, v1) = vals[i1]
                        im2_id, kpt2, (u2, v2) = vals[i2]
                        if self.images[im1_id].kp[kpt1] is None:
                            kp1 = cv2.KeyPoint()
                            kp1.pt = (u1, v1)
                            self.images[im1_id].kp[kpt1] = kp1
                        if self.images[im2_id].kp[kpt2] is None:
                            kp2 = cv2.KeyPoint()
                            kp2.pt = (u2, v2)
                            self.images[im2_id].kp[kpt2] = kp2
                        self.images[im1_id].set_kp_kp(kpt1, im2_id, kpt2)
                        self.images[im2_id].set_kp_kp(kpt2, im1_id, kpt1)

        self.images[0].T = np.eye(4, 4, dtype = np.float)
        self.images[0].P = np.dot(self.K, np.eye(3, 4, dtype = np.float))

        recovered_landmarks = []

        for idx, p_img in enumerate(self.images[:-1]):
            # p_img means previous images
            c_img = self.images[idx + 1] # current images
            src, dst = [], []
            p_kp_used = []
            for i in range(len(p_img.kp)):
                if i in p_img.kp_kp and idx + 1 in p_img.kp_kp[i]:
                    dst_kp_idx = p_img.kp_kp[i][idx + 1]
                    src.append(p_img.kp[i].pt)
                    dst.append(c_img.kp[dst_kp_idx].pt)
                    p_kp_used.append(i)
            
            src = np.float32([pt for pt in src])
            dst = np.float32([pt for pt in dst])
            #E = e_estimation(c_img.F, self.K)
            #src = cv2.undistortPoints(np.expand_dims(src, axis = 1), self.K, distCoeffs=None)
            #dst = cv2.undistortPoints(np.expand_dims(dst, axis = 1), self.K, distCoeffs=None)
            #src = np.reshape(src, (src.shape[0], -1))
            #dst = np.reshape(dst, (dst.shape[0], -1))
            E, mask = cv2.findEssentialMat(src, dst, 
                                            cameraMatrix = self.K, 
                                            method=cv2.FM_RANSAC, 
                                            prob = 0.99, 
                                            threshold = 1.0)
            _, local_R, local_T, mask = cv2.recoverPose(E, src, dst, 
                                                        cameraMatrix = self.K, mask = mask)
            T = np.eye(4)
            T[0:3, 0:3] = local_R
            T[0:3, 3:4] = local_T
            P = None
            c_img.T = T.dot(p_img.T)
            R, t = c_img.T[0:3, 0:3], c_img.T[0:3, 3:4]
            P = np.column_stack([R, t])
            c_img.P = self.K.dot(P)

            # triangulate
            points4D = cv2.triangulatePoints(p_img.P, c_img.P, src.transpose(), dst.transpose())

            # scale back
            if idx > 0:
                scale = 0.0
                count = 0
                p_camera_x = p_img.T[0, 3]
                p_camera_y = p_img.T[1, 3]
                p_camera_z = p_img.T[2, 3]
                new_pts = []
                existing_pts = []

                log0, log1, log2 = 0, 0, 0
                for j, kp_idx in enumerate(p_kp_used):
                    used_during_pose_recovery = (mask[j] != 0)
                    exist_in_prev_landmark = p_img.kp_landmark_exist(kp_idx)

                    if used_during_pose_recovery: log0 += 1
                    if exist_in_prev_landmark: log2 += 1

                    if used_during_pose_recovery and exist_in_prev_landmark:
                        point_x = points4D[0, j] / points4D[3, j]
                        point_y = points4D[1, j] / points4D[3, j]
                        point_z = points4D[2, j] / points4D[3, j]

                        landmark_idx = p_img.kp_landmark[kp_idx]
                        new_pts.append([point_x, point_y, point_z])
                        existing_pts.append(self.landmarks[landmark_idx].get_avg_pos())

                # calculate the scale
                for j, (new_pt_0, existing_pt_0) in enumerate(list(zip(new_pts, existing_pts))[:-1]):
                    for new_pt_1, existing_pt_1 in list(zip(new_pts, existing_pts))[j + 1:]:
                        pa_0, pb_0 = np.array(new_pt_0), np.array(existing_pt_0)
                        pa_1, pb_1 = np.array(new_pt_1), np.array(existing_pt_1)
                        scale += (np.linalg.norm(pb_0 - pb_1) / np.linalg.norm(pa_0 - pa_1))
                        count += 1
                if count == 0:
                    print("img", idx, 'and img',idx + 1,'doesnt match at all')
                    print(log0, log2)
                    print(self.image_names[idx])
                    print(self.image_names[idx + 1])
                
                assert(count > 0)
                scale /= count

                # apply scale and recal T and P
                local_T *= scale
                # local transform
                T = np.eye(4, dtype= np.float)
                T[0:3, 0:3] = local_R
                T[0:3, 3:4] = local_T

                P = None
                # accumulate transform
                c_img.T = T.dot(p_img.T)
                # make projection matrix
                R, t = c_img.T[0:3, 0:3], c_img.T[0:3, 3:4]
                P = np.column_stack([R, t])
                c_img.P = self.K.dot(P)

                points4D = cv2.triangulatePoints(p_img.P, 
                                                c_img.P, 
                                                src.transpose(), 
                                                dst.transpose())

            print("find good triangulated points")
            for j in range(len(p_kp_used)):
                if mask[j] != 0:
                    p_kp_idx = p_kp_used[j]
                    match_idx = p_img.kp_kp[p_kp_idx][idx + 1]
                    pt3d_x = points4D[0, j] / points4D[3, j]
                    pt3d_y = points4D[1, j] / points4D[3, j]
                    pt3d_z = points4D[2, j] / points4D[3, j]
                    pt3d = np.array([pt3d_x, pt3d_y, pt3d_z])

                    if p_img.kp_landmark_exist(p_kp_idx):
                        c_img.set_kp_landmark(match_idx, p_img.kp_landmark[p_kp_idx])

                        self.landmarks[p_img.kp_landmark[p_kp_idx]].pt += pt3d
                        self.landmarks[p_img.kp_landmark[p_kp_idx]].seen += 1
                    else:
                        new_landmark = Landmark(pt3d_x, pt3d_y, pt3d_z)
                        new_landmark.seen = 2
                        self.landmarks.append(new_landmark)
                        p_img.set_kp_landmark(p_kp_idx, len(self.landmarks) - 1)
                        c_img.set_kp_landmark(match_idx, len(self.landmarks) - 1)
                        recovered_landmarks.append([idx, p_kp_idx])

        for landmark_i, landmark in enumerate(self.landmarks):
            if landmark.seen > 2:
                landmark.set_avg_pos()
            if 0:
                recovered_im_idx, recovered_kp_idx = recovered_landmarks[landmark_i]
                for landmark_id, vals in landmark_kp_mapping.items():
                    for (im_idx, kp_idx, (u, v)) in vals:
                        if im_idx == recovered_im_idx and kp_idx == recovered_kp_idx:
                            print('----------------')
                            print(ret['landmarks'][landmark_id])
                            ox, oy, oz = ret['landmarks'][landmark_id]
                            x, y, z = landmark.pt
                            rscale = oz / z
                            x, y, z = x * rscale, y * rscale, z * rscale
                            print((x, y, z))
                            print((x - ox, y - oy, z - oz))
        
        scale = 1 #50.0 / self.images[1].T[0, 3]
        for idx, c_img in enumerate(self.images):
            T = c_img.T
            T[0:3, 3] = T[0:3, 3] * scale
            print("c_img["+str(idx)+"].P=")
            #print(T[0:3, 0:4])
            print_matrix(T[0:3, 0: 4])
            #print_matrix(c_img.P)



def read_g2o(path):
    def get_line(inputstream):
        line = inputstream.readline()
        line = line.rstrip()
        return line.split()

    f = open(path, 'r')
    ret = {}
    fields = get_line(f)
    ret['f_x'] = int(fields[0])
    ret['c_x'] = int(fields[1])
    ret['c_y'] = int(fields[2])
    fields = get_line(f)
    ret['num_pos'] = int(fields[0])
    fields = get_line(f)
    ret['num_landmark'] = int(fields[0])
    ret['landmarks'] = []
    for i in range(ret['num_landmark']):
        fields = get_line(f)
        x, y, z = float(fields[0]), float(fields[1]), float(fields[2])
        ret['landmarks'].append([x, y, z])
    landmarks_kp_mapping = collections.defaultdict(list)
    for i in range(ret['num_pos']):
        kp_landmark_num = int(get_line(f)[0])
        for j in range(kp_landmark_num):
            fields = get_line(f)
            im_idx = int(fields[0])
            landmark_idx = int(fields[1]) - ret['num_pos']
            u, v = float(fields[2]), float(fields[3])
            landmarks_kp_mapping[landmark_idx].append([im_idx, j, (u, v)])
    for i in range(ret['num_pos']):
        ret["image_"+str(i)+"_name"] = get_line(f)[0]
    f.close()
    return [ret, landmarks_kp_mapping]


if __name__ == "__main__":
    #dataset ="/Users/patrickji/workspace/visual_code/SctructureFromMotion/DataSet/desktop/*.jpg"
    PROJ_DIR = "/Users/patrickji/workspace/visual_code/SctructureFromMotion/"
    dataset = ["DataSet/hall/*.jpg",  10]
    dataset = ["DataSet/castle/*.JPG",10]
    dataset = ["DataSet/gelou/*.JPG", 10]
    dataset = ["DataSet/cabin/*.JPG", 10]
    dataset = ["DataSet/desk/*.JPG",  10]
    dataset = ["DataSet/Condo/*.jpg",   10]
    dataset = ["DataSet/berlin/*.jpg",   10]
    dataset = ["DataSet/fan/*.JPG",   10]
    dataset = ["DataSet/desktop/*.JPG",  10]
    dataset = ["DataSet/bed/*.JPG",  3]
    sfm = SFM(dataset=PROJ_DIR+dataset[0], 
                num_images = dataset[1], 
                use_dummy = False)
    sfm.feature_extraction()
    sfm.feature_matching(logging=False, view_match=False)
    sfm.pose_recover()
    #sfm.output(sys.stdout)
    f = open('sfm.g2o', 'w')
    sfm.output(f)
    #sfm.validate_P()
    f.close()