import collections
import cv2
from cv2 import DescriptorMatcher
import numpy as np
import glob
import copy
import sys
from util import print_matrix
import CameraPoseEstimation.CameraPoseEstimation as cpe
from PerspectiveNPoint import PnP as pnp

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

class SFM(object):
    def __init__(self, dataset, downsample=1, num_images=-1, use_dummy=False):
        self.dataset = dataset
        self.downsample = downsample
        self.feature = cv2.AKAZE_create()
        self.matcher = cv2.DescriptorMatcher().create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
        self.images = []
        self.landmarks = []
        self.F_mats = {}

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
                self.F_mats[(i, j)] = F
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

    def pose_recover(self, logging = False, validate_mode = False):
        if validate_mode:
            ret, landmark_kp_mapping = read_g2o('/Users/patrickji/workspace/visual_code/SctructureFromMotion/SFM_example/sfm_gen.g2o')
            self.images = []
            self.image_names = []
            self.FOCAL_LENGTH = ret['f_x']
            c_x, c_y = ret['c_x'], ret['c_y']
            f_x = ret['f_x']
            self.K = np.array([[f_x, 0, c_x], [0, f_x, c_y], [0, 0, 1]], dtype=np.float)
            #self.F_mats = ret['f_mats']
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

        for p_img_idx, p_img in enumerate(self.images[:-1]):
            # p_img means previous images
            c_img_idx = p_img_idx + 1
            c_img = self.images[c_img_idx] # current images
            src, dst = [], []
            p_kp_used = []
            for i in range(len(p_img.kp)):
                if i in p_img.kp_kp and c_img_idx in p_img.kp_kp[i]:
                    dst_kp_idx = p_img.kp_kp[i][c_img_idx]
                    src.append(p_img.kp[i].pt)
                    dst.append(c_img.kp[dst_kp_idx].pt)
                    p_kp_used.append(i)
            
            src = np.float32([pt for pt in src])
            dst = np.float32([pt for pt in dst])

            #src = cv2.undistortPoints(np.expand_dims(src, axis = 1), self.K, distCoeffs=None)
            #dst = cv2.undistortPoints(np.expand_dims(dst, axis = 1), self.K, distCoeffs=None)
            #src = np.reshape(src, (src.shape[0], -1))
            #dst = np.reshape(dst, (dst.shape[0], -1))

        
            if p_img_idx == 0:
                F, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC)
                E = cpe.e_estimation(F, self.K)
                #E = cpe.e_estimation(self.F_mats[(p_img_idx, c_img_idx)], self.K)
                RTs = cpe.cal_possible_RT(E)
                best_RT, max_val_perc, best_mask = None, -1, None
                for i, (T, R) in enumerate(RTs):
                    val_perc, mask = cpe.cheirality_check(self.K, R, T,
                                                         src,
                                                         dst)
                    if val_perc > max_val_perc:
                        best_RT = [R, T]
                        max_val_perc = val_perc
                        best_mask = mask
                local_R, local_T = best_RT[0], best_RT[1].reshape(3, 1)
                mask = best_mask
                print("The Correct RT is\n",np.column_stack([local_R, local_T]), "\n with confidence of", max_val_perc*100, '%')

                T = np.eye(4)
                T[0:3, 0:3] = local_R
                T[0:3, 3:4] = local_T
                c_img.T = T.dot(p_img.T)
                R, t = c_img.T[0:3, 0:3], c_img.T[0:3, 3:4]
                P = np.column_stack([R, t])
                c_img.P = self.K.dot(P)
                
                print("Triangulate Between First Two Images")
                points4D = cv2.triangulatePoints(p_img.P, 
                                                 c_img.P, 
                                                 src.transpose(), 
                                                 dst.transpose())
                for j, maskv in enumerate(mask):
                    if maskv != 0:
                        p_kp_idx = p_kp_used[j]
                        match_idx = p_img.kp_kp[p_kp_idx][c_img_idx]
                        pt3d_x = points4D[0, j] / points4D[3, j]
                        pt3d_y = points4D[1, j] / points4D[3, j]
                        pt3d_z = points4D[2, j] / points4D[3, j]

                        new_landmark = Landmark(pt3d_x, pt3d_y, pt3d_z)
                        new_landmark.seen = 2
                        self.landmarks.append(new_landmark)
                        p_img.set_kp_landmark(p_kp_idx, len(self.landmarks) - 1)
                        c_img.set_kp_landmark(match_idx, len(self.landmarks) - 1)
                        recovered_landmarks.append([p_img_idx, p_kp_idx])
            else:
                print("Triangulate Between IMG["+str(p_img_idx)+"] and IMG["+str(c_img_idx)+"]")

                # step 0: find the matched landmark in p_img and use pnp to recover rvec, tvec
                existing_landmark_count = 0
                for used_kp_idx in range(len(dst)):
                    p_img_kp_idx = p_kp_used[used_kp_idx]
                    if p_img_kp_idx in p_img.kp_landmark:
                        existing_landmark_count += 1
                objPoints = np.ndarray((existing_landmark_count, 3, 1))
                imgPoints = np.ndarray((existing_landmark_count, 2, 1))
                print("PNP IMG["+str(c_img_idx)+"] With "+str(existing_landmark_count)+" Landmarks")
                assert(existing_landmark_count > 8)
                current_pos = 0
                for used_kp_idx in range(len(dst)):
                    p_img_kp_idx = p_kp_used[used_kp_idx]
                    if p_img_kp_idx not in p_img.kp_landmark:
                        continue
                    landmark_idx = p_img.kp_landmark[p_img_kp_idx]
                    imgPoints[current_pos][0] = dst[used_kp_idx][0]
                    imgPoints[current_pos][1] = dst[used_kp_idx][1]
                    landmark_pt = self.landmarks[landmark_idx].get_avg_pos()
                    objPoints[current_pos][0] = landmark_pt[0]
                    objPoints[current_pos][1] = landmark_pt[1]
                    objPoints[current_pos][2] = landmark_pt[2]
                    current_pos += 1

                rvec, tvec, inliers = pnp.pnp(self.K, objPoints, imgPoints) # should use EPNP for noisy model
                rmat = cv2.Rodrigues(rvec)[0]
                #print("IMG["+str(c_img_idx)+"] RT: \n", np.column_stack([rmat, tvec]))

                c_img.T = np.eye(4)
                c_img.T[0:3, 0:3] = rmat
                c_img.T[0:3, 3:4] = tvec
                c_img.P = self.K.dot(c_img.T[0:3, 0:4])

                # step 1: use the rmat, tvec to calculate the new landmarks and add landmarks
                points4D = cv2.triangulatePoints(p_img.P, 
                                                 c_img.P, 
                                                 src.transpose(), 
                                                 dst.transpose())
                for used_kp_idx in range(len(dst)):
                    p_img_kp_idx = p_kp_used[used_kp_idx]
                    c_img_kp_idx = p_img.kp_kp[p_img_kp_idx][c_img_idx]
                    harmonic_v = points4D[3, used_kp_idx]
                    pt3d_x = points4D[0, used_kp_idx] / harmonic_v
                    pt3d_y = points4D[1, used_kp_idx] / harmonic_v
                    pt3d_z = points4D[2, used_kp_idx] / harmonic_v
                    if p_img_kp_idx not in p_img.kp_landmark:
                        landmark = Landmark(pt3d_x, pt3d_y, pt3d_z)
                        landmark.seen = 2
                        self.landmarks.append(landmark)
                        landmark_idx = len(self.landmarks) - 1
                        p_img.set_kp_landmark(p_img_kp_idx, landmark_idx)
                        c_img.set_kp_landmark(c_img_kp_idx, landmark_idx)
                    else:
                        landmark_idx = p_img.kp_landmark[p_img_kp_idx]
                        c_img.set_kp_landmark(c_img_kp_idx, landmark_idx)
                        pt3d = np.array([pt3d_x, pt3d_y, pt3d_z])
                        self.landmarks[landmark_idx].pt += pt3d
                        self.landmarks[landmark_idx].seen += 1

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
        for c_img_idx, c_img in enumerate(self.images):
            T = c_img.T
            T[0:3, 3] = T[0:3, 3] * scale
            print("c_img["+str(c_img_idx)+"].P=")
            print_matrix(T[0:3, 0: 4])

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


    # skip p_matrix
    for i in range(ret['num_pos']):
        get_line(f)
        get_line(f)
        get_line(f)
    return [ret, landmarks_kp_mapping]
    # read fundamental matrix
    ret['f_mats'] = {}
    for i in range(ret['num_pos']):
        f00, f01, f02 = get_line(f)[:3]
        f10, f11, f12 = get_line(f)[:3]
        f20, f21, f22 = get_line(f)[:3]
        F = np.array([[f00, f01, f02], [f10, f11, f12], [f20, f21, f22]])
        ret['f_mats'][(i, i + 1)] = f
    f.close()
    return [ret, landmarks_kp_mapping]


if __name__ == "__main__":
    #dataset ="/Users/patrickji/workspace/visual_code/SctructureFromMotion/DataSet/desktop/*.jpg"
    PROJ_DIR = "/Users/patrickji/workspace/visual_code/SctructureFromMotion/"
    dataset = ["DataSet/Condo/*.jpg",   10]
    dataset = ["DataSet/berlin/*.jpg",   10]
    dataset = ["DataSet/bed/*.JPG",  3]
    dataset = ["DataSet/desk/*.JPG",  10, 'Good']
    dataset = ["DataSet/desktop/*.JPG",  10]
    dataset = ["DataSet/fan/*.JPG",   3]
    dataset = ["DataSet/castle/*.JPG",11, 'Good']
    dataset = ["DataSet/hall/*.jpg",  30]
    dataset = ["DataSet/cabin/*.JPG", 10]
    VALIDATE_MODE = False
    sfm = SFM(dataset=PROJ_DIR+dataset[0], 
                num_images = dataset[1], 
                use_dummy = False)
    if not VALIDATE_MODE:
        sfm.feature_extraction()
        sfm.feature_matching(logging=False, view_match=False)
    #sfm.pose_recover_validate()
    sfm.pose_recover(validate_mode=VALIDATE_MODE)
    #sfm.output(sys.stdout)
    f = open('sfm.g2o', 'w')
    sfm.output(f)
    #sfm.validate_P()
    f.close()
