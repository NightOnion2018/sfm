import cv2
import glob
import numpy as np
import random
from matplotlib import pyplot as plt

def feature_matching(im1, im2, logging = False):
    """
    input: image_path is the image search pattern, like '../*.jpg'
    output: feature matching between any two pair of images
    """
    imname_lists = [im1, im2]
    fast = cv2.FastFeatureDetector_create()
    if logging:
        for im in imname_lists:
            print(im)
    features_descriptors = []
    for fname in imname_lists:
        im = cv2.imread(fname, 0)
        #orb = cv2.ORB_create()
        orb = cv2.AKAZE_create()
        kp, des = orb.detectAndCompute(im, None)
        features_descriptors.append([kp, des])
        if logging:
            im = cv2.drawKeypoints(im, kp, im, color=(255,0,0))
            cv2.imshow("image", im)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    # bbrute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match Descriptors
    matches_save = {}
    for im1_idx in range(len(imname_lists)):
        for im2_idx in range(im1_idx + 1, len(imname_lists)):
            matches = bf.match(features_descriptors[im1_idx][1],
                                features_descriptors[im2_idx][1])
            matches = sorted(matches, key = lambda x: x.distance)
            if len(matches) > 10:
                matches_save[(im1_idx, im2_idx)] = matches
            else:
                print("Cannot Find Enough Matches in IM_"+str(im1_idx)+" IM_"+str(im2_idx))
    if logging and len(matches_save) > 0:
        for (idx1, idx2), matches in matches_save.items():
            im1 = cv2.imread(imname_lists[idx1])
            im2 = cv2.imread(imname_lists[idx2])
            img3 = cv2.drawMatches(im1,
                            features_descriptors[idx1][0],
                            im2,
                            features_descriptors[idx2][0],
                            matches,
                            None,
                            matchColor = (255, 0, 0))
            plt.imshow(img3)
            plt.title("Matching Between IM_"+str(idx1)+" IM_"+str(idx2))
            plt.show()
    return (imname_lists, features_descriptors, matches_save[(0, 1)])

def f_estimation(matches):
    """
    Estimation of fundamental matrix from matched key port
    At least of 8 matched points should be provided
    """
    assert len(matches) >= 8
    num_of_matches = len(matches)
    A = np.ndarray([num_of_matches, 9], dtype = np.float32)
    for idx in range(num_of_matches):
        x1, y1, x2, y2 = matches[idx]
        A[idx][0] = x2 * x1
        A[idx][1] = x2 * y1
        A[idx][2] = x2
        A[idx][3] = y2 * x1
        A[idx][4] = y2 * y1
        A[idx][5] = y2
        A[idx][6] = x1
        A[idx][7] = y1
        A[idx][8] = 1
    U, SIGMA, V_t = np.linalg.svd(A)
    F = np.reshape(np.transpose(V_t)[:,-1],(3,3))
    u, sigma, v_t = np.linalg.svd(F)
    sigma[2] = 0
    sigma = np.array([[sigma[0], 0, 0], [0, sigma[1], 0], [0,0,0]])
    F = np.dot(np.dot(u, sigma), v_t)
    return F

def check_F(F, samples):
    for sample in samples:
        x1, y1, x2, y2 = sample
        X_p = np.array([x2, y2, 1])
        X = np.array([x1, y1, 1])
        res = np.dot(X_p, np.dot(F, X.reshape(3, 1)))
        print("Checking F res ",res)

def f_estimation_ransac(key_desc1, 
                        key_desc2, 
                        matches, 
                        iterations = 1000,
                        logging = False,
                        INLINER_PRECISION = 0.05,
                        t_F = None):
    # Hyper Parameters
    NUM_OF_MATCHES_FOR_F_ESTIMATION = 8

    max_inliers = 0
    F_best = None

    #max_res = 1000000
    inlier_per = 0.0

    iteration = 0
    while iteration < iterations:
        iteration += 1
        indexs = random.sample(range(len(matches)), NUM_OF_MATCHES_FOR_F_ESTIMATION)
        sampled_matches = []
        for idx in indexs:
            key_idx1 = matches[idx].queryIdx
            key_idx2 = matches[idx].trainIdx
            sampled_matches.append([key_desc1[key_idx1].pt[0],
                                    key_desc1[key_idx1].pt[1],
                                    key_desc2[key_idx2].pt[0],
                                    key_desc2[key_idx2].pt[1]])
        #check_F(t_F, sampled_matches)
        F = f_estimation(sampled_matches)
        #check_F(F, sampled_matches)
        match_count = 0
        for match in matches:
            key_idx1 = match.queryIdx
            key_idx2 = match.trainIdx
            x1, y1 = key_desc1[key_idx1].pt
            x2, y2 = key_desc2[key_idx2].pt
            P1_h = np.array([x1, y1, 1])
            P2_h = np.array([x2, y2, 1])
            res = np.dot(np.dot(P2_h, F), P1_h.reshape(3,1))
            #max_res = min(max_res, abs(res))
            if abs(res) <= INLINER_PRECISION:
                match_count += 1
        if match_count > max_inliers:
            max_inliers = match_count
            inlier_per = max_inliers * 1.0 / len(matches)
            F_best = F
            if logging:
                print("Get Better F with higher Precision:" + str(match_count * 1.0 / len(matches)))

    #if logging: print("Max Res is", max_res)
    new_matches = []
    print('len of matches', len(matches))
    for idx, match in enumerate(matches):
        key_idx1 = match.queryIdx
        key_idx2 = match.trainIdx
        x1, y1 = key_desc1[key_idx1].pt
        x2, y2 = key_desc2[key_idx2].pt
        P1_h = np.array([x1, y1, 1])
        P2_h = np.array([x2, y2, 1])
        res = np.dot(np.dot(P2_h, F_best), P1_h.reshape(3,1))
        if abs(res) <= INLINER_PRECISION:
            new_matches.append(match)
    print('len of refined matches', len(new_matches))
    return F_best, (max_inliers * 1.0) / len(matches), new_matches

def inspect_matching(im1, im2, kp1, kp2, F, matches, INLINER_PRECISION = 0.1):
    def get_fundamental_res(p1, p2, F):
        x1, y1 = p1
        x2, y2 = p2
        P1_h = np.array([x1, y1, 1])
        P2_h = np.array([x2, y2, 1])
        return np.dot(np.dot(P2_h.reshape(1, 3), F), P1_h.reshape(3,1))

    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    inlier_matches = []
    outlier_matches = []
    for match in matches:
        p1_idx = match.queryIdx
        p2_idx = match.trainIdx
        p1 = kp1[p1_idx].pt
        p2 = kp2[p2_idx].pt
        res = get_fundamental_res(p1, p2, F)
        if abs(res) <= INLINER_PRECISION:
            inlier_matches.append(match)
        else:
            outlier_matches.append(match)

    im3 = cv2.drawMatches(im1,
                        kp1,
                        im2,
                        kp2,
                        inlier_matches,
                        None,
                        matchColor = (255, 0, 0))
    im4 = cv2.drawMatches(im1,
                        kp1,
                        im2,
                        kp2,
                        outlier_matches,
                        None,
                        matchColor = (255, 0, 0))
    plt.subplot(2,1,1)
    plt.imshow(im3)
    plt.title("inliers")
    plt.subplot(2,1,2)
    plt.imshow(im4)
    plt.title("outliers")
    plt.show()

if __name__ == "__main__":
    imname_lists = glob.glob("DataSet/castle/*.JPG")
    imname_lists = glob.glob("DataSet/desktop/*.jpg")
    imname_lists.sort(key = lambda x: len(x))
    im1 = imname_lists[1]
    im2 = imname_lists[2]
    (imname_lists, 
    features_descriptors, 
    matches_save) = feature_matching(im1, im2, logging = False)
    INLINER_PRECISION = 0.03
    F, precision, _ = f_estimation_ransac(features_descriptors[0][0],
                                       features_descriptors[1][0],
                                       matches_save,
                                       logging = False,
                                       INLINER_PRECISION = INLINER_PRECISION)
    print((im1, im2), ":", precision)
    if 1:   
        inspect_matching(imname_lists[0],
                        imname_lists[1],
                        features_descriptors[0][0],
                        features_descriptors[1][0],
                        F,
                        matches_save,
                        INLINER_PRECISION=INLINER_PRECISION)