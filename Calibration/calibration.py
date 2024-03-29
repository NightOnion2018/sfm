import numpy as np
import cv2 as cv
import glob
import json
import copy
import os

def openCameraNCal(camId, num_sample, save_path):
    cam = cv.VideoCapture(camId)
    points_to_find = (5, 5)
    img_idx = 0
    while img_idx < num_sample:
        valid, image = cam.read()
        if not valid:
            break
        else:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            print("Capture One Image, Processing...")
            ret, corners = cv.findChessboardCorners(gray, points_to_find, None)
            if ret:
                img_idx += 1
                cv.imshow("cam", gray)
                print("Yes I find Corners Save this image")
                cv.imwrite(save_path+"/cam_"+str(img_idx)+".jpg", gray)


def calibrate(path, logging = False):

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    points_to_find = (5, 5)
    objp = np.zeros((points_to_find[0] * points_to_find[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:points_to_find[0],
                       0:points_to_find[1]].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(path)
    used_images = []
    for fname in images:
        img = cv.imread(fname)
        #print("Processing "+fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, points_to_find, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Find enough Corners in "+fname)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            used_images.append(fname)
            # Draw and display the corners
            if 0:
                cv.drawChessboardCorners(img, points_to_find, corners2, ret)
                cv.imshow("checker_board", img)
                cv.waitKey(0)
        else:
            print("Cannot find enough Corners in "+fname)
    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                        objpoints, imgpoints, gray.shape[::-1], None, None)

    K_path = '/'.join(path.split('/')[:-1])+"/K.txt"
    dist_path = '/'.join(path.split('/')[:-1])+"/dist.txt"
    f = open(K_path, 'w')
    for i in range(3):
        for j in range(3):
            f.write(str(mtx[i, j]) + " ")
        f.write('\n')
    f.close()
    
    f = open(dist_path, 'w')
    for i in range(5):
        f.write(str(dist[0,i]) + ' ')
    f.close()
    
    if logging:
        print("intrisc matrix")
        print(mtx)
        print("Distoration:")
        print(dist)
    return mtx

def undistort(cal_path, path):
    mtx = np.ndarray((3,3))
    dist = np.ndarray((1, 5))
    mtx_f = open(cal_path+"/K.txt", 'r')
    for i in range(3):
        line = mtx_f.readline()
        fields = list(map(float, line.split()))
        for j in range(3):
            mtx[i, j] = fields[j]
    mtx_f.close()

    dist_f = open(cal_path+"/dist.txt", 'r')
    line = dist_f.readline()
    fields = list(map(float, line.split()))
    for i in range(5):
        dist[0, i] = fields[i]
    dist_f.close()

    images = glob.glob(path)
    for im_name in images:
        new_name = im_name.rstrip('.JPG')+"_cal.JPG"
        im = cv.imread(im_name)
        h,  w = im.shape[:2]
        newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv.undistort(im, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite(new_name, dst)
    print("new camera matrix\n",newcameramtx)


if __name__ == "__main__":
    path = './DataSet/calibration/*.jpg'
    path = './DataSet/iphone_calibration/*.JPG'
    #openCameraNCal(0, num_sample = 20, save_path = "./DataSet/calibration/")
    #intrinsic = calibrate(path, logging=True)
    undistort(path.rstrip("*.JPG"), "./DataSet/desktop_org/*.JPG")