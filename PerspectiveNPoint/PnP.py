import numpy as np
import cv2 as cv
import util
def pnp(K, objPoints, imgPoints):
    ret, rvec, tvec, inliars = cv.solvePnPRansac(
                                    objPoints,
                                    imgPoints,
                                    K,
                                    None,)
    return rvec, tvec, inliars

if __name__ == "__main__":
    import util
    kpt1s, kpt2s, matches, cam1, cam2, points3D = util.gen_kpt_matches(inliar_percent=1.0)
    two_view = util.Two_View_System(cam1, cam2)
    two_view.set_points3D(points3D)
    objPoints = np.ndarray((len(matches), 3, 1))
    imgPoints = np.ndarray((len(matches), 2, 1))
    for idx, match in enumerate(matches):
        idx2 = match.trainIdx
        imgPoints[idx][0] = kpt2s[idx2].pt[0]
        imgPoints[idx][1] = kpt2s[idx2].pt[1]
        objPoints[idx][0] = points3D[idx][0]
        objPoints[idx][1] = points3D[idx][1]
        objPoints[idx][2] = points3D[idx][2]
    rvec, tvec = pnp(cam1.K, objPoints, imgPoints)
    print(rvec)
    print(tvec)