import cv2
import os
import numpy as np
import glob
import time


class stereoCalibration:
    def __init__(self, config):
        self.config = config

    def captureImage(self):
        # initiate camera
        camLeft = cv2.VideoCapture(0)
        camright = cv2.VideoCapture(1)
        pathLeft = os.path.join(os.getcwd(), 'calibrate_img', 'left_img')
        pathRight = os.path.join(os.getcwd(), 'calibrate_img', 'right_img')
        os.makedirs(pathLeft, exist_ok=True)
        os.makedirs(pathRight, exist_ok=True)
        num = 0
        print("For saving the file press s and for quit press q")
        while camLeft.isOpened():
            _, frameLeft = camLeft.read()
            _, frameRight = camright.read()
            k = cv2.waitKey(0)
            if k == ord("s"):
                cv2.imwrite(os.path.join(
                    pathLeft, "imgL"+str(num)+".png"), frameLeft)
                cv2.imwrite(os.path.join(
                    pathRight, "imgR"+str(num)+".png"), frameRight)
                print("image saved!!!")
                num += 1

            if k == ord("q"):
                cv2.destroyAllWindows()
                break
            print('wating for 10s')
            time.sleep(10)
        return [pathLeft, pathRight]

    def calibrate(self):
        if (self.config.calibImagepath is not None) and os.path.exists(self.config.calibImagepath):
            pathImg = os.listdir(self.config.calibImagepath)
            pathImg = [os.path.join(self.config.calibImagepath, pathImg[0]), os.path.join(
                self.config.calibImagepath, pathImg[1])]
        else:
            print("preparing for capturing image.....")
            time.sleep(10)
            pathImg = self.captureImage()

        left_image_file = glob.glob(
            os.path.join(pathImg[0], '*.png'))
        right_image_file = glob.glob(os.path.join(pathImg[1], '*.png'))
        img = cv2.imread(left_image_file[0])
        img_shape = img.shape
        frame_size = (img_shape[1], img_shape[0])
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # object points
        obj = np.zeros(
            (self.config.chessBoard_size[0]*self.config.chessBoard_size[1], 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:self.config.chessBoard_size[0],
                              0:self.config.chessBoard_size[1]].T.reshape((-1, 2))

        # camera clibration
        objpoints = []
        imgpointsLeft = []
        imgpointsRight = []
        for image_left, image_right in zip(left_image_file, right_image_file):
            # for left image
            img_left = cv2.imread(image_left)
            img_grey_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            retL, cornersL = cv2.findChessboardCorners(
                img_grey_left, self.config.chessBoard_size, None)
            # for right image
            img_right = cv2.imread(image_right)
            img_grey_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            retR, cornersR = cv2.findChessboardCorners(
                img_grey_right, self.config.chessBoard_size, None)

            if retL and retR:
                objpoints.append(obj)
                # Refine the corner locations for better accuracy
                corners2L = cv2.cornerSubPix(img_grey_left, cornersL, (11, 11), (-1, -1),
                                             criteria=criteria)
                imgpointsLeft.append(corners2L)
                # Refine the corner locations for better accuracy
                corners2R = cv2.cornerSubPix(img_grey_right, cornersR, (11, 11), (-1, -1),
                                             criteria=criteria)
                imgpointsRight.append(corners2R)

        # Perform camera calibration
        retL, camera_matrixL, dist_coeffsL, _, _ = cv2.calibrateCamera(
            objpoints, imgpointsLeft, frame_size, None, None)
        # undistortion
        newcammatix_L, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrixL, dist_coeffsL, (frame_size[1], frame_size[0]), 1, (frame_size[1], frame_size[0]))

        retR, camera_matrixR, dist_coeffsR, _, _ = cv2.calibrateCamera(
            objpoints, imgpointsRight, frame_size, None, None)
        # undistortion
        newcammatix_R, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrixR, dist_coeffsR, (frame_size[1], frame_size[0]), 1, (frame_size[1], frame_size[0]))

        # stereo calibration
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                           cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
        _, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, _, _ = cv2.stereoCalibrate(
            objpoints, imgpointsLeft, imgpointsRight, newcammatix_L, dist_coeffsL, newcammatix_R, dist_coeffsR, frame_size, criteria_stereo, flags)

        rectifyScale = 1
        rectL, rectR, projMatrixL, projMatrixR, _, _, _ = cv2.stereoRectify(
            newCameraMatrixL, distL, newCameraMatrixR, distR, frame_size, rot, trans, rectifyScale, (0, 0))

        map1_left, map2_left = cv2.initUndistortRectifyMap(
            newCameraMatrixL, distL, rectL, projMatrixL, frame_size, cv2.CV_16SC2)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            newCameraMatrixR, distR, rectR, projMatrixR, frame_size, cv2.CV_16SC2)

        cv_file = cv2.FileStorage(
            self.config.pathStereoMap, cv2.FILE_STORAGE_WRITE)

        # Check if the file opened correctly
        if not cv_file.isOpened():
            raise Exception("Failed to open file for writing!")
        cv_file.write('stereoMapL_x', map1_left)
        cv_file.write('stereoMapL_y', map2_left)
        cv_file.write('stereoMapR_x', map1_right)
        cv_file.write('stereoMapR_y', map2_right)

        cv_file.release()
        return 0
