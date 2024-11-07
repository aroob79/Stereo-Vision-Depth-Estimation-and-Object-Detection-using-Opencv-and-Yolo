from ultralytics import YOLO
from utils import imageCapture
from utils import depthCalculation
from utils import plot_image
from utils import generate_disperaty_map
import cv2
import numpy as np
import copy


class Detection:
    def __init__(self, config):
        self.yolo = YOLO('yolov10m')
        self.config = config
        # Camera parameters to undistort and rectify images
        cv_file = cv2.FileStorage()
        cv_file.open(self.config.pathStereoMap, cv2.FileStorage_READ)

        self.map1_left = cv_file.getNode('stereoMapL_x').mat()
        self.map2_left = cv_file.getNode('stereoMapL_y').mat()
        self.map1_right = cv_file.getNode('stereoMapR_x').mat()
        self.map2_right = cv_file.getNode('stereoMapR_y').mat()

    def undistortion(self, file_pathL=None, file_pathR=None, iscap=False):
        # first read or capture the image
        if iscap:
            frames = imageCapture()
            if frames == 0:
                print("the task cannot be performed.....")
                return 0
        else:
            imgL = cv2.imread(file_pathL)
            imgR = cv2.imread(file_pathR)
            frames = [imgL, imgR]

        # preform the undistortion
        # Apply rectification maps to stereo images
        left_image_rectified = cv2.remap(
            frames[0], self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_image_rectified = cv2.remap(
            frames[1], self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return left_image_rectified, right_image_rectified

    def detection(self, file_pathL=None, file_pathR=None, iscap=False, isDisperity=False):
        undisLeftimg, undisRightimg = self.undistortion(
            file_pathL, file_pathR, iscap)
        # convert the BGR image to RGB
        undisLeftimg_rgb = cv2.cvtColor(undisLeftimg, cv2.COLOR_BGR2RGB)
        undisRightimg_rgb = cv2.cvtColor(undisRightimg, cv2.COLOR_BGR2RGB)
        templeftRgb = copy.deepcopy(undisLeftimg_rgb)
        temprightRgb = copy.deepcopy(undisRightimg_rgb)

        # predict the bbox and object using yolo
        prediction_left = self.yolo.predict(undisLeftimg_rgb, save=False)[0]
        prediction_right = self.yolo.predict(undisRightimg_rgb, save=False)[0]

        # decode the name and number class
        num2name = prediction_left.names

        # detect the predicted box
        l1 = len(prediction_left.boxes.cls)
        l2 = len(prediction_right.boxes.cls)

        # find the similar prediction from both prediction
        similar_ = np.zeros((l1, l2))
        left_box = prediction_left.boxes.xyxy.numpy()
        right_box = prediction_right.boxes.xyxy.numpy()
        for i, box1 in enumerate(left_box):
            similar_[i, :] = np.sum(np.abs(box1-right_box), axis=1)

        minBoxindex = np.argmin(similar_, axis=1)
        # shape of the image
        h, w = prediction_left.boxes.orig_shape
        leftBoxes = []
        rightBoxes = []
        leftCenters = []
        rightCenters = []
        objectName = []
        disperitys = []
        depths = []
        for i in range(l1):
            left_box1 = prediction_left.boxes.xyxyn[i]
            left_box1[[0, 2]] = left_box1[[0, 2]]*w
            left_box1[[1, 3]] = left_box1[[1, 3]]*h
            leftBoxes.append(left_box1)
            # right box
            right_box1 = prediction_right.boxes.xyxyn[minBoxindex[i]]
            right_box1[[0, 2]] = right_box1[[0, 2]]*w
            right_box1[[1, 3]] = right_box1[[1, 3]]*h
            rightBoxes.append(right_box1)

            # find the center
            left_centerX, left_centerY = (
                left_box1[0]+left_box1[2])//2, (left_box1[1]+left_box1[3])//2
            right_centerX, right_centerY = (
                right_box1[0]+right_box1[2])//2, (right_box1[1]+right_box1[3])//2
            leftCenters.append([left_centerX, left_centerY])
            rightCenters.append([right_centerX, right_centerY])
            objName = num2name[int(prediction_left.boxes.cls[i])]
            objectName.append(objName)

            # calculate the depth
            focal_pix, diperity, zDepth = depthCalculation(
                w, self.config.alpha, self.config.baseline, left_centerX, right_centerX)
            disperitys.append(diperity)
            depths.append(zDepth)

            # plot the image
            undisLeftimg_rgb, undisRightimg_rgb = plot_image(undisLeftimg_rgb, undisRightimg_rgb,
                                                             zDepth, objName, left_box1, right_box1)

        # Concatenate images horizontally
        combined_image = np.hstack((undisLeftimg_rgb, undisRightimg_rgb))
        # Display the combined image
        cv2.imshow('Pairwise Images', combined_image)

        # Wait for keypress to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        depthMap = 0
        if isDisperity:
            # generate the disperity map
            depthMap = generate_disperaty_map(templeftRgb, temprightRgb,
                                              focal_pix, self.config.baseline)
        return leftBoxes, rightBoxes, leftCenters, rightCenters, objectName, depths, depthMap
