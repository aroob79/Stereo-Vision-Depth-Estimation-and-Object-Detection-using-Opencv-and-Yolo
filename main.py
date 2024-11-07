from configaration import config
from detection import Detection
from cameraCalibration import stereoCalibration

# first set the important parameter
chessBoard_size = (9, 6)
focal_length = 8
baseline = 9
alpha = 56.6
calibImagepath = r'E:\python\basic_code\automate the boring stuf\CAMERA CALIBRATION\sterio_img'
pathStereoMap = r'E:\python\basic_code\automate the boring stuf\CAMERA CALIBRATION\camera_param\stereoMap.xml'

config = config(chessBoard_size, focal_length, baseline,
                alpha, calibImagepath, pathStereoMap)

# perform the calibration
calibration = False
if calibration:
    stcal = stereoCalibration(config)

    _ = stcal.calibrate()

# detection and depth calculation
detectn = Detection(config)
leftBoxes, rightBoxes, leftCenters, rightCenters, objectName, depths, depthMap = detectn.detection(
    file_pathL=r'E:\python\basic_code\automate the boring stuf\CAMERA CALIBRATION\sterio_img\left_camera\imageL4.png',
    file_pathR=r'E:\python\basic_code\automate the boring stuf\CAMERA CALIBRATION\sterio_img\right_camera\imageR4.png', isDisperity=False)
