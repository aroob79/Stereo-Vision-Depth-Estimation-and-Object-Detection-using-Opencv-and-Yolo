# Stereo Vision Depth Estimation and Object Detection Using OpenCV and YOLOv10

## Overview
This project demonstrates object detection using YOLOv10m and estimates the depth of detected objects through stereo vision. The stereo vision camera is calibrated using OpenCV in Python. The calibration dataset includes 6 images for each camera (left and right), featuring a chessboard pattern to facilitate calibration.

Output Example:
![image](https://github.com/user-attachments/assets/6ab28549-b7fb-4628-9d58-3972008499ba)

## Pipeline
1. **Image Collection**: Capture calibration images, or use pre-existing calibration images, and save them in a designated folder.
2. **Calibration**: Perform camera calibration and then stereo calibration.
3. **Parameter Storage**: Save calibration parameters for future use.
4. **Object Detection and Undistortion**: Apply object detection and undistort new images or given images.
5. **Depth Estimation**: Calculate depth for detected bounding boxes.
6. **Depth Map Generation**: Generate depth maps using OpenCV.

## Dataset
Calibration images are from [this GitHub repository](https://github.com/niconielsen32/ComputerVision/tree/master/StereoVisionDepthEstimation).

## How to Use

1. **Install Required Libraries**:
   ```python
   numpy == 1.25.2
   opencv-python == 4.9.0
   ultralytics == 8.2.55
   ```

2. **Configuration Setup**:
   Configure the `config` class in the `main.py` file, setting values such as `chessBoard_size`, `focal_length`, `baseline`, `alpha`, `calibImagePath`, and `pathStereoMap`.

   ```python
   config = Config(chessBoard_size, focal_length, baseline, alpha, calibImagePath, pathStereoMap)
   ```

3. **Run the Program**:
   Execute `main.py` to perform calibration, object detection, and depth estimation.

## References
1. [Stereo Vision Tutorial - YouTube](https://www.youtube.com/watch?v=S-UHiFsn-GI&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo)
2. [Object Detection Tutorial - YouTube](https://www.youtube.com/watch?v=KOSS24P3_fY&list=PLCpB2LmtGbuel31gdKHSV_HBaZa2guc6Y)
