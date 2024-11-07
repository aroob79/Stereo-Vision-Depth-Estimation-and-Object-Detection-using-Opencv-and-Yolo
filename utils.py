import cv2
import numpy as np


def imageCapture(save=False, nameL="testL.png", nameR="testR.png"):
    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)
    if camL.isOpened() and camR.isOpened():
        _, frameL = camL.read()
        _, frameR = camR.read()
        print("image taken.....!")
        if save:
            cv2.imwrite(nameL, frameL)
            cv2.imwrite(nameR, frameR)
        cv2.destroyAllWindows()
        return [frameL, frameR]
    else:
        print('Failed to capture ...!')
        return 0


def depthCalculation(w, alpha, baseline, centerLefts, centerRights):
    f_pixel = (w * 0.5) / np.tan(alpha * 0.5 *
                                 np.pi/180)  # focal length in pixel
    # calculate the disperity

    diperity = centerLefts-centerRights  # disperity in pixel
    zDepth = (baseline*f_pixel)/diperity

    return f_pixel, diperity, zDepth


def plot_image(left_img, right_img, depth, label, leftbbox, rightbbox):
    color_values = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 0, 0),      # Black
        (255, 255, 255),  # White
        (128, 128, 128)  # Gray
    ]
    color1 = color_values[np.random.randint(0, len(color_values)-1, (1,))[0]]
    color2 = color_values[np.random.randint(0, len(color_values)-1, (1,))[0]]
    # text height position
    h1 = int(leftbbox[1])-20
    if h1 < 0:
        h1 = int(leftbbox[1])+20
    h2 = int(rightbbox[1])-20
    if h2 < 0:
        h2 = int(rightbbox[1])+20

    # Add text to both images
    cv2.putText(left_img, f'Object :{label} , depth :{depth:0.2f}', (int(leftbbox[0]), h1), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color1, 1, cv2.LINE_AA)
    cv2.putText(right_img, f'Object :{label} , depth :{depth:0.2f}', (int(rightbbox[0]), h2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color2, 1, cv2.LINE_AA)

    # Draw bounding box on each image
    cv2.rectangle(
        left_img, (int(leftbbox[0]), int(leftbbox[1])), (int(leftbbox[2]), int(leftbbox[3])), color1, 2)
    cv2.rectangle(
        right_img, (int(rightbbox[0]), int(rightbbox[1])), (int(rightbbox[2]), int(rightbbox[3])), color2, 2)

    return left_img, right_img


def generate_disperaty_map(left_image, right_image,
                           focal_length: "focal length in pixel",
                           baseline: "baseline in cm"):

    frame_right = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
    frame_left = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)

    # Initialize stereo matcher (StereoSGBM or StereoBM)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(frame_left, frame_right)

    # Calculate depth map
    baseline = baseline/100     # Example baseline in meters
    depth_map = (focal_length * baseline) / \
        (disparity + 1e-6)  # Avoid division by zero

    # Normalize the depth map to the range [0, 255] and convert to uint8 for display
    depth_map_normalized = cv2.normalize(
        depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = np.uint8(depth_map_normalized)

    return depth_image
