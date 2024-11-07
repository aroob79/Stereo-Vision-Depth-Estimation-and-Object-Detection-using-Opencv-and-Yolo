class config:
    def __init__(self, chessBoard_size: "the number of col and row chessboard has",
                 focal_length: "Camera lense's focal length [mm]",
                 baseline: "Distance between the cameras [cm]",
                 alpha: "Camera field of view in the horisontal plane [degrees]",
                 calibImagepath: " path of the calibration image if available" = None,
                 pathStereoMap: "path of the saved stereo map" = 'stereoMap.xml'
                 ):
        self.chessBoard_size = chessBoard_size
        self.focal_length = focal_length
        self.baseline = baseline
        self.alpha = alpha
        self.calibImagepath = calibImagepath
        self.pathStereoMap = pathStereoMap
