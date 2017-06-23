import numpy as np
import cv2
# import cv2.aruco as aruco
import glob
import yaml

criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# 0 for computer webcam, 1 for other webcam (e.g. Logitech webcam we intend to use)
cap = cv2.VideoCapture(0)

# capture and show mage fram until user presses 'q'
while cap.isOpened():
	ret, image = cap.read()
	cv2.imshow("Webcam Image", image)
	if (cv2.waitKey(1) & 0xFF) == ord('q'):		# if user presses 'q'
		cv2.imwrite("webcamim.png", image)		# save image for debugging/documentation
		break

# stop using webcam once we obtain picture for calibration
cap.release()

# convert image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find chessboard corners. note: watch out for board row/col dimensions!
ret, corners = cv2.findChessboardCorners(gray,(10,7),None)