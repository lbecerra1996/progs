import numpy as np
import cv2
# import cv2.aruco as aruco
import glob
import yaml

retCorners, retCalibrate = False, False

# not sure where this is coming from
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# 0 for computer webcam, 1 for other webcam (e.g. Logitech webcam we intend to use)
cap = cv2.VideoCapture(0)

# capture and show mage fram until user presses 'q'
while cap.isOpened():
	image = cap.read()[1]
	cv2.imshow("Webcam Image", image)
	if (cv2.waitKey(1) & 0xFF) == ord('q'):		# if user presses 'q'
		cv2.imwrite("webcamim.png", image)		# save image for debugging/documentation
		break

# stop using webcam once we obtain picture for calibration
cap.release()

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print gray.shape

i = 1
while i > 0:
	if i == 1:
		cv2.imshow("Gray image", gray)
	i += 1
	if (cv2.waitKey(1) & 0xFF) == ord('r'):
		i = 0

# find chessboard corners. note: watch out for board row/col dimensions!
retCorners, corners = cv2.findChessboardCorners(gray, (9,6), None)
# print "Ret after findChessboardCorners"
# print retCorners

if retCorners == True:
	objpoints.append(objp)

	# not sure where (11,11) comes from
	cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
	imgpoints.append(corners)
	
	#draw corners
	cv2.drawChessboardCorners(image, (10,7), corners, True)

	#UNDISTORTION + CAMERA MATRICES

	# print(objpoints.__len__())
	# print(imgpoints.__len__())

	#camera matrices
	# Note: retCalibrate is (at least sometimes) a double and not a boolean (what we expected)
	retCalibrate, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	h, w = image.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

	# undistorted version of webcam image we used for calibration
	dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

	# Save calibrated image
	cv2.imwrite('calibration.png', dst)

else:
	print "Error, could not identify checkerboard corners. \n \
	Please try again and make sure checkerboard is within camera frame."

# Function that mutates image by drawing lines to show calibration effect
# NOTE: mutates img
def draw(img, corners, imgpoints):
	corner = tuple(corners[0].ravel())
	img = cv2.line(img, corner, tuple(imgpoints[0].ravel()), (255,0,0),5)
	img = cv2.line(img, corner, tuple(imgpoints[1].ravel()), (0,255,0),5)
	img = cv2.line(img, corner, tuple(imgpoints[2].ravel()), (0,0,255),5)
	return img

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# for some reason, this is a double instead of a boolean
if retCalibrate > 0:
	cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
	rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)[1:]

	imgpoints, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	
	#draw corners
	image = draw(image, corners, imgpoints)
	cv2.imshow('image',image)
	cv2.waitKey(0)

	# saving results to a YAML file
	data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist(), \
			'rvecs': np.asarray(rvecs).tolist(), 'tvecs': np.asarray(tvecs).tolist()}
	with open("calibration.yaml", "w") as f:
		yaml.dump(data,f)

else:
	print "Error, calibration failed. Please try again."


cv2.destroyAllWindows()