import numpy as np
import cv2
import glob
import yaml

criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints=[] #3d points in real worls space
imgpoints=[] #2d points in image plane

images='/home/juangelo/Downloads/cb.jpg'

img= cv2.imread(images)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)

#find corners
ret, corners = cv2.findChessboardCorners(gray,(7,6),None)
#print(corners)

if ret == True:
	objpoints.append(objp)


	cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	imgpoints.append(corners)
	
	#draw corners
	cv2.drawChessboardCorners(img,(7,6), corners,ret)
	#print(img.shape)
	#cv2.imshow('img',img)
	#cv2.waitKey(0)

#cv2.destroyAllWindows()

print(objpoints.__len__())
print(imgpoints.__len__())

#UNDISTORTION + CAMERA MATRICES

#camera matrices
ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
img=cv2.imread(images)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#cv2.undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

#crop
"""x,y,w,h = roi
dst=dst[y:y+h, x:x+w]"""
cv2.imwrite('calibresult.png', dst)

ret, corners = cv2.findChessboardCorners(gray,(7,6),None)

def draw(img, corners, imgpoints):
	corner= tuple(corners[0].ravel())
	img = cv2.line(img, corner, tuple(imgpoints[0].ravel()), (255,0,0),5)
	img = cv2.line(img, corner, tuple(imgpoints[1].ravel()), (0,255,0),5)
	img = cv2.line(img, corner, tuple(imgpoints[2].ravel()), (0,0,255),5)
	return img

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

print('cool')
print(ret)

if ret == True:
	cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	print(corners)
	rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)[1:]

	imgpoints, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	
	#draw corners
	img = draw(img,corners,imgpoints)
	cv2.imshow('img',img)
	cv2.waitKey(0)

	# saving results to a YAML file
	data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
	with open("calibration.yaml", "w") as f:
		yaml.dump(data, f)


cv2.destroyAllWindows()