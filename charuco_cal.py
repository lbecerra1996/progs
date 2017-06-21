import time
import cv2
import cv2.aruco as aruco
import numpy as np

#lengths in cm
square_length = 6.35
markerLength = 5.08
dictionary = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250) #AR tag dictionary
board = cv2.aruco.CharucoBoard_create(4,2,square_length,markerLength,dictionary)
img = board.draw((700*4,700*2))

cv2.imwrite('charuco.png',img)

#capture images to calibrate
cap = cv2.VideoCapture(0)

#create lists
allCorners = []
allIds = []
decimator = 0

for i in range(300):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary) #output: corners, ids,rejected imgpts

    print('loop: ',i)
    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board) #retval, charuco corners, charuco ids
        print("past res2")
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:# and decimator%3 == 0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

        #cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

    cv2.imshow('frame',gray)
    cv2.imwrite("cal.jpg", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator += 1

imsize = gray.shape
print(allCorners)
print(allIds)
#print(board)
#print(imsize)

try:
    a = cv2.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
    print(a)
    ret, mtx, dist, rvecs, tvecs = a
except:
    print("there was an error")
    cap.release()

cap.release()
cv2.destroyAllWindows()

#data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
#with open("calibration.yaml", "w") as f:
    #yaml.dump(data, f)
