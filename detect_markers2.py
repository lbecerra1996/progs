import time
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml

with open("calibration.yaml") as yaml_file:
    test_data = yaml.load(yaml_file)
print test_data

#lengths in cm
square_length = 6.35
markerLength = 5.08
dictionary = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250) #AR tag dictionary
board = cv2.aruco.CharucoBoard_create(4,2,square_length,markerLength,dictionary)
#img = board.draw((700*4,700*2))

#cv2.imwrite('charuco.png',img)

#capture images to calibrate
cap = cv2.VideoCapture(0)

#create lists
allCorners = []
allIds = []
decimator = 0

for i in range(100):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary) #output: corners, ids,rejected imgpts

    if res[0] is not None:
        allCorners.append(res[0])

    if res[1] is not None:
        allIds.append(res[1])

        cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

    cv2.imshow('frame',gray)
    cv2.imwrite("cal.jpg", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator += 1

print(allCorners)
print(allIds)

cap.release()
cv2.destroyAllWindows()
