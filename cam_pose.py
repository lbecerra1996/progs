from scipy.spatial import distance as dist
import time
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os

# given a list of four points, returns a numpy array of them in counterclockwise order, starting with top-left
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

# given a list of corners(each of these being a numpy array of four corner points), return numpy array representing their average
def corners_avg(cornersList):
    numSamples = len(cornersList)

    # initialize sum
    sumSamples = np.zeros_like(cornersList[0])

    for sample in cornersList:
        sumSamples += sample

    return sumSamples/numSamples

def ids_avg(IDsList):
    numSamples = len(IDsList)
    numIDs = len(IDsList[0])

    avgSamples = [sum(IDsList[i][j][0] for i in range(numSamples))/numSamples for j in range(numIDs)]

    return avgSamples


# open yaml file containing calibration data
with open("calibration.yaml") as f:
    calibration_data = yaml.load(f)

# extract relevant information from calibration data
# convert to numpy arrays for usage in OpenCV/ARUCO functions
camera_matrix = np.asarray(calibration_data['camera_matrix'])
dist_coeff = np.asarray(calibration_data['dist_coeff'])

square_length = 6.35    # cm
markerLength = 5.08     # cm
# number of AR tags we're using
numTags = 2
dictionary = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250) #AR tag dictionary
board = cv2.aruco.CharucoBoard_create(4, 2, square_length, markerLength, dictionary)
arucoParams = aruco.DetectorParameters_create()
#img = board.draw((700*4,700*2))

# initialize lists to store all detected corners and IDs
allCorners = []
allIDs = []

# initialize video stream capture
cap = cv2.VideoCapture(0)       # 0 for default camera, 1 for external connection
# iterate over arbitrary amount of frames
for i in range(100):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
    # unclear: does res contain the same ordering for IDs and matching corners?
    tagCorners, tagIDs = res[0], res[1]

    # check that all IDs and corners were detected
    if tagCorners is not None and tagIDs is not None and len(tagIDs) == numTags:
        # For each corner (set of 4 points), sort its points in counterclockwise order, starting with top-left point
        # NOTE: sortedCorners is a list of numpy arrays, one for each tag's set of four points
        sortedCorners = [order_points(tagCorners[i][0]) for i in range(numTags)]

        # TO DO: make sure order of IDs matches order of corners

        # store corners and IDs
        allCorners.append(sortedCorners)
        allIDs.append(tagIDs)

        rvec, tvec = aruco.estimatePoseSingleMarkers(np.asarray(tagCorners), markerLength, camera_matrix, dist_coeff)

        # for visual debugging
        cv2.aruco.drawDetectedMarkers(gray, tagCorners, tagIDs)
        #gray1 = cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
        #gray2 = cv2.aruco.drawAxis(gray, np.asarray(camera_matrix), np.asarray(dist_coeff), rvec, tvec, 100)
    cv2.imshow("frame", gray)

cv2.imwrite("cal.jpg", gray)
# wait until user presses 'q' to continue
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# take average of all detected corners and IDs
avgCorners = corners_avg(allCorners)
print(avgCorners)
avgIDs = ids_avg(allIDs)
print(avgIDs)
numIds = len(avgIDs)    # should be equal to numTags
print (numIds)

# #compare y values of 2 tags to determine which is higher tag and lower tag
# if (corners[0][0][0][1]+corners[0][0][2][1]) > corners[1][0][0][1]+corners[1][0][2][1]:
#     lowerId = ids[0]
#     higherId = ids[1]
#     #find top corner y value of lower marker
#     y_val_low = [corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]]
#     y_low = min(y_val_low) #use min because horizontal axis is flipped
#     #find bottom corner y value of top marker
#     y_val_high = [corners[1][0][0][1], corners[1][0][1][1], corners[1][0][2][1], corners[1][0][3][1]]
#     y_high = max(y_val_high)
# else:
#     lowerId = ids[1]
#     higherId = ids[0]
#     #find top corner y value of lower marker
#     y_val_low = [corners[1][0][0][1], corners[1][0][1][1], corners[1][0][2][1], corners[1][0][3][1]]
#     y_low = min(y_val_low)
#     #find bottom corner y value of top marker
#     y_val_high = [corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]]
#     y_high = max(y_val_high)

# print 'lowerId:',lowerId
# print 'higherId:',higherId
# print 'bottom marker y value:', y_low
# print 'top marker y value:', y_high
# distance = y_low - y_high
# print distance

# for i in range(numIds):
#     print'markerId'+str(ids[i])+'='+str(corners[i])

cap.release()
cv2.destroyAllWindows()
