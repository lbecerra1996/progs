from scipy.spatial import distance as dist
import time
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os

square_length = 6.35    # cm
markerLength = 5.08     # cm

# TO DO: add try-except mechanisms!

# given a list of four points, returns a numpy array of them in counterclockwise order, starting with top-left
# NOTE: assumes pts has 4 elements
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

# given the coordinates of several corners, return their average coordinate
def avg_corner(corners):
    n = len(corners) if corners is not None else 0
    if n == 0:
        return None
    sum_points = np.zeros_like(corners[0])
    for point in corners:
        sum_points += point
    return sum_points / n

def ids_avg(IDsList):
    numSamples = len(IDsList)
    numIDs = len(IDsList[0])

    avgSamples = [sum(IDsList[i][j][0] for i in range(numSamples))/numSamples for j in range(numIDs)]

    return avgSamples

# given two tag corners, return 0 if the first tag has the lowest y value, else 1
# NOTE: assumes lower y value = physically higher location
# NOTE: assumes y coordinate is the second value in a 2d-coordinate pair
def id_above(corners):
    ymin0 = min(corners[0][i][1] for i in range(4))
    ymin1 = min(corners[1][i][1] for i in range(4))

    return 0 if ymin0 < ymin1 else 1

# given a list of tags' corners, determine lowest x value
# NOTE: assumes lowest x value = left-most physical location
# NOTE: assumes x coordinate is the first value in a 2d-coordinate pair
def lowest_x_val(cornersList):
    minX = 10000
    for corners in cornersList:
        for point in corners:
            x = point[0]
            if x < minX:
                minX = x
    return minX

# given a list of tags' corners, determine highest x value
# NOTE: assumes highest x value = right-most physical location
# NOTE: assumes x coordinate is the first value in a 2d-coordinate pair
def highest_x_val(cornersList):
    maxX = 0
    for corners in cornersList:
        for point in corners:
            x = point[0]
            if x > maxX:
                maxX = x
    return maxX

# given a list of tags' corners, x value and margin, and a boolean indicating left or right,
# outputs the corners of tags in the left or right edge (depending on value of isLeft)
def edge_tags(cornersList, x_val, error_margin, isLeft):
    edgeTags = []
    for corners in cornersList:
        for point in corners:
            x = point[0]
            left = (x < x_val + error_margin)
            right = (x > x_val - error_margin)
            if (isLeft and left) or (not isLeft and right):
                edgeTags.append(corners)
                break
    return edgeTags

# given a list of tags' corners, determine highest y value
# NOTE: assumes higher y value = lower physical location
# NOTE: assumes y coordinate is the second value in a 2d-coordinate pair
def highest_y_val(cornersList):
    maxY = 0
    for corners in cornersList:
        for point in corners:
            y = point[1]
            if y > maxY:
                maxY = y
    return maxY

# given a list of tags' corners, a y value, and an error margin, return number of tags with at least one 
# y coordinate above y_val - error_margin (physically: is less than error_margin above y_val)
def num_bottom_tags(cornersList, y_val, error_margin):
    numBottom = 0
    for corners in cornersList:
        for point in corners:
            y = point[1]
            if y > y_val - error_margin:
                numBottom += 1
                break
    return numBottom
 
# given a list of the corners of three (leftmost or rightmost) tags, return y value difference between middle and bottom tags
# TO DO: incorporate constants to account for distances from AR tags to borders
def measure_height(threeCorners, bot_to_top_distance):
    topCorners, midCorners, botCorners = sorted(threeCorners, key=lambda corners: corners[0][1])

    bot_to_top = botCorners[0][1] - topCorners[3][1]
    bot_to_mid = botCorners[0][1] - midCorners[3][1]

    return (bot_to_top_distance) * (bot_to_mid/bot_to_top)

# given a list of tags' corners, sizes of side and bottom margins, return list of coordinates of the tags in the middle region
# NOTE: assumes higher y value = lower physical location, higher x value = righter-most physical location
# NOTE: assumes (x,y) 2d-coordinate system
def middle_tags(cornersList, side_margin, bot_margin):
    x_low, x_high, y_high = lowest_x_val(cornersList), highest_x_val(cornersList), highest_y_val(cornersList)
    middleTags = []
    for corners in cornersList:
        avgCorner = avg_corner(corners)
        x, y = avgCorner
        isHighEnough = (y < y_high - bot_margin)
        isRightEnough = (x > x_low + side_margin)
        isLeftEnough = (x < x_high - side_margin)
        isInCenter = isHighEnough and isRightEnough and isLeftEnough
        if isInCenter:
            middleTags.append(corners)
    return middleTags

# given a list of tags' corners, returns list horizontal pairwise distances between tags
# NOTE: assumes x coordinate is first coordinate
def horizontal_distances(cornersList):
    # extract the x_values of each tag's top-left corner
    x_values = map(lambda corners: corners[0][0], cornersList)
    # sort the x values in increasing order
    sorted_x = sorted(x_values)
    # determine horizontal distances between tags
    dists = [(sorted_x[i+1] - sorted_x[i]) for i in range(len(sorted_x) - 1)]
    return dists


def vertical_main(video_cap, duration=2, botToTopDistance=100, numOnSides=3):
    # open yaml file containing calibration data
    with open("calibration.yaml") as f:
        calibration_data = yaml.load(f)

    # extract relevant information from calibration data
    # convert to numpy arrays for usage in OpenCV/ARUCO functions
    camera_matrix = np.asarray(calibration_data['camera_matrix'])
    dist_coeff = np.asarray(calibration_data['dist_coeff'])

    # number of AR tags we're using
    numTagsThreshold = 2
    # arbitrary error margin for determining if a tag counts as a bottom tag
    errorMargin = 5
    # distance between top left corner of bottom AR tag and bottom left corner of top AR tag
    botToTopDistance = 100
    # arbitrary margin from side edge to qualify a point as being in the center region
    sideMargin = 25
    # arbitrary margin from bottom edge to qualify a point as being in the center region
    botMargin = 25
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250) #AR tag dictionary
    board = cv2.aruco.CharucoBoard_create(4, 2, square_length, markerLength, dictionary)
    arucoParams = aruco.DetectorParameters_create()
    #img = board.draw((700*4,700*2))

    # initialize lists to store all detected corners and IDs
    allCorners = []
    allIDs = []
    # keeps track of number of detected tags in each video frame/sample
    numDetected = []
    # keeps track of number of bottom tags detected in each valid video frame / sample
    numBottom = []

    numLeft = []
    numRight = []
    numMiddle = []

    # keep track of height measurements
    # list of tuples of the form (i, height)
    heightsLeft = []
    heightsRight = []

    marginsFixed = False

    # iterate over arbitrary amount of frames
    while video_cap.isOpened() and (datetime.datetime.now() - startTime).total_seconds() < duration:
        ret, frame = video_cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
        # unclear: does res contain the same ordering for IDs and matching corners?
        tagCorners, tagIDs = res[0], res[1]
        numTags = len(tagIDs) if tagIDs is not None else 0
        numDetected.append(numTags)

        # check that all IDs and corners were detected
        if tagCorners is not None and tagIDs is not None and numTags >= numTagsThreshold:
            # For each corner (set of 4 points), sort its points in counterclockwise order, starting with top-left point
            # NOTE: sortedCorners is a list of numpy arrays, one for each tag's set of four points
            sortedCorners = [order_points(tagCorners[j][0]) for j in range(numTags)]

            if not marginsFixed:
                # difference between x values of top-left and top-right corners of one tag
                # is a reasonable approximate for the AR tags' edge size
                edgeSize = abs(sortedCorners[0][0][0] - sortedCorners[0][1][0])
                errorMargin = edgeSize / 2
                print("new error margin: " + str(errorMargin))
                sideMargin, botMargin = (edgeSize * 1.5, edgeSize * 1.5)
                print("new side and bottom margin: " + str(sideMargin))
                marginsFixed = True

            # collect information on the leftmost tags, to determine degree of vertical openness
            lowestXval =  lowest_x_val(sortedCorners)
            leftTags = edge_tags(sortedCorners, lowestXval, errorMargin, isLeft=True)
            numLeftTags = len(leftTags)
            numLeft.append(numLeftTags)
            if numLeftTags == numOnSides:
                heightLeft = measure_height(leftTags, botToTopDistance)
                heightsLeft.append((i, heightLeft))

            # collect information on the rightmost tags, to determine degree of vertical openness (identical to leftmost tags process)
            highestXval =  highest_x_val(sortedCorners)
            rightTags = edge_tags(sortedCorners, highestXval, errorMargin, isLeft=False)
            numRightTags = len(rightTags)
            numRight.append(numRightTags)
            if numRightTags == numOnSides:
                heightRight = measure_height(rightTags, botToTopDistance)
                heightsRight.append((i, heightRight))

            # store corners and IDs (not sure if necessary)
            allCorners.append(sortedCorners)
            allIDs.append(tagIDs)

    leftAvg = float(sum(heightsLeft))/len(heightsLeft) if heightsLeft else -1
    rightAvg = float(sum(heightsRight))/len(heightsRight) if heightsRight else -1
    print "Left height average: " + str(leftAvg)
    print "Right height average: " + str(rightAvg)
    return max(leftAvg, rightAvg)
# -----------------------------------------------------------------------------------------------

# iterate over arbitrary amount of frames
while cap.isOpened() and (datetime.datetime.now() - startTime).total_seconds() < duration:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
    # unclear: does res contain the same ordering for IDs and matching corners?
    tagCorners, tagIDs = res[0], res[1]
    numTags = len(tagIDs) if tagIDs is not None else 0
    numDetected.append(numTags)

    # check that all IDs and corners were detected
    if tagCorners is not None and tagIDs is not None and numTags >= numTagsThreshold:
        # For each corner (set of 4 points), sort its points in counterclockwise order, starting with top-left point
        # NOTE: sortedCorners is a list of numpy arrays, one for each tag's set of four points
        sortedCorners = [order_points(tagCorners[j][0]) for j in range(numTags)]

        if not marginsFixed:
            # difference between x values of top-left and top-right corners of one tag
            # is a reasonable approximate for the AR tags' edge size
            edgeSize = abs(sortedCorners[0][0][0] - sortedCorners[0][1][0])
            errorMargin = edgeSize / 2
            print("new error margin: " + str(errorMargin))
            sideMargin, botMargin = (edgeSize * 1.5, edgeSize * 1.5)
            print("new side and bottom margin: " + str(sideMargin))
            marginsFixed = True

        # count the number of tags at the bottom (for motion detection purposes)
        highestYval = highest_y_val(sortedCorners)
        numBottomTags = num_bottom_tags(sortedCorners, highestYval, errorMargin)
        numBottom.append(numBottomTags)

        # collect information on the leftmost tags, to determine degree of vertical openness
        lowestXval =  lowest_x_val(sortedCorners)
        leftTags = edge_tags(sortedCorners, lowestXval, errorMargin, isLeft=True)
        numLeftTags = len(leftTags)
        numLeft.append(numLeftTags)
        if numLeftTags == 3:
            heightLeft = measure_height(leftTags, botToTopDistance)
            heightsLeft.append((i, heightLeft))

        # collect information on the rightmost tags, to determine degree of vertical openness (identical to leftmost tags process)
        highestXval =  highest_x_val(sortedCorners)
        rightTags = edge_tags(sortedCorners, highestXval, errorMargin, isLeft=False)
        numRightTags = len(rightTags)
        numRight.append(numRightTags)
        if numRightTags == 3:
            heightRight = measure_height(rightTags, botToTopDistance)
            heightsRight.append((i, heightRight))

        # collect information about the tags on the center reigon of the board, to determine degree of horizontal openness
        middleTags = middle_tags(sortedCorners, sideMargin, botMargin)
        numMiddleTags = len(middleTags)
        numMiddle.append(numMiddleTags)
        # assuming there are 4 tags in the middle section of the fume hood
        if numMiddleTags == 4:
            # calculate horizontal distances between tags
            d_1, d_2, d_3 = horizontal_distances(middleTags)  # supposed to be a list of three values

            # TO DO: add method to determine degree of openness based on these distances
            # might need to know edge coordinates too? (and those distances too)

            #TO DO: store this information in a dedicated list similar to the ones for height measurements

        # TO DO: make sure order of IDs matches order of corners
        # edit: not sure if this is relevant/ important

        # store corners and IDs (not sure if necessary)
        allCorners.append(sortedCorners)
        allIDs.append(tagIDs)

        # rvec, tvec = aruco.estimatePoseSingleMarkers(np.asarray(tagCorners), markerLength, camera_matrix, dist_coeff)

        # # for visual debugging
        # cv2.aruco.drawDetectedMarkers(gray, tagCorners, tagIDs)
        #gray1 = cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
        #gray2 = cv2.aruco.drawAxis(gray, np.asarray(camera_matrix), np.asarray(dist_coeff), rvec, tvec, 100)
    cv2.imshow("frame", gray)

cv2.imwrite("cal.jpg", gray)
# wait until user presses 'q' to continue
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("number of tags identified in each frame:\n" + str(numDetected))
# take average of all detected corners and IDs
print("Number of valid frames: " + str(len(allCorners)))
print("Number of bottom tags in each valid frame:\n" + str(numBottom))
print("Number of leftmost tags in each valid frame:\n" + str(numLeft))
print("Left-measured heights:\n" + str(heightsLeft))
print("Number of rightmost tags in each valid frame:\n" + str(numRight))
print("Right-measured heights:\n" + str(heightsRight))
print("Number of middle tags in each valid frame:\n" + str(numMiddle))
# avgCorners = corners_avg(allCorners)
# print(avgCorners)
# avgIDs = ids_avg(allIDs)
# print(avgIDs)
# numIds = len(avgIDs)    # should be equal to numTags
# print (numIds)

# id_above, id_below = (avgIDs[0], avgIDs[1]) if id_above(avgCorners) == 0 else (avgIDs[1], avgIDs[0])

# print("id_above: " + str(id_above))
# print("id_below: " + str(id_below))
# initialize video stream capture
cap = cv2.VideoCapture(0)       # 0 for default camera, 1 for external connection
cap.release()
cv2.destroyAllWindows()

