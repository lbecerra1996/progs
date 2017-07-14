from scipy.spatial import distance as dist
import datetime
import cv2
import cv2.aruco as aruco
import numpy as np

# given a list of four points, returns a numpy array of them in counterclockwise order, starting with top-left
# NOTE: assumes pts has 4 elements
# TO DO: remove scipy dependency (i.e. write our own euclidean distance function)
# SOURCE: http://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
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

# given a list of corners(each of these being a numpy array of x corner points), return numpy array representing their average
# assumes corners are numpy arrays
def corners_avg(cornersList):
    # make sure it's not a null list
    if not cornersList: return None
    numSamples = len(cornersList)
    # initialize sum
    sumSamples = np.zeros_like(cornersList[0])
    # add all samples
    for sample in cornersList:
        sumSamples += sample
    # divide by number of samples to obtain average
    return sumSamples/numSamples

# given the coordinates of several corners, return their average coordinate
# assumes each corner is a numpy array
# i.e. given the coordinates of four corners, return the coordinates of the center of the geometric figure they form
def center_point(corners):
    # make sure it's not a null list
    if not corners: return None
    # sum all corners and divide by the number of corners to obtain average
    n = len(corners)
    sum_points = np.zeros_like(corners[0])
    for point in corners:
        sum_points += point
    return sum_points / n

# given a list of tags' corners, determine lowest x value
# NOTE: our interpretation assumes that lowest x value = left-most physical location
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
# NOTE: our interpretation assumes that highest x value = right-most physical location
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

# given a list of the corners of three (leftmost or rightmost) tags, return y value difference between middle and bottom tags
# TO DO: incorporate constants to account for distances from AR tags to borders
def measure_height(threeCorners, bot_to_top_distance):
	# sort the corners by increasing y value (i.e. decreasing height)
    topCorners, midCorners, botCorners = sorted(threeCorners, key=lambda corners: corners[0][1])
    # y-coordinate distance from top left corner of bottom AR tag to bottom left corner of top AR tag
    bot_to_top = botCorners[0][1] - topCorners[3][1]
    # y-coordinate distance from top left corner of bottom AR tag to bottom left corner of middle AR tag
    bot_to_mid = botCorners[0][1] - midCorners[3][1]

    return (bot_to_top_distance) * (bot_to_mid/bot_to_top)

def vertical_main(video_cap, duration=2, side_margin=3, botToTopDistance=100, numOnSides=3):
    squareLength = 6.35    # cm
    markerLength = 5.08     # cm
    # minimum number of AR tags to consider a frame valid
    numTagsThreshold = 2
    # distance between top left corner of bottom AR tag and bottom left corner of top AR tag
    BOT_TO_TOP_DISTANCE = botToTopDistance
    # arbitrary margin from side edge to qualify a point as being in the center region
    SIDE_MARGIN = side_margin
    AR_tag_dictionary = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250) #AR tag dictionary
    board = cv2.aruco.CharucoBoard_create(4, 2, squareLength, markerLength, AR_tag_dictionary)
    arucoParams = aruco.DetectorParameters_create()


    # keeps track of number of detected tags in each video frame/sample
    numDetected = []
    numLeft = []
    numRight = []

    # keep track of height measurements
    # list of tuples of the form (i, height)
    heightsLeft = []
    heightsRight = []

    marginsFixed = False

    startTime = datetime.datetime.now()

    # iterate over arbitrary amount of frames
    while video_cap.isOpened() and (datetime.datetime.now() - startTime).total_seconds() < duration:
        ret, frame = video_cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, AR_tag_dictionary, parameters=arucoParams)
        # unclear: does res contain the same ordering for IDs and matching corners?
        tagCorners = res[0]
        numTags = len(tagCorners) if tagCorners is not None else 0
        numDetected.append(numTags)

        # check that at least one corner was detected
        if tagCorners is not None and len(tagCorners) > 0:
            # For each corner (set of 4 points), sort its points in counterclockwise order, starting with top-left point
            # NOTE: sortedCorners is a list of numpy arrays, one for each tag's set of four points
            sortedCorners = [order_points(tagCorners[j][0]) for j in range(numTags)]

            if not marginsFixed:
                # difference between x values of top-left and top-right corners of one tag
                # is a reasonable approximate for the AR tags' edge size
                edgeSize = abs(sortedCorners[0][0][0] - sortedCorners[0][1][0])
                SIDE_MARGIN = edgeSize * side_margin
                marginsFixed = True

            # collect information on the leftmost tags, to determine degree of vertical openness
            lowestXval =  lowest_x_val(sortedCorners)
            # recall that last argument: isLeft
            leftTags = edge_tags(sortedCorners, lowestXval, SIDE_MARGIN, True)
            numLeftTags = len(leftTags)
            numLeft.append(numLeftTags)
            if numLeftTags == numOnSides:
                heightLeft = measure_height(leftTags, BOT_TO_TOP_DISTANCE)
                heightsLeft.append(heightLeft)

            # collect information on the rightmost tags, to determine degree of vertical openness (identical to leftmost tags process)
            highestXval =  highest_x_val(sortedCorners)
            rightTags = edge_tags(sortedCorners, highestXval, SIDE_MARGIN, False)
            numRightTags = len(rightTags)
            numRight.append(numRightTags)
            if numRightTags == numOnSides:
                heightRight = measure_height(rightTags, BOT_TO_TOP_DISTANCE)
                heightsRight.append(heightRight)

    try:
    	leftAvg = float(sum(heightsLeft))/len(heightsLeft)
    except:
    	leftAvg = -1
    try:
    	rightAvg = float(sum(heightsRight))/len(heightsRight)
    except:
    	rightAvg = -1
    print "Left height average: " + str(leftAvg)
    print "Right height average: " + str(rightAvg)
    # arbitrary choice: return the maximum measured height (in theory they should be approximately equal)
    return max(leftAvg, rightAvg)

