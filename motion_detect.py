import argparse
import datetime
import imutils
import time
import cv2

#binary output
motion_list = []
ap = argparse.ArgumentParser()
#define minimum pixel size of region of frame to be considered actual motion
ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
time.sleep(0.25)
#initialize first frame
prevFrame = None
i = 0
while cap.isOpened() and i < 20:
    ret, frame = cap.read()
    start = time.time()
    #if no frame, end of video
    if not ret:
        break

    #resize frame, convert to grayscale, blur
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #21 x 21 region for gaussian smoothing average pixel intensity
    #smooths out high frequency noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    #initialize firstFrame
    if prevFrame is None:
        prevFrame = gray

    #subtract frames
    frameDelta = cv2.absdiff(prevFrame, gray)
    #threshold delta to reveal regions with significan changes in pixel intensity
    #if delta < 25, disregard, else, count as motion
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    #dilate image(fill in holes with white) and find contours
    thresh = cv2.dilate(thresh, None, iterations=2)
    ( _,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = 0
    for c in cnts:
        #if contour is small (too insignificant) then ignore it
        if cv2.contourArea(c) > args["min_area"]:
            motion = 1
            #print motion
    motion_list.append(motion)
    prevFrame = gray

    i += 1

print motion_list
cap.release()
cv2.destroyAllWindows()
