import argparse
import datetime
import imutils
import time
import cv2

#binary output
motion = 0
ap = argparse.ArgumentParser()
#define minimum pixel size of region of frame to be considered actual motion
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(1)
time.sleep(0.25)
#initialize first frame
firstFrame = None

while cap.isOpened():
    ret, frame = cap.read()
    text = "Unoccupied"
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
    if firstFrame is None:
        firstFrame = gray

    #subtract frames
    frameDelta = cv2.absdiff(firstFrame, gray)
    #threshold delta to reveal regions with significan changes in pixel intensity
    #if delta < 25, disregard, else, count as motion
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    #dilate image(fill in holes with white) and find contours
    thresh = cv2.dilate(thresh, None, iterations=2)
    ( _,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        #if contour is small (too insignificant) then ignore it
        if cv2.contourArea(c) < args["min_area"]:
            motion = 0
        else:
            motion = 1
        """#draw box of movement
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        text = "Motion Detected"

        #optional time and date with status
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)"""
        cv2.imshow("Security Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(motion)
cap.release()
cv2.destroyAllWindows()
