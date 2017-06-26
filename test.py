import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("webcam", gray)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):		# if user presses 'q'
		cv2.imwrite("webcam.png", gray)		# save image for debugging/documentation
		break

cap.release()
cv2.destroyAllWindows()
