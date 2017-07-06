import time
import cv2

cap = cv2.VideoCapture(0)
i = 0

while True:
	try:
	    ret, frame = cap.read()
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    cv2.imwrite("July5_sample_no_" + str(i) + ".png", gray)

	    # sleep for 10 seconds
	    time.sleep(10)

	    i += 1

	    # terminate the process after 5000 samples
	    # equivalent to approximately 14 hours
	    if i == 5000:
	    	break

	except:
		continue

cap.release()
cv2.destroyAllWindows()