from scipy.spatial import distance as dist
import datetime
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os
from motion_detect import detect_motion as detect_motion
from fume_hood_ar_logic import vertical_main as fume_hood_height


# threshold to determine if fume hood is "open enough" to trigger alarm
# sashHeight is a number between 0 and 1 indicating ratio of sash height / maximum height
HEIGHT_THRESHOLD = 50

# threshold to determine if there is enough motion for the sash to be in use
# motion is a number between 0 and 1 (with a higher value indicating more motion)
MOTION_THRESHOLD = 0.05

# amount of time the fume hood has to be open and without use before triggering the alarm
TIME_TO_ALARM = 30	# seconds

# minimum amount of time between two measurements
SLEEP_INTERVAL = 20

HEIGHT_DURATION = 2
MOTION_DURATION = 15

# initialize variable to keep track of time ellapsed between two measurements
# used for integrating power
prevTime = datetime.datetime.now()

# initialize variable to keep track of the last time the fume hood was in use
timeLastUsed = datetime.datetime.now()

date_string = str(timeLastUsed.date())

time_string = "{}-{}".format(str(timeLastUsed.hour), str(timeLastUsed.minute))

file_name = "fume_data_{}_{}.csv".format(date_string, time_string)

with open(file_name, "w") as f:
	f.write("Fume Hood Data Log: " + date_string + "\n")

# initialize video capture
cap = cv2.VideoCapture(0)

data = []

recording_start = datetime.datetime.now()
RECORDING_LENGTH = 60 # seconds

finished = 0
while not finished:
	# time elapsed (in seconds) since last measurement
	timeElapsed = (datetime.datetime.now() - prevTime).total_seconds()

	# update prevTime with the current time
	# not sure if timeElapsed and prevTime are in optimal code location
	# Crucial to make sure all time is accounted for (i.e. prevTime should immediately follow timeElapsed)
	prevTime = datetime.datetime.now()

	# Checking the degree to which the fume hood is open
	# NOTE: assume that this takes a non-trivial amount of time
	try:
		# TO DO: actually get sash state
		sashHeight = fume_hood_height(cap, HEIGHT_DURATION)
	except:
		print "Error computing sashHeight, defaults to 0"
		sashHeight = 0

	# Checking for motion
	# NOTE: assume that this takes a non-trivial amount of time
	try:
		# # TO DO: actually check for motion
		# first arg: video capture
		# second arg: duration for which to check for motion (in seconds)
		motion = detect_motion(cap, MOTION_DURATION)
	except:
		print "Error computing motion, defaults to 0"
		motion = 0

	# if there is motion (i.e. the fume hood is currently in use), update the value of timeLastUsed
	if motion > MOTION_THRESHOLD:
		timeLastUsed = datetime.datetime.now()

	# compute the time elapsed (in seconds) since the fume hood was last used
	timeSinceUse = (datetime.datetime.now() - timeLastUsed).total_seconds()

	# signal to turn alarm on(1)/off(0)
	# if the sash is open and hasn't been in use, turn on alarm
	alarm = 1 if (sashHeight > HEIGHT_THRESHOLD) and (timeSinceUse > TIME_TO_ALARM) else 0
	if alarm:
		print "ALARM ON"
		# TO DO: actually turn on alarm
	else:
		print "ALARM OFF"
		# TO DO: make sure alarm is off

	# TO DO: add info to data file
	# include sashState, motion, timeElapsed

	data.append((str(datetime.datetime.now()), timeElapsed, sashHeight, motion, "alarm: {}".format(alarm)))

	with open(file_name, "a") as f:
		data_string = ""
		for data_entry in data:
			data_string += ",".join([str(e) for e in data_entry]) + '\n'
		# clear data so that it doesn't get repeated in future writes
		data = []
		f.write(data_string)

	# wait until SLEEP_INTERVAL is over before making the next measurement
	# listen for 'q' as a signal to quit
	while (datetime.datetime.now() - prevTime).total_seconds() < SLEEP_INTERVAL:
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        finished = 1

	# since pressing 'q' doesn't seem to work, terminate program after RECORDING_LENGTH has elapsed
	if (datetime.datetime.now() - recording_start).total_seconds() > RECORDING_LENGTH:
		finished = 1

print(data)

cap.release()
cv2.destroyAllWindows()



