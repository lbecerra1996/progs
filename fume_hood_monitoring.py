# Author: Juan Ferrua

from scipy.spatial import distance as dist
import datetime
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import os
from motion_detect import detect_motion
from fume_hood_ar_logic import vertical_main as fume_hood_height

class FumeHood():
	# amount of time (sec) to be spent detecting sash height for each measurement
	HEIGHT_DURATION = 2
	# amount of time (sec) to be spent detecting motion for each measurement (when alarm is OFF)
	MOTION_DURATION = 15
	# amount of time (sec) to be spent detecting motion for each measurement (when alarm is ON)
	ALARM_MOTION_DURATION = 2
	# make a measurement each LOG_FREQ seconds
	LOG_FREQ = 20
	# minimum time of inactivity (in sec) required to trigger alarm when sash is left open
	TIME_TO_ALARM = 30

	def __init__(self, height_threshold=50, motion_threshold=0.05):
		self.HEIGHT_THRESHOLD = height_threshold
		self.MOTION_THRESHOLD = motion_threshold

		# Create file with the date for storing fume hood data
		self.DATE_STRING = str(datetime.datetime.now().date())
		self.FILE_NAME = "fume_data_" + self.DATE_STRING + ".csv"

		with open(self.FILE_NAME, "w") as f:
			f.write("Fume Hood Data Log: " + self.DATE_STRING + "\n")

	def run(self, recording_length=-1):
		# recording_length: time (in sec) to spend recording data.
		# -1: record for indefinite amount of time, until terminated by pressing 'q' or powering off the system

		# initialize video capture
		cap = cv2.VideoCapture(0)

		data = []
		recording_start = datetime.datetime.now()
		prevTime = datetime.datetime.now()
		timeLastUsed = datetime.datetime.now()
		finished = False

		while not finished:
			timeStep = (datetime.datetime.now() - prevTime).total_seconds()

			try:
				sashHeight = fume_hood_height(cap, HEIGHT_DURATION, MAX_HEIGHT)
			except:
				print "Error computing sashHeight, defaults to 0"
				sashHeight = 0

			try:
				motion_time = ALARM_MOTION_DURATION if alarm_signal else MOTION_DURATION
				motion = detect_motion(cap, motion_time)
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
			alarm_signal = 1 if (sashHeight > HEIGHT_THRESHOLD) and (timeSinceUse > TIME_TO_ALARM) else 0
			self.alarm(alarm_signal)

			data.append((str(datetime.datetime.now()), timeElapsed, sashHeight, motion, "alarm: {}".format(alarm_signal)))
			# optional: if we wanted to store data every X iterations, we would incorporate that feature here
			if True:
				self.write(data)
				data = []

			# wait until LOG_FREQ is over before making the next measurement
			# listen for 'q' as a signal to quit
			# NOTE: if alarm is on, bypass this wait time to avoid taking too long to turn off alarm
			if not alarm_signal:
				while (datetime.datetime.now() - prevTime).total_seconds() < LOG_FREQ:
				    if cv2.waitKey(1) & 0xFF == ord('q'):
				        finished = 1

			# if we are recording for a limited amount of time and that time has elapsed, we are finished
			if (recording_length > -1) and (datetime.datetime.now() - recording_start).total_seconds() > recording_length:
				finished = 1

		# turn off alarm
		self.alarm(0)
		# close video capture and cv2 windows
		cap.release()
		cv2.destroyAllWindows()

	def alarm(self, alarm_signal):
		if alarm_signal:
			print "ALARM ON"
		else:
			print "ALARM OFF"

	def write(self, data):
		data_string = ""
		for data_entry in data:
			data_string += ",".join([str(e) for e in data_entry]) + '\n'
		with open(self.FILE_NAME, "a") as f:
			f.write(data_string)

if __name__ == "__main__":
	# TO DO: give user the option to specify height_threshold, motion_threshold, recording_length from command line
	fume_hood = FumeHood()
	fume_hood.run()
