# Author: Juan Ferrua

from scipy.spatial import distance as dist
import datetime
import cv2
import cv2.aruco as aruco
import numpy as np
from motion_detect import detect_motion as get_motion
from fume_hood_ar_logic import vertical_main as fume_hood_height
import gpio
import time

class FumeHood():
    # amount of time (sec) to be spent detecting sash height
    HEIGHT_DURATION = 2
    # amount of time (sec) to be spent detecting motion (when alarm is OFF)
    MOTION_DURATION = 3
    # amount of time (sec) to be spent detecting motion (when alarm is ON)
    # ALARM_MOTION_DURATION = 2
    # make a measurement each LOG_FREQ seconds (when alarm is OFF)
    # LOG_FREQ = 20
    # min. time of inactivity (in sec) to trigger alarm when sash is left open
    TIME_TO_ALARM = 40

    def __init__(self, height_threshold=50, motion_threshold=0.05, 
                side_margin=3):
        self.HEIGHT_THRESHOLD = height_threshold
        self.MOTION_THRESHOLD = motion_threshold
        self.SIDE_MARGIN = side_margin

        # Create file with the date for storing fume hood data
        self.START_TIME = datetime.datetime.now()
        # format: fume_data_yyyy-mm-dd_hh-mm.csv
        self.FILE_NAME = "fume_data_{}_{}-{}.csv".format(self.START_TIME.date(),
                                                        self.START_TIME.hour, 
                                                        self.START_TIME.minute)

        with open(self.FILE_NAME, "w") as f:
            headerString = "Fume Hood Data Log: " + str(self.START_TIME) + "\n"
            headerString += "Date, Time Step, Left Height, Right Height, " + \
                            "Max Height, Motion, Alarm\n"
            f.write(headerString)

    def run(self, recording_length=-1):
        # recording_length: time (in sec) to spend recording data.
        # -1: record for indefinite amount of time
        timeIsFinite = recording_length > -1

        # initialize video capture
        cap = cv2.VideoCapture(0)

        data = []
        recording_start = datetime.datetime.now()
        prevTime = datetime.datetime.now()
        timeLastUsed = datetime.datetime.now()
        finished = False
        alarm_signal = 0
        # recent_alarm = False
        # index to keep track of when to write data to file
        i_write = 0

        #gpio stuff
        gpio.setup(57, gpio.OUT)
        gpio.set(57, 0)

        while not finished:
            timeStep = (datetime.datetime.now() - prevTime).total_seconds()

            prevTime = datetime.datetime.now()

            # obtain sashHeight: (left height, right height, max height)
            try:
                sashHeight = fume_hood_height(cap, 
                        duration=FumeHood.HEIGHT_DURATION,
                        side_margin=self.SIDE_MARGIN)
            except:
                print "Error computing sashHeight, values default to 0"
                sashHeight = (0, 0, 0)

            try:
                # if alarm_signal:
                #     motion_time = FumeHood.ALARM_MOTION_DURATION
                # else:
                #     motion_time = FumeHood.MOTION_DURATION
                # motion = get_motion(cap, motion_time)
                motion = get_motion(cap, FumeHood.MOTION_DURATION)
            except:
                print "Error computing motion, defaults to 0"
                motion = 0

            # if there is motion, update the value of timeLastUsed
            if motion > self.MOTION_THRESHOLD:
                timeLastUsed = datetime.datetime.now()

            # time elapsed (in sec) since the fume hood was last used
            timeInactive = datetime.datetime.now() - timeLastUsed
            timeSinceUse = timeInactive.total_seconds()

            # signal to turn alarm on(1)/off(0)
            # if the sash is open and hasn't been in use, turn on alarm
            isOpen = sashHeight[2] > self.HEIGHT_THRESHOLD
            isInactive = timeSinceUse > FumeHood.TIME_TO_ALARM
            alarm_signal = 1 if isOpen and isInactive else 0
            self._alarm(alarm_signal)

            data_entry = (str(datetime.datetime.now()), timeStep, \
                        sashHeight[0], sashHeight[1], sashHeight[2], \
                        motion, alarm_signal)
            data.append(data_entry)
            # optional: if we wanted to store data every X iterations, 
            # we would incorporate that feature here
            i_write += 1
            if (i_write % 5) == 0:
                self._write(data)
                data = []

            # wait until LOG_FREQ is over before making the next measurement
            # NOTE: only if the alarm is off
            # NOTE: not right after an alarm has been off
            # if not alarm_signal:
            #     # find the time remaining (sec) until LOG_FREQ
            #     timeSoFar = datetime.datetime.now() - prevTime
            #     timeRemaining = self.LOG_FREQ - timeSoFar.total_seconds()
            #     # wait until timeRemaining has elapsed
            #     # exception: when there has been a recent alarm (was causing a bug)
            #     if not recent_alarm:
            #         time.sleep(timeRemaining)
            #     else:
            #         # by the next iteration, the last alarm will not have been recent
            #         recent_alarm = False
            # else:
            #     # if alarm_signal is 1, there is a recent alarm
            #     recent_alarm = True

            # if recording time is finite and that time has elapsed, stop
            timeElapsed = datetime.datetime.now() - recording_start
            timeOver = timeElapsed.total_seconds() > recording_length
            if timeIsFinite and timeOver:
                finished = True

        # make sure alarm is off
        self._alarm(0)
        # make sure any remaining data is written to the file
        self._write(data)
        # close video capture and cv2 windows
        cap.release()
        cv2.destroyAllWindows()

    def _blink(self, pin):
        # beep for 2 seconds
        gpio.set(pin, 1)
        time.sleep(2)
        gpio.set(pin, 0)

    def _alarm(self, alarm_signal):
        if alarm_signal:
            # pin 57 controls the alarm hardware
            self._blink(57)
            print "ALARM ON"
        else:
            print "ALARM OFF"

    def _write(self, data):
        data_string = ""
        for data_entry in data:
            data_string += ",".join([str(e) for e in data_entry]) + '\n'
        # append new data to the end of the file
        with open(self.FILE_NAME, "a") as f:
            f.write(data_string)

if __name__ == "__main__":
    # TO DO: give user the option to specify height_threshold, 
    # motion_threshold, recording_length from command line
    # possible option: argparser

    # calibrate this value to match each specific configuration
    height_threshold = 50
    motion_threshold = 0.005
    # width of each column, measured in AR tag side lengths
    side_margin = 3
    # time the program will run for (sec); -1 --> run indefinitely
    recording_length = -1
    fume_hood = FumeHood(height_threshold, motion_threshold, side_margin)
    fume_hood.run(recording_length)
