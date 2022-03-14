# watch CPU loading on the ubuntu
# ps axu | grep [M]OTF_process_pool.py | awk '{print $2}' | xargs -n1 -I{} ps -o sid= -p {} | xargs -n1 -I{} ps --forest -o user,pid,ppid,cpuid,%cpu,%mem,stat,start,time,command -g {}

# import the necessary packages
from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import imutils
import cv2
import os
import datetime

class mot_class_arch1():
    # private

    # for saving tracker objects
    # detected flag
    __detection_ok = False
    # if below variable set to True, this result will not show tracking bbox on the video
    # ,it will show number on the terminal
    __frame_size_width = 3840
    __detect_people_qty = 0

    # initialize the list of class labels MobileNet SSD was trained to detect
    __CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                 "sofa", "train", "tvmonitor"]
    # can not use class video_capture variable, otherwise this process will crash
    # __vs = 0
    __processor_task_num = []

    def __get_algorithm_tracker(self, algorithm):
        if algorithm == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif algorithm == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif algorithm == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif algorithm == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif algorithm == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif algorithm == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif algorithm == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif algorithm == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        return tracker

    # public
    def __init__(self, bboxes, frame, resize_width):
        # start the frames per second throughput estimator
        self.__fps = FPS().start()

        self.__frame_size_width = resize_width
        self.__detect_people_qty = len(bboxes)
        print("detect_people_qty: %d" % self.__detect_people_qty)

        #fps = FPS().start()
        print("--------------------------------")
        print(datetime.datetime.now());
        self.__tracker = cv2.MultiTracker_create()
        #fps.stop()
        print(datetime.datetime.now());
        print("--------------------------------")
        #print("==========[INFO] create elapsed time: {:.2f}=========".format(fps.elapsed()))

        for i, bbox in enumerate(bboxes):
            mbbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            self.__tracker.add(self.__get_algorithm_tracker("CSRT"), frame, mbbox)

        
        self.__now_frame = frame
        # tracking person on the video

    def tracking(self, args):
        vs = cv2.VideoCapture(args["video"])
        # print("tracking")
        # loop over frames from the video file stream
        while True:

            # grab the next frame from the video file
            if self.__detection_ok == True:
                (grabbed, frame) = vs.read()
                # print("vs read ok")
                # check to see if we have reached the end of the video file
                if frame is None:
                    break
            else:
                frame = self.__now_frame
                self.__detection_ok = True

            bboxes = []
            ok, bboxes = self.__tracker.update(frame)
            # print(bboxes)

            for i, bbox in enumerate(bboxes):
                # print(bbox)
                startX = int(bbox[0])
                startY = int(bbox[1])
                endX = int(bbox[0] + bbox[2])
                endY = int(bbox[1] + bbox[3])

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, "person", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            # print("before imshow")
            frame = imutils.resize(frame, 1280)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            self.__fps.update()

        # stop the timer and display FPS information
        self.__fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.__fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.__fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.release()
