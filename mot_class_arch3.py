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


class mot_class_arch3():
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

    def __assign_amount_of_people_for_tracker(self, detect_people_qty, using_processor_qty):
        # it should brings (left, top, width, height) to tracker.init() function
        # parameters are left, top , right and bottom in the box 
        # so those parameters need to minus like below to get width and height 
        left_num = detect_people_qty % using_processor_qty
        process_num = int(detect_people_qty / using_processor_qty)
        processor_task_num = []
        process_num_ct = 0
        # print("bboxes:")
        # print(bboxes)
        for i in range(using_processor_qty):
            task_ct = 0
            for j in range(process_num_ct, process_num_ct + process_num):
                task_ct = task_ct + 1
                process_num_ct = process_num_ct + 1
            processor_task_num.append(task_ct)
        if left_num != 0:
            counter = 0
            k = detect_people_qty - using_processor_qty * process_num
            for k in range(k, k + left_num):
                # print("k:%d" % k)
                processor_task_num[counter] = processor_task_num[counter] + 1
                counter = counter + 1
                # print("processor_task_number:")
        # print(processor_task_num)
        return processor_task_num

    # public
    def __init__(self, bboxes, frame, resize_width):
        # start the frames per second throughput estimator
        self.__fps = FPS().start()

        self.inputQueues = []
        self.outputQueues = []

        self.__frame_size_width = resize_width
        self.__detect_people_qty = len(bboxes)
        if self.__detect_people_qty >= (os.cpu_count() - 2):
            using_processor_qty = os.cpu_count() - 2
        else:
            using_processor_qty = self.__detect_people_qty

        self.__processor_task_num = self.__assign_amount_of_people_for_tracker(self.__detect_people_qty,
                                                                               using_processor_qty)

        ct = 0
        for i in range(using_processor_qty):
            bboxes_for_trackers = []
            for j in range(int(self.__processor_task_num[i])):
                bboxes_for_trackers.append(bboxes[ct])
                ct += 1

            iq = multiprocessing.Queue()
            oq = multiprocessing.Queue()
            self.inputQueues.append(iq)
            self.outputQueues.append(oq)

            processes = multiprocessing.Process(
                target=self.start_tracker,
                args=(frame, bboxes_for_trackers, iq, oq))
            processes.daemon = True
            processes.start()

        print("detect_people_qty: %d" % self.__detect_people_qty)
        print("processor_task_num")
        print(self.__processor_task_num)

        self.__now_frame = frame

    def start_tracker(self, frame, bboxes, inputQueue, outputQueue):
        # print("start_tracker")
        tracker = cv2.MultiTracker_create()
        for i, bbox in enumerate(bboxes):
            mbbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            tracker.add(self.__get_algorithm_tracker("CSRT"), frame, mbbox)

        while True:
            bboxes_org = []
            bboxes_transfer = []
            frame = inputQueue.get()
            # print("receive frame")
            ok, bboxes_org = tracker.update(frame)

            for box in bboxes_org:
                startX = int(box[0])
                startY = int(box[1])
                endX = int(box[0] + box[2])
                endY = int(box[1] + box[3])
                bbox = (startX, startY, endX, endY)
                bboxes_transfer.append(bbox)

            outputQueue.put(bboxes_transfer)

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

            for i, iq in enumerate(self.inputQueues):
                iq.put(frame)

            bboxes = []
            for i, oq in enumerate(self.outputQueues):
                bboxes.append(oq.get())

            # print("=====================================")
            # print(bboxes)
            for i, bbox in enumerate(bboxes):
                for j in range(self.__processor_task_num[i]):
                    # print(bbox[j])
                    (startX, startY, endX, endY) = bbox[j]

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
