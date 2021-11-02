# please goto below link to download yolov4.cfg and yolov4.weights,
# wget https://github.com/AlexeyAB/darknet/tree/master/cfg/yolov4.cfg
# wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# and put those files into yolo-coco_v4
# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import os


class yolo_object_detection():
# private
    __labelsPath = "./yolo-coco_v4/coco.names"
    __weightsPath = "yolo-coco_v4/yolov4.weights"
    __configPath = "yolo-coco_v4/yolov4.cfg"
    __confidence_setting = 0.5
    __threshold = 0.1

# public
    def __init__(self):
        np.random.seed(42)
        self.__LABELS = open(self.__labelsPath).read().strip().split("\n")
        self.__COLORS = np.random.randint(0, 255, size=(len(self.__LABELS), 3), dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        # load our YOLO object detector trained on COCO dataset (80 classes)
        self.__net = cv2.dnn.readNetFromDarknet(self.__configPath, self.__weightsPath)


    def run_detection(self, frame, specify_obj):
        # determine only the *output* layer names that we need from YOLO
        ln = self.__net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]

        # initialize the width and height of the frames in the video file
        W = None
        H = None

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        boxes = []
        return_bboxes = []
        confidences = []
        classIDs = []
        crop_width = 960
        crop_high = 540
        crop_switch = True
        # Add a switch to decide whether to cutout pictures to improve YOLO
        if crop_switch==True:
            if crop_switch:
                for crop_x in range(0, 3456, 576):
                    for crop_y in range(0, 1944, 324):
                        crop_img = frame[crop_y:crop_y + crop_high, crop_x:crop_x + crop_width]
                        blob = cv2.dnn.blobFromImage(crop_img, 1 / 255.0, (416, 416),
                                                     swapRB=True, crop=False)
                        self.__net.setInput(blob)
                        layerOutputs = self.__net.forward(ln)
                        # loop over each of the layer outputs
                        for output in layerOutputs:
                            # loop over each of the detections
                            for detection in output:
                                # extract the class ID and confidence (i.e., probability)
                                # of the current object detection
                                scores = detection[5:]
                                classID = np.argmax(scores)
                                confidence = scores[classID]

                                # filter out weak predictions by ensuring the detected
                                # probability is greater than the minimum probability
                                if confidence > self.__confidence_setting:
                                    # scale the bounding box coordinates back relative to
                                    # the size of the image, keeping in mind that YOLO
                                    # actually returns the center (x, y)-coordinates of
                                    # the bounding box followed by the boxes' width and
                                    # height
                                    box = detection[0:4] * np.array([crop_width, crop_high, crop_width, crop_high])
                                    (centerX, centerY, width, height) = box.astype("int")
                                    if width < crop_width * 2 / 3:
                                        # use the center (x, y)-coordinates to derive the top
                                        # and and left corner of the bounding box
                                        x = int(centerX - (width / 2))
                                        y = int(centerY - (height / 2))

                                        # update our list of bounding box coordinates,
                                        # confidences, and class IDs
                                        boxes.append([crop_x + x, crop_y + y, int(width), int(height)])
                                        confidences.append(float(confidence))
                                        classIDs.append(classID)
        else :
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            self.__net.setInput(blob)
            layerOutputs = self.__net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > self.__confidence_setting:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.__confidence_setting, self.__threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                if self.__LABELS[classIDs[i]] == specify_obj:
                    return_bboxes.append(boxes[i])
                    color = [int(c) for c in self.__COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.__LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite("frame.jpg",frame)
        return return_bboxes

