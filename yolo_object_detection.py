# please goto below link to download yolov4.cfg and yolov4.weights,
# https://github.com/AlexeyAB/darknet/tree/master/cfg/yolov4.cfg copy this context and save to yolov4.cfg
# wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# and put those files into yolo-coco_v4
# import the necessary packages
from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import cv2
import os


class yolo_object_detection():
    # private
    __labelsPath = "./yolo-coco_v4/coco.names"
    __weightsPath = "yolo-coco_v4/yolov4.weights"
    __configPath = "yolo-coco_v4/yolov4.cfg"
    __confidence_setting = 0.7
    __threshold = 0.1
    __total_detection_img = 10
    __cutout_frame_list = []
    # image 1 -> full size
    # image 2 -> crop_y 0:1080, crop_x 0:1920
    # image 3 -> crop_y 0:1080, crop_x 960:1920
    # image 4 -> crop_y 0:1080, crop_x 1920:3840
    # image 5 -> crop_y 540:1620, crop_x 1920:3840
    # image 6 -> crop_y 540:1620 crop_x 960:2880
    # image 7 -> crop_y 540:1620 crop_x 1920:3840
    # image 8 -> crop_y 1080:2160, crop_x 1920:3840
    # image 9 -> crop_y 1080:2160 crop_x 960:2880
    # image 10 -> crop_y 1080:2160 crop_x 1920:3840
        
    
    def __judge_object_confidence_label_to_get_bbox_for_crop(self, layerOutputs, boxes, crop_width, crop_height, crop_x, crop_y, classIDs, confidences):
        for output in layerOutputs:   
            # loop over each of the detections
            for detection in output:  
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if self.__LABELS[classID] !=  self.__target_label:
                    continue

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.__confidence_setting:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height          
                    box = detection[0:4] * np.array([crop_width, crop_height, crop_width, crop_height])
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

    def __judge_object_confidence_label_to_get_bbox_for_full(self, layerOutputs, boxes, W, H, classIDs, confidences):
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if self.__LABELS[classID] !=  self.__target_label:
                    continue

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
                    if width < W / 3:
                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)


    # public
    def __init__(self, target_label):
        np.random.seed(42)
        self.__LABELS = open(self.__labelsPath).read().strip().split("\n")
        self.__COLORS = np.random.randint(0, 255, size=(len(self.__LABELS), 3), dtype="uint8")
        self.__target_label = target_label

    def cutout_frame(self, frame):
        self.__cutout_frame_list = []
        self.__start_position = []
        # full image
        self.__cutout_frame_list.append(frame)
        self.__start_position.append([0,0])

        crop_width = 1920
        crop_height = 1080
        x_interval = int(crop_width/2)
        y_interval = int(crop_height/2)

        ct = 1
        for crop_x in range(0, 2840, x_interval):
            for crop_y in range(0, 1620, y_interval):
                self.__start_position.append([])
                crop_img = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
                self.__cutout_frame_list.append(crop_img)
                self.__start_position[ct].append(crop_y)
                self.__start_position[ct].append(crop_x)
                ct = ct + 1
    
    def cutout_frame1(self, frame, crop_x, crop_y):

        crop_width = 1920
        crop_height = 1080
        x_interval = int(crop_width/2)
        y_interval = int(crop_height/2)

        crop_img = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        return crop_img

    def cal_cutout_crop_x_cropy(self):
        self.__start_position = []
        # full image
        self.__start_position.append([0,0])

        crop_width = 1920
        crop_height = 1080
        x_interval = int(crop_width/2)
        y_interval = int(crop_height/2)

        ct = 1
        for crop_x in range(0, 2840, x_interval):
            for crop_y in range(0, 1620, y_interval):
                self.__start_position.append([])
                self.__start_position[ct].append(crop_y)
                self.__start_position[ct].append(crop_x)
                ct = ct + 1

    def run_multi_core_detection_setting(self, frame):
        self.inputQueues = []
        self.outputQueues = []

        self.cal_cutout_crop_x_cropy()

        # derive the paths to the YOLO weights and model configuration
        # load our YOLO object detector trained on COCO dataset (80 classes)
        for i in range(self.__total_detection_img):
            oq = multiprocessing.Queue()
            self.outputQueues.append(oq)

            processes = multiprocessing.Process(
                            target = self.yolo_detection,
                            args = (frame, self.__configPath, self.__weightsPath, self.__target_label, \
                                    self.__LABELS, self.__COLORS, self.__confidence_setting, self.__threshold, self.__start_position[i], oq, i))
            processes.daemon = True
            processes.start() 

    def yolo_detection(self, frame, configPath, weightsPath, target_label, LABELS, COLORS , \
                        confidence_setting, threshold, start_position, outputQueue, index):
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        return_bboxes = []
        confidences = []
        classIDs = []
        crop_width = 1920
        crop_height = 1080
        crop_y = start_position[0] 
        crop_x = start_position[1]

        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # initialize the width and height of the frames in the video file
        W = None
        H = None

        if index == 0:
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]
       
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
        else:
            crop_frame = self.cutout_frame1(frame, crop_x, crop_y)
        
            blob = cv2.dnn.blobFromImage(crop_frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)

        net.setInput(blob)
        layerOutputs = net.forward(ln)
        # loop over each of the layer outputs
        if index == 0:
            self.__judge_object_confidence_label_to_get_bbox_for_full(layerOutputs, boxes, W, H, classIDs, confidences)
        else:
            self.__judge_object_confidence_label_to_get_bbox_for_crop(layerOutputs, boxes, crop_width, crop_height, crop_x, crop_y, classIDs, confidences)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_setting, threshold)
        #print(len(idxs))
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                #(x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                if LABELS[classIDs[i]] == target_label:
                    #if w < W/3 or h < H/3:
                    return_bboxes.append(boxes[i])
                    #color = [int(c) for c in COLORS[classIDs[i]]]
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    #text = "{}: {:.4f}".format(self.__LABELS[classIDs[i]], confidences[i])
                    #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        outputQueue.put(return_bboxes)

    def run_multi_core_detection(self, frame):
        get_bboxes = []
        for i,oq in enumerate(self.outputQueues):
            bbox_temp = oq.get()
            if len(bbox_temp)!=0:
                get_bboxes.append(bbox_temp)

        #print(" ======== get_bboxes =========")
        #print(get_bboxes)

        return_bboxes = []
        for i in range(len(get_bboxes)):
            for j,bbox in enumerate(get_bboxes[i]):
                return_bboxes.append(bbox)
                (x, y) = (bbox[0], bbox[1])
                (w, h) = (bbox[2], bbox[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

        cv2.imwrite("frame_multi_core.jpg", frame)
        return return_bboxes

    def run_detection(self, frame):
        self.__net = cv2.dnn.readNetFromDarknet(self.__configPath, self.__weightsPath)
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
        crop_width = 1920
        crop_height = 1080
        crop_switch = False
        # Add a switch to decide whether to cutout pictures to improve YOLO
        # Each time crop a 1920*1080 from frame to YOLO
        # Overlap 50%
        if crop_switch == False:
            for crop_x in range(0, 2840, 960):
                for crop_y in range(0, 1620, 540):
                    #print("crop_y:%d" % crop_y)
                    #print("crop_y + crop_height:%d" % (crop_y + crop_height))
                    #print("crop_x:%d" % crop_x)
                    #print("crop_x + crop_width:%d" % (crop_x + crop_width))

                    crop_img = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
                    blob = cv2.dnn.blobFromImage(crop_img, 1 / 255.0, (416, 416),
                                                 swapRB=True, crop=False)
                    self.__net.setInput(blob)
                    layerOutputs = self.__net.forward(ln)
                    # loop over each of the layer outputs
                    self.__judge_object_confidence_label_to_get_bbox_for_crop(layerOutputs, boxes, crop_width, crop_height, crop_x, crop_y, classIDs, confidences)
            
            # full image
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            self.__net.setInput(blob)
            layerOutputs = self.__net.forward(ln)
            self.__judge_object_confidence_label_to_get_bbox_for_full(layerOutputs, boxes, W, H, classIDs, confidences)
        else:
            # full image
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            self.__net.setInput(blob)
            layerOutputs = self.__net.forward(ln)
            self.__judge_object_confidence_label_to_get_bbox_for_full(layerOutputs, boxes, W, H, classIDs, confidences)


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
                return_bboxes.append(boxes[i])
                color = [int(c) for c in self.__COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.__LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite("frame_one_core.jpg", frame)
        return return_bboxes
