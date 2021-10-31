import mot_class_arch1 as mtc
import os              
import argparse
import yolo_object_detection as yolo_obj
import cv2
import imutils


def read_user_input_info():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    args = read_user_input_info()
    vs = cv2.VideoCapture(args["video"])
    (grabbed, frame) = vs.read()
    resize_width = 1280
    frame = imutils.resize(frame, width=resize_width)
    (h, w) = frame.shape[:2]

    yolo = yolo_obj.yolo_object_detection()
    bboxes = []
    bboxes = yolo.run_detection(frame, 'person')

    #print(bboxes)
    # load our serialized model from disk                                   
    mc = mtc.mot_class_arch1(bboxes, frame, resize_width)
    mc.tracking(args)
