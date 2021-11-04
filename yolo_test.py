import yolo_object_detection as yolo_obj
from imutils.video import FPS
import cv2

if __name__ == '__main__':
    img = cv2.imread('./image/0_1.jpg')
    fps = FPS().start()
    yolo = yolo_obj.yolo_object_detection('person')
    one_core_to_run = True
    fps = FPS().start()
    if one_core_to_run == True:
        yolo.run_detection(img)
    else:
        yolo.run_multi_core_detection_setting(img)
        yolo.run_multi_core_detection(img)

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    
