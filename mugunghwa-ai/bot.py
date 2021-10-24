import cv2
from motion_detection import MotionDetector
from mask_rcnn_torch import Segmenter
import numpy as np
import time


class MugungHwaBot:
    def __init__(self):
        self.model = Segmenter()
        self.detector = MotionDetector()
        print("Mugungwha bot is initialized")

    def start(self):
        while True:
            for i in range(5, 0, -1):
                print(f"Detecting in {i}")
                time.sleep(1)

            self.detector.start()
            self.model.reset()
            cv2.imshow('faces', np.zeros((300, 300, 1)))
            cv2.waitKey(10)
            
            while True:
                motion_mask, src = self.detector.next()
                if motion_mask is None:
                    break
                cv2.imshow("mask", motion_mask * 255)
                cv2.imshow("src", src)
                cv2.waitKey(10)
                if np.sum(motion_mask) > 10:
                    self.model.set_img(src)
                    instance_mask = self.model.get_mask()
                    cv2.imshow("inst", (instance_mask.transpose(1, 2, 0) / 3 * 255).astype(np.uint8))
                    cv2.waitKey(10)
                    instance_moved = np.unique(motion_mask * instance_mask)

                    faces = self.model.find_instance_faces(instance_moved)

                    if len(faces) > 0:
                        face_img = np.hstack(faces)
                        cv2.imshow('faces', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(10)
                    else:
                        cv2.imshow('faces', np.zeros((300, 300, 1)))
                        cv2.waitKey(10)


