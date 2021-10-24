import cv2
from motion_detection import MotionDetector
from mask_rcnn_torch import Segmenter
from face_detection import FaceNet
from comm import SocketComm
import numpy as np
import time
import json
from base64 import b64encode
from pdb import set_trace as bp

class MugungHwaBot:
    def __init__(self):
        self.segmenter = Segmenter()
        self.facenet = FaceNet()
        self.detector = MotionDetector()
        # self.conn = SocketComm()
        # self.conn.start()
        # while self.conn.conn is None:
        #     time.sleep(0.1)
        print("Mugungwha bot is initialized")

    def start(self):
        while True:
            # self.conn.send(b'{"c":1}')
            # for i in range(6, 0, -1):
            #     print(f"Detecting in {i}")
            #     time.sleep(1)
            # self.conn.send(b'{"c":0}')
            self.detector.start()
            self.facenet.reset_log()

            cv2.imshow('Face log', np.zeros((300, 300, 1)))
            cv2.waitKey(10)
            
            while True:
                motion_mask, src = self.detector.next()
                if motion_mask is None:
                    break
                cv2.imshow("Motion mask", motion_mask * 255)
                cv2.imshow("src", src)
                cv2.waitKey(10)
                if np.sum(motion_mask) > 1000:
                    # self.conn.send(b'{"c":2}')
                    self.segmenter.set_img(src)
                    instance_mask_combined, instance_mask_list = self.segmenter.get_instance_mask_combined()

                    cv2.imshow("Instance mask", (instance_mask_combined * 255 / len(instance_mask_list)).astype(np.uint8))
                    cv2.waitKey(10)

                    for instance_mask in instance_mask_list:
                        masked_img = (src * instance_mask).astype(np.uint8)
                        self.facenet.set_img(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
                        face_log = self.facenet.update_face_log(thres=1.5)
                    
                    if len(face_log) > 0:
                        face_log = np.hstack(face_log)
                        cv2.imshow('Face log', cv2.cvtColor(face_log, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(10)


