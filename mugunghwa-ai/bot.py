import cv2
from motion_detection import MotionDetector
from mask_rcnn_torch import Segmenter
from face_detection import FaceNet
from comm import SocketComm
import numpy as np
import time
import json
from base64 import b64encode


class MugungHwaBot:
    def __init__(self, verbose=False, motion_threshold=1000, host='127.0.0.1', port=65432):
        self.segmenter = Segmenter()
        self.facenet = FaceNet()
        self.detector = MotionDetector()
        self.conn = SocketComm(host=host, port=port)
        self.verbose = verbose
        self.motion_threshold = motion_threshold
        self.conn.start()
        print("Mugungwha bot is initialized")

    def start(self):
        while True:
            self.conn.send(b'{"c":1}')
            print("Waiting...")
            print(self.conn.recv(1024).decode())
            self.conn.send(b'{"c":0}')
            self.detector.start()
            self.facenet.reset_log()
            
            if self.verbose:
                cv2.imshow('Face log', np.zeros((300, 300, 1)))
                cv2.waitKey(10)
            
            while True:
                motion_mask, src = self.detector.next()
                if motion_mask is None:
                    break
                if self.verbose:
                    cv2.imshow("Motion mask", motion_mask * 255)
                    cv2.imshow("src", src)
                    cv2.waitKey(10)
                if np.sum(motion_mask) > self.motion_threshold:
                    print("Somebody moved!")
                    self.conn.send(b'{"c":2}')
                    self.segmenter.set_img(src)
                    instance_mask_combined, instance_mask_list = self.segmenter.get_instance_mask_combined()
                    if self.verbose:
                        cv2.imshow("Instance mask", (instance_mask_combined * 255 / len(instance_mask_list)).astype(np.uint8))
                        cv2.waitKey(10)

                    for instance_mask in instance_mask_list:
                        masked_img = (src * instance_mask).astype(np.uint8)
                        self.facenet.set_img(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
                        face_log = self.facenet.update_face_log(thres=1.5)
                    
                    if len(face_log) > 0:
                        face_strings = []
                        for face in face_log:
                            _, b = cv2.imencode('.png', cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))
                            face_strings.append(b64encode(b).decode())
                        packet = {"c": 3, "imgs": face_strings}
                        self.conn.send(json.dumps(packet).encode())

                        if self.verbose:
                            face_log = np.hstack(face_log)
                            cv2.imshow('Face log', cv2.cvtColor(face_log, cv2.COLOR_RGB2BGR))
                            cv2.waitKey(10)


