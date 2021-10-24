import cv2
from motion_detection import MotionDetector
from mask_rcnn_torch import Segmenter
from comm import SocketComm
import numpy as np
import time
import json
from base64 import b64encode

class MugungHwaBot:
    def __init__(self):
        self.model = Segmenter()
        self.detector = MotionDetector()
        self.conn = SocketComm()
        self.conn.start()
        while self.conn.conn is None:
            time.sleep(0.1)
        print("Mugungwha bot is initialized")

    def start(self):
        while True:
            self.conn.send(b'{"c":1}')
            for i in range(6, 0, -1):
                print(f"Detecting in {i}")
                time.sleep(1)
            self.conn.send(b'{"c":0}')
            self.detector.start()
            
            while True:
                motion_mask, src = self.detector.next()
                if motion_mask is None:
                    break
                # cv2.imshow("mask", motion_mask * 255)
                # cv2.imshow("src", src)
                # cv2.waitKey(10)
                if np.sum(motion_mask) > 1000:
                    self.conn.send(b'{"c":2}')
                    self.model.set_img(src)
                    instance_mask = self.model.get_mask()
                    # cv2.imshow("inst", (instance_mask.transpose(1, 2, 0) / 3 * 255).astype(np.uint8))
                    # cv2.waitKey(10)
                    instance_moved, counts = np.unique(motion_mask * instance_mask, return_counts=True)
                    faces = self.model.find_instance_faces(instance_moved[counts > 1000])
                    if len(faces) > 0:
                        face_strings = []
                        for face in faces:
                            _, b = cv2.imencode('.png', cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))
                            face_strings.append(b64encode(b).decode())
                        packet = {"c": 3, "imgs": face_strings}
                        self.conn.send(json.dumps(packet).encode())
                    #     face_img = np.hstack(faces)
                    #     cv2.imshow('faces', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    #     cv2.waitKey(10)
                    # else:
                    #     cv2.imshow('faces', np.zeros((300, 300, 1)))
                    #     cv2.waitKey(10)


