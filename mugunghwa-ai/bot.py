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
    def __init__(self, verbose=False,
                motion_threshold=1000, pixel_count_threshold=300, new_face_threshold=1.0,
                host='127.0.0.1', port=65432):
        self.segmenter = Segmenter()
        self.facenet = FaceNet()
        self.detector = MotionDetector()
        self.conn = SocketComm(host=host, port=port)
        self.detector.start()
        self.verbose = verbose
        self.motion_threshold = motion_threshold
        self.pixel_count_threshold = pixel_count_threshold
        self.new_face_threshold = new_face_threshold
        self.conn.start()
        print("Mugungwha bot is initialized")

    def start(self):
        while True:
            # Create windows and initialize
            if self.verbose:
                cv2.imshow("Source image", np.zeros((300, 300, 1)))
                cv2.imshow("Motion mask", np.zeros((300, 300, 1)))
                cv2.imshow("Instance mask", np.zeros((300, 300, 1)))
                cv2.imshow('Face log', np.zeros((300, 300, 1)))
                cv2.waitKey(10)
            
            # Handshake
            self.conn.send(b'{"c":1}')
            print("Waiting...")
            print(self.conn.recv(1024).decode())
            self.conn.send(b'{"c":0}')

            # Reset
            self.detector.reset()
            self.facenet.reset()
            
            while True:
                motion_mask, src = self.detector.next()
                if motion_mask is None:
                    break

                if self.verbose:
                    cv2.imshow("Source image", src)
                    cv2.imshow("Motion mask", motion_mask * 255)
                    cv2.waitKey(10)

                # If motion detected
                if np.sum(motion_mask) > self.motion_threshold:
                    print("Somebody moved!")
                    self.conn.send(b'{"c":2}')

                    # Get instance segmentation mask
                    self.segmenter.set_img(src)
                    instance_mask_combined, instance_mask_list = self.segmenter.get_instance_mask_combined()

                    if self.verbose:
                        cv2.imshow("Instance mask", (instance_mask_combined * 255 / len(instance_mask_list)).astype(np.uint8))
                        cv2.waitKey(10)

                    # Get instances that moved
                    instance_list, count_list = np.unique(motion_mask * instance_mask_combined.squeeze(2), return_counts=True)

                    count_list = count_list[instance_list != 0]
                    instance_list = instance_list[instance_list != 0]
                    instance_moved_list = instance_list[count_list > self.pixel_count_threshold]

                    # Get corresponding face
                    for idx, instance_mask in enumerate(instance_mask_list):
                        instance = idx + 1
                        if instance in instance_moved_list:
                            masked_img = (src * instance_mask).astype(np.uint8)
                            self.facenet.set_img(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
                            face_log = self.facenet.update_face_log(new_face_threshold=self.new_face_threshold)
                    
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


