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

            self.detector.start(show=True)
            
            while True:
                frame, pts = self.detector.next()
                if pts is None:
                    break

                if len(pts) > 0:
                    print("Movement detected")
                    self.model.set_img(frame)
                    self.model.get_mask()
                    for pt in pts:
                        face = self.model.find_containing_face(pt)
                        if face is not None:
                            face = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                            face = cv2.resize(face, (400, 400))
