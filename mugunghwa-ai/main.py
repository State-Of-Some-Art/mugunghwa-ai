import time
import cv2
from motion_detection import detect_motion
from mask_rcnn_torch import Segmenter
from PIL import Image
import numpy as np


LINES = ["무궁화", "꽃이", "피었", "습니", "다!"]

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    model = Segmenter()

    print("Ready to play some game?")
    time.sleep(3)
    while True:
        pts, frame = detect_motion(5, cap, show=True)
        if pts is None:
            break
        else:
            if len(pts) > 0:
                print("Somebody moved!")
                model.set_img(frame)
                model.get_mask()
                print(pts)
                for pt in pts:
                    print(pt)
                    face = model.find_containing_face(pt)
                    if face is not None:
                        face = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                        face = cv2.resize(face, (400, 400))
                        cv2.imshow("face", face)
                        cv2.waitKey(10)
            else:
                print("Nobody moved")
        

        for i in range(5):
            print(LINES[i])
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()