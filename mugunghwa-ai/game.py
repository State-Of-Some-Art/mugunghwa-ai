import time
import cv2
from motion_detection import detect_motion, MotionDetector
from mask_rcnn_torch import Segmenter
from PIL import Image
import numpy as np


LINES = ["MugungHwa", "GGochi", "PeeUt", "SeupNi", "Da!"]

if __name__ == "__main__":
    FRONT = cv2.imread("src/FRONT.png")
    BACK = cv2.imread("src/BACK.png")
    CATCH = cv2.imread("src/CATCH.png")
    LOGO = cv2.imread("src/logo.png")

    cv2.imshow("main", LOGO)
    cv2.imshow("motion", LOGO)
    cv2.imshow("face", LOGO)
    cv2.waitKey(10)
    model = Segmenter()
    detector = MotionDetector()

    print("Ready to play some game?")

    while True:
        for i in range(5):
            BACK_TEXT = BACK.copy()
            cv2.putText(BACK_TEXT, LINES[i], (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("main", BACK_TEXT)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        
        cv2.imshow("main", FRONT)
        cv2.waitKey(10)
        detector.start(show=True)
        
        while True:
            frame, pts = detector.next()
            if pts is None:
                break
            if len(pts) > 0:
                print("Somebody moved!")
                cv2.imshow("main", CATCH)
                cv2.waitKey(10)
                model.set_img(frame)
                model.get_mask()
                for i, pt in enumerate(pts):
                    face = model.find_containing_face(pt)
                    if face is not None:
                        face = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                        face = cv2.resize(face, (400, 400))
                        cv2.imshow(f"face {i}", face)
                        cv2.waitKey(10)
        print("Nobody moved")   

    cap.release()
    cv2.destroyAllWindows()