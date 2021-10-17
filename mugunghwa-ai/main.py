import time
import cv2
from motion_detection import detect_motion
from mask_rcnn_torch import Segmenter
from PIL import Image


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    model = Segmenter()

    print("Ready to play some game?")
    
    while True:
        pts, frame = detect_motion(5, cap, show=True)
        if pts is None:
            break
        else:
            if len(pts) > 0:
                print("Somebody moved!")
                model.set_img(frame)
                model.get_mask()
                for pt in pts:
                    print(pt)
                    face = model.find_containing_face(pt)
                    face.show()
            else:
                print("Nobody moved")
        
        for i in range(5, -1, -1):
            print(i)
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()