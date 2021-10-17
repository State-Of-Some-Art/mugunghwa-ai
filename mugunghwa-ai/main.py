import time
import cv2
from motion_detection import detect_motion
from mask_rcnn import MASK_RCNN
from PIL import Image


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    m = MASK_RCNN()
    
    while True:
        pts, frame = detect_motion(5, cap, show=True)
        if pts is None:
            break
        else:
            if len(pts) > 0:
                print("Somebody moved!")
                m.img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                m.set_image_size(frame.shape[:2])
                m.get_mask()
                for pt in pts:
                    print(pt)
                    face = m.find_containing_mask(pt)
                    face.show()

            else:
                print("Nobody moved")
        
        for i in range(5, -1, -1):
            print(i)
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()