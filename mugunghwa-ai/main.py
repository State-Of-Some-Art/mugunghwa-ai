import time
import cv2
from motion_detection import detect_motion


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    while True:
        motion = detect_motion(5, cap, show=True)
        if motion is None:
            break
        else:
            if len(motion) > 0:
                print("Somebody moved!")
                print(motion)
            else:
                print("Nobody moved")
        
        for i in range(5, -1, -1):
            print(i)
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()