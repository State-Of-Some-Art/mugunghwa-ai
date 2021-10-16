import cv2
import numpy as np


def smooth_frame(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    return blur

def calc_diff(src1, src2):
    frameDelta = cv2.absdiff(src1, src2)
    _, thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)
    return cv2.dilate(thresh, None, iterations=2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    ret, bg_frame = cap.read()
    bg_frame = smooth_frame(bg_frame)

    while ret:
        ret, frame = cap.read()
        frame = smooth_frame(frame)
        diff = calc_diff(frame, bg_frame)
        cv2.imshow("asd", diff)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
