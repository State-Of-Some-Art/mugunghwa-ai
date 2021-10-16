import cv2
import numpy as np
import imutils
import time


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
    time.sleep(3)
    ret, bg_frame = cap.read()
    bg_frame = smooth_frame(bg_frame)

    while ret:
        ret, frame = cap.read()
        frame2 = smooth_frame(frame)
        diff = calc_diff(frame2, bg_frame)

        cnts = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < 8000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("asd", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
