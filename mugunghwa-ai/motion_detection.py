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

def detect_motion(duration: float, cap: cv2.VideoCapture, threshold: float = 8000, show: bool = False):
    start_time = time.time()
    ret, bg_frame = cap.read()
    bg_frame = smooth_frame(bg_frame)

    while ret:
        ret, frame = cap.read()
        frame2 = smooth_frame(frame)
        diff = calc_diff(frame2, bg_frame)

        cnts = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        moved_centers = []
        for c in cnts:
            if cv2.contourArea(c) > threshold:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                moved_centers.append((x + w / 2, y + h / 2))
        if len(moved_centers) > 0:
            return moved_centers, frame
        if show:
            cv2.imshow("motion", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                return None
        
        if time.time() - start_time > duration:
            return [], None



