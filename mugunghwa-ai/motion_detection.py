import threading
import cv2
import imutils
import time
from threading import Thread


def smooth_frame(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    return blur

def calc_diff(src1, src2):
    frameDelta = cv2.absdiff(src1, src2)
    _, thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)
    return cv2.dilate(thresh, None, iterations=2)

class MotionDetector:
    def __init__(self):
        self.worker = None
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.pts = None

    def run(self, duration, img_size, threshold, show):
        start_time = time.time()
        ret, bg_frame = self.cap.read()
        bg_frame = cv2.resize(smooth_frame(bg_frame), img_size)    

        while ret:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, img_size)
            frame2 = smooth_frame(frame)
            diff = calc_diff(frame2, bg_frame)

            cnts = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            moved_points = []
            for c in cnts:
                if cv2.contourArea(c) > threshold:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    moved_points.append((int(x + w / 2), int(y + h / 2)))
            self.frame = frame.copy()
            self.pts = moved_points

            if show:
                cv2.imshow("motion", frame2)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            if time.time() - start_time > duration:
                break

        self.frame, self.pts = None, None
            
    def start(self, duration = 5, img_size = (400, 400), threshold = 8000, show = False):
        self.frame, self.pts = None, []
        self.worker = Thread(target=self.run, args=(duration, img_size, threshold, show,), daemon=True)
        self.is_running = True
        self.worker.start()

    def next(self):
        return self.frame, self.pts
    
    def release(self):
        self.cap.release()
    

if __name__ == "__main__":
    d = MotionDetector(show=True)
    d.start()
    while True:
        src, pts = d.next()
        if src is None:
            break
        print(pts)
        cv2.imshow("frame", src)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    input()


def detect_motion(duration: float, cap: cv2.VideoCapture, img_size = (400, 400), threshold: float = 8000, show: bool = False):
    start_time = time.time()
    ret, bg_frame = cap.read()
    bg_frame = cv2.resize(smooth_frame(bg_frame), img_size)    

    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, img_size)
        frame2 = smooth_frame(frame)
        diff = calc_diff(frame2, bg_frame)

        cnts = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        moved_centers = []
        for c in cnts:
            if cv2.contourArea(c) > threshold:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
                moved_centers.append((int(x + w / 2), int(y + h / 2)))
        
        if show:
            cv2.imshow("motion", frame2)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                return None

        if len(moved_centers) > 0:
            return moved_centers, frame
        
        if time.time() - start_time > duration:
            return [], None



