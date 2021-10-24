import socket
from threading import Thread
import time

class SocketComm:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.conn = None
        self.worker = None
        self.is_running = False

    def start(self):
        self.is_running = True
        self.worker = Thread(target=self.connect, daemon=True)
        self.worker.start()

    def send(self, data):
        self.conn.sendall(data)

    def recv(self, size):
        return self.conn.recv(size)

    def stop(self):
        self.is_running = False

    def connect(self):
        print("Waiting for unity connection...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            self.conn, _ = s.accept()
            print("Connected to Unity")
            while self.is_running:
                time.sleep(0.1)
