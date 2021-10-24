import socket

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    conn, address = s.accept()

    with conn:
        data = conn.recv(1024)
        print(data)
        while True:
            s = input()
            conn.sendall(s.encode())
