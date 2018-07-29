import numpy as np
import cv2
import socket

server_socket = socket.socket()
server_socket.bind(('192.168.100.9', 12473))
server_socket.listen(0)
connection, client_address = server_socket.accept()
connection = connection.makefile('rb')


print ("Connection from: ", client_address)
print ("Streaming...")
print ("Press 'q' to exit")
#stream_bytes = ' '
stream_bytes = b' '
while True:
    stream_bytes += connection.read(1024)
    #first = stream_bytes.find('\xff\xd8')
    #last = stream_bytes.find('\xff\xd9')
    first = stream_bytes.find(b'\xff\xd8')
    last = stream_bytes.find(b'\xff\xd9')
    if first != -1 and last != -1:
        jpg = stream_bytes[first:last + 2]
        stream_bytes = stream_bytes[last + 2:]
        #image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
        #image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.LOAD_IMAGE_UNCHANGED)
		#image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

connection.close()
server_socket.close()
