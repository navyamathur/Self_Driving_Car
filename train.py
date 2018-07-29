import numpy as np
import cv2
import serial
import pygame
from pygame.locals import *
import socket
import time
import os

server_socket = socket.socket()
server_socket.bind(('192.168.1.106', 12349))
server_socket.listen(0)

# accept a single connection
connection = server_socket.accept()[0].makefile('rb')
#self.ser = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1)
send_inst = True
# create labels
k = np.zeros((4, 4), 'float')
for i in range(4):
    k[i, i] = 1
temp_label = np.zeros((1, 4), 'float')
pygame.init()
saved_frame = 0
total_frame = 0
# collect images for training
print ('Start collecting images...')
e1 = cv2.getTickCount()
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 4), 'float')
stream_bytes = b' '
frame = 1
while send_inst:
    stream_bytes += connection.read(1024)
    first = stream_bytes.find(b'\xff\xd8')
    last = stream_bytes.find(b'\xff\xd9')
    if first != -1 and last != -1:
        jpg = stream_bytes[first:last + 2]
        stream_bytes = stream_bytes[last + 2:]
        
        image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        roi = image[120:240, :]
        cv2.imwrite('training_img/frame{:>05}.jpg'.format(frame), image)
        #cv2.imshow('image', image)
        #print(image.shape)
        temp_array = roi.reshape(1, 38400).astype(np.float32)
        #temp_array = roi.reshape(1, 76800).astype(np.float32)
        frame += 1
        total_frame += 1
        time.sleep(0.5)
        a=input("enter: ")
        
        if a=='w':
            print("Forward")
            saved_frame += 1
           # print(roi.shape)
           # print(temp_array.shape)
            
            image_array = np.vstack((image_array, temp_array))
            label_array = np.vstack((label_array, k[2]))
                #self.ser.write(chr(1))

        elif a=='s':
            print("Reverse")
            saved_frame += 1
            image_array = np.vstack((image_array, temp_array))
            label_array = np.vstack((label_array, k[3]))
                    #self.ser.write(chr(2))
                            
        elif a=='d':
            print("Right")
            image_array = np.vstack((image_array, temp_array))
            label_array = np.vstack((label_array, k[1]))
            saved_frame += 1
                    #self.ser.write(chr(3))

        elif a=='a':
            print("Left")
            image_array = np.vstack((image_array, temp_array))
            label_array = np.vstack((label_array, k[0]))
            saved_frame += 1
                    #self.ser.write(chr(4))

        elif a=='wd':
            print("Forward Right")
            image_array = np.vstack((image_array, temp_array))
            label_array = np.vstack((label_array, k[1]))
            saved_frame += 1
            #self.ser.write(chr(6))

        elif a=='wa':
            print("Forward Left")
            image_array = np.vstack((image_array, temp_array))
            label_array = np.vstack((label_array, k[0]))
            saved_frame += 1
            #self.ser.write(chr(7))

        elif a=='x':
            print ('exit')
            send_inst = False
                    #ser.write(chr(0))
            break
            #elif event.type == pygame.KEYUP:
                #self.ser.write(chr(0))
             #   print("extra")

        train = image_array[1:, :]
        train_labels = label_array[1:, :]
        #print(train)
        #print(train_labels)
            # save training data as a numpy file
        file_name = str(int(time.time()))
        directory = "training_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            #np.save(directory + '/' + file_name + '.jpg')
            #cv2.imwrite(directory+'/'+file_name+ '.jpg',train)
            np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
        except IOError as e:
            print(e)


        e2 = cv2.getTickCount()
        time0 = (e2 - e1) / cv2.getTickFrequency()
        print ('Streaming duration:', time0)

        #print(train.shape)
        #print(train_labels.shape)
        #print ('Total frame:', total_frame)
        print ('Saved frame:', saved_frame)
        #print ('Dropped frame', total_frame - saved_frame)
#   connection.close()
    server_socket.close()
