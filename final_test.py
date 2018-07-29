import threading
import socketserver
import numpy as np
import math
import cv2

global com

class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d

class NeuralNetwork(object):

    def __init__(self):
        #self.model = cv2.ANN_MLP()
        print("INSIDE Neural n/w")

    def create(self):
        layer_size = np.int32([38400, 32, 4])
        self.model = cv2.ml.ANN_MLP_create()
        #self.model.create(layer_size)
        self.model.setLayerSizes(layer_size)

        
        self.model.load('F:\\Car\\AutoRCCar-master (1)\\AutoRCCar-master\\computer\\mlp_xml\\mlp.xml')
        #self.model.load_weights('C:\\Users\\MAYANK\\Desktop\\mlp.xml')

    def predict_classes(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

class DriveHandler(socketserver.BaseRequestHandler):

    def handle(self):
        try:
            while True:
                self.request.send(com.encode('utf-8'))
                print(com)

        finally:
            print("Connection closed on thread 3")

    
class SensorDataHandler(socketserver.BaseRequestHandler):

    data = " "

    def handle(self):
        global sensor_data
        try:
            while self.data:
                self.data = self.request.recv(1024)
                print(type(self.data))
                print(self.data)
                sensor_data = round(float(self.data), 1)
                #print(type(sensor_data))
                #print "{} sent:".format(self.client_address[0])
                print (sensor_data)
        finally:
            print ("Connection closed on thread 2")

class RCControl(object):
    
    def __init__(self):
        print("Inside RC Control")

    
    def steer(self,prediction):

        def server_thread3(host, port):
            server = socketserver.TCPServer((host, port), DriveHandler)
            server.serve_forever()


        drive_thread = threading.Thread(target=server_thread3, args = ('192.168.100.9',12472))
        
        if prediction == 2:
            com='w'
            print("Forward")
            #drive_thread.start()            
            
        elif prediction == 0:
            com='a'
            print("Left")
            #drive_thread.start()
            
        elif prediction == 1:
            com='d'
            print("Right")
            #drive_thread.start()
            
        else:
            com='q'
            self.stop()

        drive_thread.start()
    
            
    def stop(self):
        com='q'
        print("Stop")
        #drive_thread.start()
        

class VideoStreamHandler(socketserver.StreamRequestHandler):

    # h1: stop sign
    h1 = 15.5 - 10  # cm
    # h2: traffic light
    h2 = 15.5 - 10

    model = NeuralNetwork()
    model.create()

    #obj_detection = ObjectDetection()

    rc_car = RCControl()

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('F:\\Car\\AutoRCCar-master (1)\\AutoRCCar-master\\computer\\cascade_xml\\stop_sign.xml')
    light_cascade = cv2.CascadeClassifier('F:\\Car\\AutoRCCar-master (1)\\AutoRCCar-master\\computer\\cascade_xml\\traffic_light.xml')

    #he=int(input("Enter prediction: "))
    #rc_car.steer(he)

    d_to_camera = DistanceToCamera()
    d_stop_sign = 25
    d_light = 25

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0

    def handle(self):

        global sensor_data
        stream_bytes = b' '
        stop_flag = False
        stop_sign_active = True

        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    # lower half of the image
                    half_gray = gray[120:240, :]

                    # object detection
                    v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)
                    v_param2 = self.obj_detection.detect(self.light_cascade, gray, image)

                    # distance measurement
                    if v_param1 > 0 or v_param2 > 0:
                        d1 = self.d_to_camera.calculate(v_param1, self.h1, 300, image)
                        d2 = self.d_to_camera.calculate(v_param2, self.h2, 100, image)
                        self.d_stop_sign = d1
                        self.d_light = d2

                    cv2.imshow('image', image)
                    #cv2.imshow('mlp_image', half_gray)

                    # reshape image
                    image_array = half_gray.reshape(1, 38400).astype(np.float32)
                    
                    # neural network makes prediction
                    #print(image_array)
                    prediction = self.model.predict_classes(image_array)
                    print(prediction)
                    

                    # stop conditions
                    #sensor_data = float(sensor_data)
                    if sensor_data is not None and sensor_data < 30.0:
                    #if sensor_data is None:                    
                        print("Stop, obstacle in front")
                        self.rc_car.stop()
                    
                    elif 0.0 < self.d_stop_sign < 25.0 and stop_sign_active:
                        print("Stop sign ahead")
                        self.rc_car.stop()

                        # stop for 5 seconds
                        if stop_flag is False:
                            self.stop_start = cv2.getTickCount()
                            stop_flag = True
                        self.stop_finish = cv2.getTickCount()

                        self.stop_time = (self.stop_finish - self.stop_start)/cv2.getTickFrequency()
                        print ("Stop time: %.2fs" % self.stop_time)

                        # 5 seconds later, continue driving
                        if self.stop_time > 5:
                            print("Waited for 5 seconds")
                            stop_flag = False
                            stop_sign_active = False

                    elif 0 < self.d_light < 30:
                        #print("Traffic light ahead")
                        if self.obj_detection.red_light:
                            print("Red light")
                            self.rc_car.stop()
                        elif self.obj_detection.green_light:
                            print("Green light")
                            pass
                        elif self.obj_detection.yellow_light:
                            print("Yellow light flashing")
                            pass
                        
                        self.d_light = 30
                        self.obj_detection.red_light = False
                        self.obj_detection.green_light = False
                        self.obj_detection.yellow_light = False

                    else:
                        self.rc_car.steer(prediction)
                        self.stop_start = cv2.getTickCount()
                        self.d_stop_sign = 25

                        if stop_sign_active is False:
                            self.drive_time_after_stop = (self.stop_start - self.stop_finish)/cv2.getTickFrequency()
                            if self.drive_time_after_stop > 5:
                                stop_sign_active = True

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.rc_car.stop()
                        break

            cv2.destroyAllWindows()

        finally:
            print ("Connection closed on thread 1")

class ThreadServer(object):

    def server_thread2(host, port):
        server = socketserver.TCPServer((host, port), SensorDataHandler)
        server.serve_forever()

    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

        
    distance_thread = threading.Thread(target=server_thread2, args=('192.168.100.9', 12473))
    distance_thread.start()
    video_thread = threading.Thread(target=server_thread('192.168.100.9', 12474))
    video_thread.start()
    
if __name__ == '__main__':
    ThreadServer()
