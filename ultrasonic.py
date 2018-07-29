import time
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
def measure():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    start = time.time()
    while GPIO.input(GPIO_ECHO)==0:
        start = time.time()
    while GPIO.input(GPIO_ECHO)==1:
        stop = time.time()
    elapsed = stop-start
    distance = (elapsed * 34300)/2
    return distance
# referring to the pins by GPIO numbers
GPIO.setmode(GPIO.BCM)
GPIO_TRIGGER = 21
GPIO_ECHO    = 20
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
GPIO.setup(GPIO_ECHO,GPIO.IN)
GPIO.output(GPIO_TRIGGER, False)
while True:
	distance = measure()
	print ("Distance : %.1f cm" % distance)
    time.sleep(0.5)

GPIO.cleanup()
