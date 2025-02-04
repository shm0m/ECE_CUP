 #!/usr/bin/python3
# File name   : findline.py
# Description : line tracking 
# Website     : www.adeept.com
# Author      : William
# Date        : 2019/11/21
import RPi.GPIO as GPIO
import time
import GUImove as move
import servo
import LED
import cv2
import numpy as np

line_pin_right = 19
line_pin_middle = 16
line_pin_left = 20

def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(line_pin_right,GPIO.IN)
    GPIO.setup(line_pin_middle,GPIO.IN)
    GPIO.setup(line_pin_left,GPIO.IN)
    #motor.setup()
    
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img):
    _,contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)  

        if area > 500:
            perimeter = cv2.arcLength(cnt, True) 
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            objCorner = len(approx)
            x, y, w, h = cv2.boundingRect(approx) 

            if objCorner == 3:
                print("triangle")
                objectType = 'Triangle'
                print(objectType)
            elif objCorner == 4:
                aspectRatio = float(w) / float(h)
                if 0.95 < aspectRatio < 1.05:
                    objectType = 'Square'
                else:
                    objectType = "Rectangle"
            elif objCorner == 10:
                objectType = 'Etoile'
                print(objectType)
            elif objCorner > 4:
                objectType = 'Circle'
            else:
                objectType = "None"

def checkCam():
    while 1:
        cap = cv2.VideoCapture(0)
    
        ret, img = cap.read()
        if not ret:
            print("Lecture echoue")
            break
        
        imgContour = img.copy()
        
        imgGray = cv2.cvtColor(img,
                               cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        imgCanny = cv2.Canny(imgBlur, 50,
                             50)
        getContours(imgCanny)
        
        imgBlank = np.zeros_like(img)
        
        imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
    
        lower_red = np.array([0, 40, 40]) 
        upper_red = np.array([5, 255, 255])  
        
       
        lower_blue = np.array([100, 50, 50])  
        upper_blue = np.array([130, 255, 255])  
        
       
        lower_green = np.array([40, 40, 40])  
        upper_green = np.array([80, 255, 255])  
        
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        
        maskRouge = cv2.inRange(imageHSV, lower_red, upper_red)
        maskBlue = cv2.inRange(imageHSV, lower_blue, upper_blue)
        maskGreen = cv2.inRange(imageHSV, lower_green, upper_green)
        mask_yellow = cv2.inRange(imageHSV, lower_yellow, upper_yellow)
        
        combined_mask = maskBlue + maskGreen + mask_yellow + maskRouge
        
        _,contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)  
            if area > 5000:  
                cv2.drawContours(imgContour, [cnt], -1, (255, 0, 0),
                                 3)  
                perimeter = cv2.arcLength(cnt, True) 
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) 
                objCorner = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                count_red = cv2.countNonZero(maskRouge)
                count_blue = cv2.countNonZero(maskBlue)
                count_yellow = cv2.countNonZero(mask_yellow)
                count_green = cv2.countNonZero(maskGreen)
        
                is_red_predominant = count_red > count_blue and count_red > count_yellow and count_red > count_green
                is_blue_predominant = count_blue > count_red and count_blue > count_yellow and count_blue > count_green
                is_yellow_predominant = count_yellow > count_red and count_yellow > count_blue and count_yellow > count_green
                is_Green_predominant = count_green > count_red and count_green > count_blue and count_green > count_yellow
        
                if is_Green_predominant:
                    print("vert")
                elif is_yellow_predominant:
                    print("yellow")
                elif is_blue_predominant:
                    print("blue")
                elif is_red_predominant:
                    print("red")
        cap.release()
        break

def TournerTeteGauche():
    for i in range(0,20):
        servo.lookleft(5)
        checkCam()
        
def TournerTeteDroite():
    for i in range(0,20):
        servo.lookright(5)
        checkCam()

led = LED.LED()
turn_status = 0
speed = 70
angle_rate = 0.3
color_select = 1 # 0 --> white line / 1 --> black line
check_true_out = 0
backing = 0
last_turn = 0
def run():
    global turn_status, speed, angle_rate, color_select, led, check_true_out, backing, last_turn
    status_right = GPIO.input(line_pin_right)
    status_middle = GPIO.input(line_pin_middle)
    status_left = GPIO.input(line_pin_left)
    #print('R%d   M%d   L%d'%(status_right,status_middle,status_left))
    

    if status_middle == color_select and status_right == color_select and status_left == color_select:
        check_true_out = 0
        backing = 0
        print('stop')
        turn_status = 0
        servo.turnMiddle()
        move.move(0, 'no')
        time.sleep(0.2)
        TournerTeteGauche()
        time.sleep(2)
        TournerTeteDroite()
        TournerTeteDroite()
        time.sleep(2)
        TournerTeteGauche()
        move.move(speed, 'forward')
        
        
		
    elif status_right == color_select:
        check_true_out = 0
        backing = 0
        print('left')
        led.colorWipe(0,255,0)
        turn_status = -1
        last_turn = -1
        servo.turnLeft(angle_rate)
        move.move(speed, 'forward')
    elif status_left == color_select:
        check_true_out = 0
        backing = 0
        print('right')
        turn_status = 1
        last_turn = 1
        led.colorWipe(0,0,255)
        servo.turnRight(angle_rate)
        move.move(speed, 'forward')

    elif status_middle == color_select:
        check_true_out = 0
        backing = 0
        print('middle')
        led.colorWipe(255,255,255)
        turn_status = 0
        servo.turnMiddle()
        move.move(speed, 'forward')
        time.sleep(0.2)
    
    else:
        print('none')
        led.colorWipe(255,0,0)
        if not backing == 1:
            if check_true_out == 1:
                check_true_out = 0
                if turn_status == 0:
                    if last_turn == 1:
                        servo.turnRight(angle_rate)
                    else:
                        servo.turnLeft(angle_rate)
                    move.move(speed, 'backward')
                    time.sleep(0.3)
                elif turn_status == 1:
                    time.sleep(0.3)
                    servo.turnLeft(angle_rate)
                else:
                    time.sleep(0.3)
                    servo.turnRight(angle_rate)
                move.move(speed, 'backward')
                backing = 1
                # time.sleep(0.2)
            else:
                time.sleep(0.1)
                check_true_out = 1

if __name__ == '__main__':
    try:
        setup()
        move.setup()
        while 1:
            run()
        pass
    except KeyboardInterrupt:
		    move.destroy()
