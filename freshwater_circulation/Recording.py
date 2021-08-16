# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:29:07 2020

@author: 邱妍敏
"""

import cv2, time, random, csv
import csv

time.sleep(30)
video=cv2.VideoCapture(0)
frDivider = 15
int_time = 43200*((time.time()-3600)//43200)+3600
timetag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/pi/Mud/Vid/A' + str(timetag) + '.avi', fourcc, 30.0, (640, 480))

framec = 0

try:
    while (video.isOpened()):
        check, frame = video.read()
        if check == True:
            if (time.time() - int_time) >= 43200:
                int_time = int_time + 43200
                out.release()
                timetag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                out = cv2.VideoWriter('/home/pi/Mud/Vid/A' + str(timetag) + '.avi', fourcc, 30.0, (640, 480))
            if not framec%frDivider:
                with open('/home/pi/Mud/timetag/A' + str(timetag) + '.csv', 'a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([str(time.time())])
                out.write(frame)
        else:
            break
        framec = framec%frDivider
        framec += 1
    video.release()
    out.release()
except:
    video.release()
    out.release()