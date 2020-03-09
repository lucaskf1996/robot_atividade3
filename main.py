#!/usr/bin/python

# -*- coding: utf-8 -*-

# __author__      = "Matheus Dib, Fabio de Miranda" ==> Modificado
__author__ = "Carlos Dip, João Andrade, Lucas Fukada"


# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from classes import Point, Line


# Setup webcam video capture
cap = cv2.VideoCapture("vid2.mp4")
time.sleep(1)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def treatForLines(frame):
    # Shape detection using color (cv2.inRange masks are applied over orginal image)
    mask = cv2.inRange(cv2.GaussianBlur(frame,(5,5),0),np.array([30,60,220]),np.array([255,255,255]))
    morphMask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((6, 6)))
    contornos, arvore = cv2.findContours(morphMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_out = cv2.drawContours(morphMask, contornos, -1, [0, 0, 255], 3)
    return frame_out

running = True
frameCount = 0
buffering = 15
lista_goodLeft = [0]*buffering
lista_goodRight = [0]*buffering

while running:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    maskedFrame = treatForLines(frame)
    bordas = auto_canny(maskedFrame)

    lines = cv2.HoughLines(bordas, 1, np.pi/180, 180)

    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = Point(int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = Point(int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            lin = Line(pt1, pt2)
            # cv2.line(maskedFrame,pt1,pt2,(255,0,0),2)

            if lin.m < -0.2:
                lista_goodLeft.pop(0)
                lista_goodLeft.append(lin)
            elif lin.m > 0.2:
                lista_goodRight.pop(0)
                lista_goodRight.append(lin)
    
   #print(lista_goodLeft, lista_goodRight)

    average_Left = lista_goodLeft[np.random.randint(buffering)]
    average_Right = lista_goodRight[np.random.randint(buffering)]
    if 0 not in lista_goodLeft and 0 not in lista_goodRight:
        #print(tuple(average_Left[0]),tuple(average_Left[1]))
        a, b = average_Left.getPoints()
        c, d = average_Right.getPoints()
        cv2.line(frame, a, b,(255,0,0),2)
        cv2.line(frame, c, d,(255,0,0),2)
        inter = average_Left.intersect(average_Right)
        print(inter)
        cv2.circle(frame, inter, 5,(0,255,255), 5)

        
    # Display the resulting frame
    cv2.imshow('Detector de circulos',frame)
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    frameCount += 1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
