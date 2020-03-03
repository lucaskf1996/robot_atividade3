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

# STATE MACHINE TO CONTROL OUTPUT:
# ------------------------------------------------
# ------------------------------------------------
SHOW_BRISK = 0                                #--- Mostra a captura analisada com BRISK
SHOW_BASE = 0                                 #--- Mostra somente a captura direta
SHOW_MAG_MASK = 0                             #--- Mostra a captura da cor magente
SHOW_BLU_MASK = 0                             #--- Mostra a captura da cor azul
SHOW_LINES_DIST = 0                           #--- Mostra os círculos, assim como a linha entre os dois, angulo entre eles e a horizontal(graus), e distância entre eles e a câmera(cm)
SHOW_BITWISE_MAGBLU = 0                       #--- Mostra a junção das máscaras azul e magenta. Não funciona muito bem, mas é interessante.
# ------------------------------------------------
# ------------------------------------------------

# Setup webcam video capture
cap = cv2.VideoCapture("vid2.mp4")
time.sleep(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
    mask = cv2.inRange(cv2.GaussianBlur(frame,(5,5),0),np.array([0,0,200]),np.array([180,70,255]))
    morphMask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((6, 6)))
    contornos, arvore = cv2.findContours(morphMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_out = cv2.drawContours(morphMask, contornos, -1, [0, 0, 255], 3)
    return frame_out

def calcula_coef(x1, y1, x2, y2):
    dy = (y1 - y2)
    dx = (x1 - x2)
    direcao = 0
    if dx != 0:
        coef_angular = dy/dx
    else:
        coef_angular = 0
    if coef_angular > 0.3:
        direcao = 1
    elif  coef_angular < -0.3:
        direcao = -1
    return direcao


running = True
frameCount = 0
buffering = 3
lista_goodLeft = [0]*buffering
lista_goodRight = [0]*buffering

while running:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    maskedFrame = treatForLines(frame_hsv)
    bordas = auto_canny(maskedFrame)

    lines = cv2.HoughLines(bordas, 1, np.pi/180, 100)
    if lines is not None:
        # print(lines)
        # break
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = [int(x0 + 1000*(-b)), int(y0 + 1000*(a))]
                pt2 = [int(x0 - 1000*(-b)), int(y0 - 1000*(a))]
                # cv2.line(maskedFrame,pt1,pt2,(255,0,0),2)
                state = calcula_coef(pt1[0], pt1[1], pt2[0], pt2[1])
                if  state == -1:
                    lista_goodLeft.pop(0)
                    lista_goodLeft.append((pt1,pt2))
                if state == 1:
                    lista_goodRight.pop(0)
                    lista_goodRight.append((pt1,pt2))

        average_Left = lista_goodLeft[np.random.randint(buffering)]
        average_Right = lista_goodRight[np.random.randint(buffering)]
        if 0 not in lista_goodLeft and 0 not in lista_goodRight:
            cv2.line(frame,tuple(average_Left[0]),tuple(average_Left[1]),(255,0,0),2)
            cv2.line(frame,tuple(average_Right[0]),tuple(average_Right[1]),(255,0,0),2)
    # Display the resulting frame
    if SHOW_BASE:
        cv2.imshow('Detector de circulos',frame)
    else:
        cv2.imshow('Detector de circulos',frame)
    # time.sleep(0.05)
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    frameCount += 1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
