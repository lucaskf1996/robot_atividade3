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
    mask = cv2.inRange(cv2.GaussianBlur(frame,(5,5),0),np.array([30,60,220]),np.array([255,255,255]))
    morphMask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((6, 6)))
    contornos, arvore = cv2.findContours(morphMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_out = cv2.drawContours(morphMask, contornos, -1, [0, 0, 255], 3)
    return frame_out

def calcula_coef(x1, x2, y1, y2):
    dy = (y1 - y2)
    dx = (x1 - x2)
    if dx != 0:
        coef_angular = dy/dx
    else:
        coef_angular = 0
    if coef_angular > 0.2:
        direcao = 1
    elif  coef_angular < -0.2:
        direcao = -1
    else:
        direcao = 0
    return direcao

def calcula_inter(rx1, ry1, rx2, ry2, sx1, sy1, sx2, sy2):
    dxr = rx1 - rx2
    dyr = ry1 - ry2
    dxs = sx1 - sx2
    dys = sy1 - sy2

    mr = None
    ms = None
    
    if dxr != 0:
        mr = dyr/dxr

    if dxs != 0:
        ms = dys/dxs

    _a = sx1/mr
    _b = (sy1 + ry1)/(ms-mr)
    _c = -rx1/ms
    out_x = int(_a + _b + _c)
    out_y = int(mr * out_x - mr * rx1 + ry1)
    return (out_x, out_y)

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
            pt1 = [int(x0 + 1000*(-b)), int(y0 + 1000*(a))]
            pt2 = [int(x0 - 1000*(-b)), int(y0 - 1000*(a))]
            # cv2.line(maskedFrame,pt1,pt2,(255,0,0),2)

            if calcula_coef(pt1[0], pt1[1], pt2[0], pt2[1]) == -1:
                lista_goodLeft.pop(0)
                lista_goodLeft.append((pt1,pt2))
            elif calcula_coef(pt1[0], pt1[1], pt2[0], pt2[1]) == 1:
                lista_goodRight.pop(0)
                lista_goodRight.append((pt1,pt2))
    
    # print(lista_goodLeft, lista_goodRight)

    average_Left = lista_goodLeft[np.random.randint(buffering)]
    average_Right = lista_goodRight[np.random.randint(buffering)]
    if 0 not in lista_goodLeft and 0 not in lista_goodRight:
        print(tuple(average_Left[0]),tuple(average_Left[1]))
        cv2.line(frame,tuple(average_Left[0]),tuple(average_Left[1]),(255,0,0),2)
        cv2.line(frame,tuple(average_Right[0]),tuple(average_Right[1]),(255,0,0),2)
        inter = calcula_inter(average_Right[0][0], average_Right[0][1], average_Right[1][0], average_Right[1][1], average_Left[0][0], average_Left[0][1], average_Left[1][0], average_Left[1][1])
        print(inter)
        cv2.circle(frame, inter, 5,(0,255,255), 5)
    # Display the resulting frame
    if SHOW_BASE:
        cv2.imshow('Detector de circulos',frame)
    else:
        cv2.imshow('Detector de circulos',frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    frameCount += 1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
