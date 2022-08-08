# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:12:16 2022

@author: mandleman
@project name : Embedded AI hub in Daegu
"""


import argparse
parser=argparse.ArgumentParser(description='add video file path')
parser.add_argument('--input',type=str,default='car.mp4')
opt=parser.parse_args()

import cv2
import numpy as np
import copy
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)


def canny_edge(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)

 
def region_of_interest(img,mask,vertices):
    try:
        if len(img.shape)>2:
            channel_count=img.shape[2]
            ignore_mask_color=(255,)*channel_count
        else:
            ignore_mask_color=255
        cv2.fillPoly(mask,vertices,ignore_mask_color)
        masked_image=cv2.bitwise_and(img,mask)
        return masked_image
    except Exception as e:
        print(e)
        return img


def draw_lines(img,lines,color=[0,0,255],thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)

def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
   
    lines=cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),
                          minLineLength=min_line_len,
                          maxLineGap=max_line_gap)
    line_img=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img

def weighted_img(img,img_initial,alpha=0.8,beta=1.,gamma=0.):
    return cv2.addWeighted(img_initial,alpha,img,beta,gamma)


#############################################################################333
################################################################
#variables 
rho=2
theta=np.pi/180
threshold=90
min_line_len=120
max_line_gap=150


kernel_size=9
################################################################################
#init 
#print("imshape : ",imshape)
#example image :  600 x 1000
# imshape[0] == height 600
# imshape[1]== width 1000
path=opt.input
cap=cv2.VideoCapture(path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 5)
ret,img=cap.read()

imshape=img.shape
print(imshape)
vertices=np.array([[(int(imshape[1]/8),imshape[0]),
                      (int(imshape[1]*2/8),int(imshape[0]*2/5)),
                      (int(imshape[1]*6/8),int(imshape[0]*2/5)),
                      (int(imshape[1]*7/8),imshape[0])]],dtype=np.int32)

################################################################################
# while loop start , imread from video file
loop=True
while True:
    while(cap.isOpened()):
        ret,img=cap.read()
        if ret:
            try:
                gray=grayscale(img)
                blur_gray9=gaussian_blur(gray,kernel_size)
                #cv2.imshow('test',gray)
            
                canny2=canny_edge(blur_gray9,100,200)
                #cv2.imshow('canny',canny2)
                mask=np.zeros_like(gray)
                bitmask_img=region_of_interest(canny2,mask,vertices)
                lines=hough_lines(bitmask_img,rho,theta,threshold, min_line_len,max_line_gap)
                lines_edge=weighted_img(lines,img)
                cv2.imshow('origin add lines',lines_edge)
                if cv2.waitKey(33)&0xFF==ord('q'):
                    loop=False
                    break
            except Exception as e:
                print('line not exist')
        else:
            break
    cap.release()
    if loop==False:
        break
    else:
        print('restart')
    cap=cv2.VideoCapture(path)

##############################################

