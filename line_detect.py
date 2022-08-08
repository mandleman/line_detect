# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:12:16 2022

@author: mandleman
@project name : Embedded AI hub in Daegu
"""


#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import cv2
import numpy as np
import copy
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)


def canny_edge(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)


img=cv2.imread('solidwhitecurve.jpg')

img = cv2.resize(img, (1000, 600))

#plt.figure(figsize=(10,8))
rows=6
cols=2
index=1
#fig = plt.figure(figsize=(8, 8))

print("this image is : ",type(img),'with dimensions : ',img.shape)
# plt.imshow(img)
# plt.show()
'''
fig.add_subplot(rows, cols, index)
plt.xlabel('origin')
plt.imshow(img)
index+=1
'''
#cv2.imshow('test',img)

gray=grayscale(img)
# plt.figure(figsize=(10,8))
print("this image is : ",type(gray),'with dimensions : ',gray.shape)
# plt.imshow(gray)
# plt.imshow(gray,cmap='gray')
# plt.show()
#fig.add_subplot(rows,cols,index)
#plt.xlabel('grayscale')
#plt.imshow(gray,cmap='gray')
#index+=1
#cv2.imshow('gray',gray)


kernel_size=9
blur_gray9=gaussian_blur(gray,kernel_size)

canny2=canny_edge(blur_gray9,100,200)

mask=np.zeros_like(gray)
cv2.imshow('mask',mask)
print('mask shape:',mask.shape)
print('canny shape:',canny2.shape)

imshape=img.shape
vertices=np.array([[(100,imshape[0]),
                      (450,320),
                      (550,320),
                      (imshape[1]-100,imshape[0])]],dtype=np.int32)
 
 
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


bitmask_img=region_of_interest(canny2,mask,vertices)

cv2.imshow('bitmask',bitmask_img)
canny3=canny_edge(bitmask_img,100,200)


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

canny3_blurred=gaussian_blur(canny3,kernel_size)
# houghline_img= copy.deepcopy(canny3_blurred)
rho=2
theta=np.pi/180
threshold=90
min_line_len=120
max_line_gap=150
lines=hough_lines(canny3_blurred,rho,theta,threshold,
                  min_line_len,max_line_gap)
#fig.add_subplot(rows,cols,index)
#plt.xlabel('hough line')
#plt.imshow(lines,cmap='gray')
#index+=1
####################################################################################
cv2.imshow('line detect',lines)

def weighted_img(img,img_initial,alpha=0.8,beta=1.,gamma=0.):
    return cv2.addWeighted(img_initial,alpha,img,beta,gamma)

lines_edge=weighted_img(lines,img)
#fig.add_subplot(rows,cols,index)
#plt.xlabel('weighted')
#plt.imshow(lines_edge)
#index+=1

##################################################################################
cv2.imshow('origin add lines',lines_edge)

# houghline_img= copy.deepcopy(img_resize)

# lines = cv2.HoughLines(canny3,1,np.pi/180,150)
# for line in lines:
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(houghline_img,(x1,y1),(x2,y2),(0,0,255),1)
# fig=plt.figure()
# plt.xlabel('hough line')
# plt.imshow(houghline_img,cmap='gray')
# index+=1


cv2.waitKey()
