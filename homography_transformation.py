# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:47:39 2019

@author: Jimmy
"""

import cv2
import numpy as np
import matplotlib.path as mplPath

def compute_h(pt_s, pt_t):
    A = []
    for n in range(pt_s.shape[0]):
        A.append([[pt_s[n][0],pt_s[n][1],1,0,0,0,-pt_t[n][0]*pt_s[n][0],-pt_t[n][0]*pt_s[n][1],-pt_t[n][0]],
                  [0,0,0,pt_s[n][0],pt_s[n][1],1,-pt_t[n][1]*pt_s[n][0],-pt_t[n][1]*pt_s[n][1],-pt_t[n][1]]])
    A = np.array(A)
    A = A.reshape(A.shape[0]*A.shape[1],A.shape[2])

    # SVD
    u, s, vh = np.linalg.svd(A)    
    
    # get the smallest singular value of SVD
    h = vh[-1,:].reshape(3,3)   
    
    return A, h

def backprop(img, pt_obj1, pt_obj2, h):
    img_trans = np.zeros(img.shape)
    # from pt_obj1 to pt_obj2
    pt_obj2[[2, 3]] = pt_obj2[[3, 2]]
    boundary = mplPath.Path(pt_obj2)
    for y in range(min(pt_obj2[:,1]),max(pt_obj2[:,1])):
        for x in range(min(pt_obj2[:,0]),max(pt_obj2[:,0])):
            if boundary.contains_point((x,y)):
                target = np.array([x,y,1])
                target_trans = np.dot(np.linalg.inv(h),target.T)
                target_trans = target_trans / target_trans[-1]
                img_trans[y,x,:] = img[int(round(target_trans[1])),int(round(target_trans[0])),:]
    
    # from pt_obj2 to pt_obj1
    pt_obj1[[2, 3]] = pt_obj1[[3, 2]]
    boundary = mplPath.Path(pt_obj1)
    for y in range(min(pt_obj1[:,1]),max(pt_obj1[:,1])):
        for x in range(min(pt_obj1[:,0]),max(pt_obj1[:,0])):
            if boundary.contains_point((x,y)):
                target = np.array([x,y,1])
                target_trans = np.dot(h,target.T)
                target_trans = target_trans / target_trans[-1]
                img_trans[y,x,:] = img[int(round(target_trans[1])),int(round(target_trans[0])),:]

    mask = img_trans==0
    img_trans = mask * img + img_trans
    
    return img_trans

def forwardprop(img, pt_obj1, pt_obj2, h):
    img_trans = np.zeros(img.shape)
    # from pt_obj1 to pt_obj2
    pt_obj1[[2, 3]] = pt_obj1[[3, 2]]
    boundary = mplPath.Path(pt_obj1)
    for y in range(min(pt_obj1[:,1]),max(pt_obj1[:,1])):
        for x in range(min(pt_obj1[:,0]),max(pt_obj1[:,0])):
            if boundary.contains_point((x,y)):
                target = np.array([x,y,1])
                target_trans = np.dot(h,target.T)
                target_trans = target_trans / target_trans[-1]
                img_trans[int(round(target_trans[1])),int(round(target_trans[0])),:] = img[y,x,:]
    
    # from pt_obj2 to pt_obj1
    pt_obj2[[2, 3]] = pt_obj2[[3, 2]]
    boundary = mplPath.Path(pt_obj2)
    for y in range(min(pt_obj2[:,1]),max(pt_obj2[:,1])):
        for x in range(min(pt_obj2[:,0]),max(pt_obj2[:,0])):
            if boundary.contains_point((x,y)):
                target = np.array([x,y,1])
                target_trans = np.dot(np.linalg.inv(h),target.T)
                target_trans = target_trans / target_trans[-1]
                img_trans[int(round(target_trans[1])),int(round(target_trans[0])),:] = img[y,x,:]

    mask = img_trans==0
    img_trans = mask * img + img_trans
    
    return img_trans

if __name__ == '__main__':

    # load img
    img = cv2.imread('input images/ImgA.jpg')
    
    # load labeled point
    pt_2d = np.load('Point2D_ImgA.npy')
    
    pt_obj1 = pt_2d[0:4,:]
    pt_obj2 = pt_2d[4:8,:]
    
    # compute homography transformation
    A, h = compute_h(pt_obj1, pt_obj2)
    
    # switch pt_obj1 and pt_obj2
    img_trans = backprop(img, pt_obj1, pt_obj2, h)
    #img_trans = forwardprop(img, pt_obj1, pt_obj2, h)
    
    cv2.imshow('transformed image', img_trans/255)
    #cv2.imwrite('result images/ImgA_backward.jpg',img_trans)
      
    