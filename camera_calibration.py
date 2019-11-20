# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:26:54 2019

@author: Jimmy
"""

import cv2
import numpy as np
import visualize
import math

def compute_P(pt_2d, pt_3d):
    A = []
    for n in range(pt_2d.shape[0]):
        A.append([[pt_3d[n][0],pt_3d[n][1],pt_3d[n][2],1,0,0,0,0,-pt_2d[n][0]*pt_3d[n][0],-pt_2d[n][0]*pt_3d[n][1],-pt_2d[n][0]*pt_3d[n][2],-pt_2d[n][0]],
                  [0,0,0,0,pt_3d[n][0],pt_3d[n][1],pt_3d[n][2],1,-pt_2d[n][1]*pt_3d[n][0],-pt_2d[n][1]*pt_3d[n][1],-pt_2d[n][1]*pt_3d[n][2],-pt_2d[n][1]]])
    A = np.array(A)
    A = A.reshape(A.shape[0]*A.shape[1],A.shape[2])
    
    # SVD
    u, s, vh = np.linalg.svd(A)

    # get the smallest singular value of SVD
    P = vh[-1,:].reshape(3,4)

    return A, P

def compute_KR(P):
    M = P[0:3,0:3]
    # QR decomposition
    q, r = np.linalg.qr(np.linalg.inv(M))
    R = np.linalg.inv(q)
    K = np.linalg.inv(r)
    # translation vector
    t = np.dot(np.linalg.inv(K),P[:,-1])
    
    D = np.array([[np.sign(K[0,0]),0,0],
              [0,np.sign(K[1,1]),0],
              [0,0,np.sign(K[2,2])]])
    
    # K,R,t correction
    K = np.dot(K, D)
    R = np.dot(np.linalg.inv(D), R)
    t = np.dot(np.linalg.inv(D), t)    
    t = np.expand_dims(t,axis=1)
    
    # normalize K
    K = K / K[-1,-1]
        
    return K, R, t

def reproject(pt_3d, K, R, t):
    pt_3d = np.hstack((pt_3d,np.ones((pt_3d.shape[0],1))))
    
    Rt = np.hstack((R,t))
    KRt = np.dot(K, Rt)
    
    pt_reproject = np.dot(KRt,pt_3d.T)
    pt_reproject = pt_reproject / pt_reproject[-1,:]
    
    return  KRt, pt_reproject#, extrinsic, projection, intrinsic, P_combined

if __name__ == '__main__':

    R_list = []
    t_list = []
    for i in range(1,3):
        # load 3d img
        img = cv2.imread('input images/chessboard_{}.jpg'.format(i))
        
        # load labeled 2d point
        pt_2d = np.load('Point2D_{}.npy'.format(i))
        
        # load labeled 3d point
        f = open('Point3D.txt','r')
        
        pt_3d_list = f.read().split('\n')
        
        pt_3d = np.array([c.split(' ') for c in pt_3d_list]).astype(int)
    
        # compute projection matrix
        A, P = compute_P(pt_2d, pt_3d)
    
        # compute QR decomposition
        K, R, t = compute_KR(P)
        
        # reproject
        KRt, pt_reproject = reproject(pt_3d, K, R, t)
        
        # compute RMS error
        RMS_e = math.sqrt(np.sum((pt_2d-pt_reproject[0:2].T)**2) / pt_2d.shape[0])
        print(RMS_e)
        
        R_list.append(R)
        t_list.append(t)
        
        # save projected points on image
        img_reprojected = img
        for j, row in enumerate(pt_reproject[0:2].T):
            cv2.circle(img,tuple(pt_2d[j]), 3, (0, 255, 255), 1)
            cv2.circle(img_reprojected, tuple(row.astype(int)), 2, (0,0,255), -1)
        cv2.imshow('reprojected',img_reprojected/255.0)
        #cv2.imwrite('result images/reprojected image_{}.jpg'.format(i),img_reprojected)
    
    # visualize
    
    visualize.visualize(pt_3d, R_list[0], t_list[0], R_list[1], t_list[1])
    