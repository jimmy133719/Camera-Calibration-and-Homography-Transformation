# Camera-Calibration-and-Homography-Transformation
### Camera_Calibration.py

##### Packages used:
1. numpy: array operation
2. cv2: load, save, show image
3. visualize: visualize camera pose
4. math: generate some special arithmetic operators 

##### Function:
1. compute_P: 
```
    description:
        compute projection matrix of 2d and 3d points
    input: 
        pt_2d: 2d point array(36,2)
        pt_3d: 3d point array(36,3)
    output: 
        A: matching matrix(72,12)
	P: projection matrix(3,4)
 ```
2. compute_KR:
```
    description:    
        compute QR decomposition to get K,R,t
    input: 
        P: projection matrix(3,4)
    output: 
        K: intrinsic matrix(3,3)
        R: rotation matrix(3,3)
        t: translation(3,)
```
3. reproject: 
```
    description:    
        reproject 2d point from 3d point and K,R,t
    input:
        pt_3d: 3d point array(36,3)
        K: intrinsic matrix(3,3)
        R: rotation matrix(3,3)
        t: translation(3,)  
    output:
        KRt: combined projection matrix(3,4)
	pt_reproject: reprojected 2d points
```

##### Procedure:
1. get labeled 2d points array by running "clicker.py"     
2. get 3d points from "Point3D.txt"
3. compute projection matrix from 2d points and 3d points
4. apply QR decomposition to projection matrix
5. reproject 2d points from combined projection matrix
6. compute RMS error
7. visualize camera pose

-----------------------------------------------------------------------------------

### Homography_Transformation.py

##### Packages used:
1. numpy: array operation
2. cv2: load, save, show image
3. matplotlib.path: form polygon according to points

##### Function:
1. compute_h: 
```
    description:
        compute homography transformation matrix of two planes
    input: 
        pt_s, pt_t: plane corner points array(4,2)
    output: 
        A: matching matrix(8,9)
	h: homography transformation matrix(3,3)
```
2. backprop: 
```
    description:    
        transformation through backward warping
    input: 
        img: original image(W,H,3)
	pt_obj1, pt_obj2: plane corner points array(4,2)
	h: homography transformation matrix(3,3) 
    output: 
        img_trans(img1_trans/img2_trans): transformed image(W,H,3)
```
3. forwardprop: 
```
    description:    
        transformation through forwardward warping
    input: 
        img: original image(W,H,3)
	pt_obj1, pt_obj2: plane corner points array(4,2)
	h: homography transformation matrix(3,3) 
    output: 
        img_trans(img1_trans/img2_trans): transformed image(W,H,3)
```
##### Procedure:
1. load image
2. get labeled points array by running "clicker.py"
3. compute homography transformation matrix
4. switch the planes with transformation through backward(forward) warping 
