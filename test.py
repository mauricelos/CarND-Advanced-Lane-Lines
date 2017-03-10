import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


#### CALIBRATION OF THE CAMERA TO UNDISTORT IMAGES LATER ####

nx = 9
ny = 6

images = glob.glob('camera_cal/calibration*.jpg')

objpoints = []     #objpoints of succesfully detected chessboard corners
imgpoints = []     #correlating image points of succesfully detected chessboard corners

objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

for img in images:           #taking all images, perform grayscale on them and find chessboard corners
    
    image = mpimg.imread(img)
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)



image1 = mpimg.imread('test_images/straight_lines1.jpg')

cv2.line(image1, (585, 460), (203,720), (255,0,0))
cv2.line(image1, (1127,720), (705,460), (255,0,0))

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
undist = cv2.undistort(image1, mtx, dist, None, mtx) 


src = np.float32([[585, 460], [203,720], [1127,720], [705,460]])   #points on the original image
dst = np.float32([[320,0], [320,720], [960,720], [960,0]])        #destination point on the warped image

M = cv2.getPerspectiveTransform(src, dst)    #calculate matrix for transformation

imshape = image.shape

warped = cv2.warpPerspective(undist, M, (imshape[1],imshape[0]))


plt.imshow(warped)
#plt.savefig('output_images/straight_lines2_confirmation.jpg')
