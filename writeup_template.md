##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.jpg "Undistorted"
[image2]: ./output_images/test1_undist.jpg "Road Transformed"
[image3]: ./output_images/test1_binary.jpg "Binary Example"
[image4]: ./output_images/straight_lines1_confirmation.jpg "Warp Example"
[image5]: ./output_images/test1_out_img.jpg "Fit Visual"
[image6]: ./output_images/test1_result.jpg "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 11 through 36 of the file called `lane_finding.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To undistort an image I use cv2.undistort, which takes in the image, a camera matrix and a distortion coefficient. The camera matrix and distortion coefficient are generated with cv2.calibrateCamera which uses the object and image points to generate those values. Here is an example of a undistorted image:

![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 56 through 77 in `lane_finding.py`). I transformed my color space to HLS. Then I applied Sobel x on the L-Channel and color thresholded the S-Channel. After that both thresholds are stacked on each other. Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in lines 82 through 89 in the file `lane_finding.py`. It takes as inputs an image (`binary`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[585, 460],
    [203, 720],
    [1127, 720],
    [705, 460]])
dst = np.float32(
    [[320, 0],
    [320, 720],
    [960, 720],
    [960, 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

So to identify lane_line pixels first we have to apply a histogram over the image to identify peaks (which in this case are all pixel, that are not black). Now we use 9 "windows", which function as little histograms to identfy all lane line pixel over the whole width of the image. Right after the windows gone over the image (line 141-154) good points (indices) are stored in the left_land_inds and right_lane_inds list (line 156-157), if they pass the "bigger than minpix" function x and y get extracted (line 159-171). After that a second order polynomial is created out of those points (line 173-174). The end result looks like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For the curvature of the lane I took the curvature of the right and left lane and converted them to curvature in meters, then I added those up and divided the result by 2 (line 271). For the postion of the vehicle with respect to  the center of the lane I took the most outter points of the detected area between the two lines ot the lane (the green projection on the road) and subtracted these from the center of the camera (half of the length of the image). The result was the difference of the car's position to the center of the lane (line 304-316).

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 276 through 291 in my code in `lane_finding.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I put the result in my project folder!

Here: (./project_video_result.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

  It took the most time to find appropriate values for the warping process, I ended up coming to almost the same result as in the writeup_template example. Also the funetuning of all parameters took some time. I increased the margin of the windows  to 130 and decreased the margin for the search for pixel in range to 80. Which resulted in better performance on the project video.
  My pipeline works really well on roads with small curvature (eg. highways). But on curvy roads my pipeline wouldn't do so well, because I use a region of interest mask to reduce noise from light highway walls or car wheels and this does restrict the ability to detect harsh turns (because they would be masked by the region of interest mask). To make my pipeline more robust I would implement a better perfoming color threshold and work on something that smooths out fast curvature changes, because most of the time the direction of the road doesn't change in one frame and in the next it's back to the old direction (this just happens when a line gets detected wrong, outlier). I think this would improve my pipeline!

