[//]: # (Image References)

[image1]: ./output_images/camera_undistort.jpg "Undistorted"
[image2]: ./output_images/pipeline_undistort.jpg "Road Transformed"
[image3]: ./output_images/color_spaces2.jpg "Color Spaces"
[image4]: ./output_images/gradients1.jpg "Gradients"
[image5]: ./output_images/thresholded.jpg "Thresholded"
[image6]: ./output_images/top_down.jpg "Top Down"
[image7]: ./output_images/lines.jpg "Lines"
[image8]: ./output_images/result.jpg "Result"
[video1]: ./project_video.mp4 "Video"

#Advanced Lane Finding Project

### Camera Calibration

The code for this step is contained in the calibrate_camera function main.py.  

This function works loading a directory of chessboard images of varying camera angles/distoritions. For each image I first grayscale it then attempt to search for chessboard corners.
If any corners were detected, I append the new image points(2D) and corresponding object points(3D) into arrays.
After each calibration image has been processed I then call cv2.calibrateCamera which returns the camera matrix, distortion coefficients, and rotation/translation vectors.
Since this process is relatively slow I then pickle the camera matrix and distortion coefficients so that in future runs of the program I can just load the previously calculated results.
To undistort an image I then just need to call cv2.undistort passing in the camera matrix and distortion coefficients.

![alt text][image1]

### Pipeline

#### 1. Image Undistortion

The first step of the pipeline is to undistort the input camera image. I do this by loading pickled data containing the previously calibrated camera matrix and distortion coefficients and then calling cv2.undistort. This code is in the undistort function in main.py.

![alt text][image2]

#### 2. Color / Gradient Thresholding

The next step was to apply some color/gradient thresholding to the undistorted image in order to extract as much of the lane lines as possible from each image.
This involved a bit of exploration of the various color spaces and combining with different gradient techniques. The first thing I did was just plot a single image in 9 different color spaces and figure out which spaces extracted the lane lines the best.
The color spaces I test were RGB, HSL, and YUV. Out of these spaces Red and Saturation channels were pretty good at extracting the lines. I also noticed that the V channel was really good at not detecting the yellow lane line. I decided to take the inverse of the V channel and got a result that was really good at extracting the yellow lane line.

![alt text][image3]

I also tested sobel gradient in x and y directions, magnitude and direction of the gradients.
 
![alt text][image4]

In the end I decided I could get the most out of just using color thresholding by combining the red, saturation and inverse v channels.
The code for this step is in threshold_image in main.py

![alt text][image5]

#### 3. Perspective Transform

The next step in the pipeline was to take the thresholded image and transform it to a top down view. I did this by first plotting a straight line road and determining what perspective source points I should use.
Then for the destination points I just specified a rectangle slightly smaller than the image size. 
After I determine the source and destination points it was just a matter of calling cv2.warpPerspective on the image.
The code for this is in transform_top_down and perspectiveTransform in main.py.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 470      | 200, 100      | 
| 716, 470      | 1080, 100     |
| 1030,670      | 1080, 620     |
| 280, 670      | 200, 620      |

![alt_text][image6]


#### 4. Sliding Window

The next step was to identify the left and right lane lines and fit separate polynomials to them. I went with the sliding window approach. I created a Line class which remembers line data about previous frames.
If lines didn't detect a line in the previous frame, it runs through the sliding window algorithm. First I calculate the histogram of the bottom half of the image. Next I determine the peak left and right values in the histogram by getting the max value left of the midpoint and the max value right of the midpoint.
I set these peaks to be the base positions for the first window. I determine that I want to divide the image into 9 windows 100 pixels wide. Starting with the calculated base position I draw a window around it and store off each non zero pixel within the window.
Then if there were more than 50 pixels found in the window I recalculate the base position to be the mean of those pixels.
Once I have done this for each 9 windows I fit a polynomial for the left and right pixels.

The next frame I can check for pixels within a near margin of the calculated polynomials rather than doing another sliding window search.
In order to make sure the lane was rendered smoothly I averaged the last 5 frames together.
This code was implemented in slidingWindow function in main.py.

![alt text][image7]

#### 5. Radius of Curvature

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Result

To get the final result I just needed to take the image with the lines and lane rendered and do an inverse perspective transform to get it back to the original perspective.
Then I just combined that with the original image.

![alt text][image8]

---

### Pipeline (video)

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

My approach to this project was to first implement the simplest solution and see how well that performed. With just applying a sobel gradient the pipeline worked fairly well. However, it failed when coming to very light or dark images.
This lead me to pursue a more in-depth exploration of color/gradient thresholding. I plotted many different color spaces including RGB, HSL, YUV and LAB. I also, plotted Sobel X, Sobel Y, Magnitude, and Directional gradients. As discussed above, I determined that I could do pretty well with combining the Red, Saturation and inverse V channels.
An improvement to this project would be to experiment with more challenging videos that introduced areas that were missing lane lines, or a night or even in inclement weather conditions. And maybe in those conditions other gradients would need to be applied.

The other major improvement would be do a weighted average of the previous frames giving more weight to more recent frames. Also, potentially giving more weight to higher confidence lines.
What determines a high confidence line is undetermined, but I could start by simply counting the number of pixels in the line. I stubbed out code to check if a line is valid and to only add it if it is valid.
However, I didn't get time to implement the validation function. This improvement could make it so I don't add low confidence lines to my history resulting in bad lane rendering.