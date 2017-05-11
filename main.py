import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
from Line import Line
from moviepy.editor import VideoFileClip

# -------------------------------------------------------------------
# Globals
leftLine = Line()
rightLine = Line()


# -------------------------------------------------------------------
# Calibrate the camera
#
# param     rows            Number of rows in the chessboard images
# param     cols            Number of columns in the chessboard images
# param     image_shape     size of the image
# returns                   The camera matrix, and distortion coefficients
#
def calibrate_camera(rows, cols, image_shape, load=False):
    if load == True:
        camera_calibration = pickle.load(open("camera_calibration.p", "rb"))
        mtx = camera_calibration["mtx"]
        dist = camera_calibration["dist"]
    else:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((rows*cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:

            # Read Image
            img = cv2.imread(fname)

            # Convert to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (cols,rows),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

        # Save the calibration parameters for future sessions
        pickle.dump({"mtx": mtx, "dist": dist}, open("camera_calibration.p", "wb"))

    return mtx, dist


# -------------------------------------------------------------------
# Unwarp an image
#
# param     img             The image to unwarp
# returns                   The unwarped image
#
def undistort(img):
    # Calculate camera calibration parameters
    mtx, dist = calibrate_camera(9, 6, (img.shape[0], img.shape[1]), load=True)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# -------------------------------------------------------------------
# Get the perspective transform
#
# param     bInv      If true, return the inverse perspective transform
# returns             The perspective transform
#
def perspectiveTransform(bInv = False):
    src = np.float32([[570, 470], [716, 470], [1030, 670], [280, 670]])
    img_size = (1280, 720)
    offset = (200, 100)
    dst = np.float32([[offset[0], offset[1]], [img_size[0] - offset[0], offset[1]],
                      [img_size[0] - offset[0], img_size[1] - offset[1]],
                      [offset[0], img_size[1] - offset[1]]])
    if bInv:
        return cv2.getPerspectiveTransform(dst, src)
    else:
        return cv2.getPerspectiveTransform(src, dst)


# -------------------------------------------------------------------
# View the image from top-down
#
# param     img
#
def transform_top_down(img):
    top_down = cv2.warpPerspective(img, perspectiveTransform(), (1280, 720), flags=cv2.INTER_LINEAR)
    return top_down


# -------------------------------------------------------------------
# Creates a binary thresholded version of an image
#
# param     img     The image to be processed
# thresh            The min/max threshold to except
# returns           The binary image
#
def binary_threshold(img, thresh=(0,255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary


# -------------------------------------------------------------------
# Visualize various color spaces
#
# param     img             The RGB image to be processed
#
def explore_color_spaces(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    white = np.zeros_like(img)
    white.fill(255)
    y_channel = yuv[:, :, 0]
    u_channel = yuv[:, :, 1]
    v_channel = yuv[:, :, 2]

    rv_channels = r_channel | v_channel
    rs_channels = r_channel | s_channel
    sv_channels = s_channel | v_channel

    figure = plt.figure()
    figure.set_size_inches(8, 8)

    # -------------------------------------------------------
    # RGB
    ax1 = figure.add_subplot(3, 3, 1)
    ax1.set_title('Red')
    ax1.imshow(r_channel, cmap='gray')

    ax2 = figure.add_subplot(3, 3, 2)
    ax2.set_title('Green')
    ax2.imshow(g_channel, cmap='gray')

    ax3 = figure.add_subplot(3, 3, 3)
    ax3.set_title('Blue')
    ax3.imshow(b_channel, cmap='gray')

    # -------------------------------------------------------
    # HLS
    ax4 = figure.add_subplot(3, 3, 4)
    ax4.set_title('Hue')
    ax4.imshow(h_channel, cmap='gray')

    ax5 = figure.add_subplot(3, 3, 5)
    ax5.set_title('Saturation')
    ax5.imshow(s_channel, cmap='gray')

    ax6 = figure.add_subplot(3, 3, 6)
    ax6.set_title('Lightness')
    ax6.imshow(l_channel, cmap='gray')

    # -------------------------------------------------------
    # YUV
    ax7 = figure.add_subplot(3, 3, 7)
    ax7.set_title('Y')
    ax7.imshow(y_channel, cmap='gray')

    ax8 = figure.add_subplot(3, 3, 8)
    ax8.set_title('U')
    ax8.imshow(u_channel, cmap='gray')

    ax9 = figure.add_subplot(3, 3, 9)
    ax9.set_title('V')
    ax9.imshow(v_channel, cmap='gray')
    plt.savefig(os.path.join('output_images', 'color_spaces2.jpg'))

    # -------------------------------------------------------
    # LAB
    # ax10 = figure.add_subplot(3, 3, 7)
    # ax10.set_title('L')
    # ax10.imshow(lab1, cmap='gray')
    #
    # ax11 = figure.add_subplot(3, 3, 8)
    # ax11.set_title('A')
    # ax11.imshow(lab2, cmap='gray')
    #
    # ax12 = figure.add_subplot(3, 3, 9)
    # ax12.set_title('B')
    # ax12.imshow(lab3, cmap='gray')


# -------------------------------------------------------------------
# Visualize various gradients
#
# param     img             The RGB image to be processed
#
def explore_gradients(img):
    gradx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10, 150))
    grady_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(15, 100))
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(20, 100))
    dir_binary = dir_threshold(img, sobel_kernel=31, thresh=(np.pi * .4, np.pi * .6))

    combined1 = np.zeros_like(dir_binary)
    combined1[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    combined2 = np.zeros_like(dir_binary)
    combined2[((mag_binary == 1) & (dir_binary == 1))] = 1

    figure = plt.figure()
    figure.set_size_inches(5, 5)

    ax1 = figure.add_subplot(3, 3, 1)
    ax1.set_title('Sobel X')
    ax1.imshow(gradx_binary, cmap='gray')

    ax2 = figure.add_subplot(3, 3, 2)
    ax2.set_title('Sobel Y')
    ax2.imshow(grady_binary, cmap='gray')

    ax3 = figure.add_subplot(3, 3, 3)
    ax3.set_title('Magnitude')
    ax3.imshow(mag_binary, cmap='gray')

    ax4 = figure.add_subplot(3, 3, 4)
    ax4.set_title('Directional')
    ax4.imshow(dir_binary, cmap='gray')

    ax5 = figure.add_subplot(3, 3, 5)
    ax5.set_title('Combined 1')
    ax5.imshow(combined1, cmap='gray')

    ax6 = figure.add_subplot(3, 3, 6)
    ax6.set_title('Combined 2')
    ax6.imshow(combined2, cmap='gray')

    plt.savefig(os.path.join('output_images', 'gradients1.jpg'))


# -------------------------------------------------------------------
# Apply a sobel directional gradient
#
# param     img             The image to be processed
# param     orient          Which axis to apply the gradient
# param     sobel_kernel    kernel size of the sobel function
# param     thresh          Threshold of values to output
# returns                   The processed image
#
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary


# -------------------------------------------------------------------
# Apply a gradient magnitude threshold
#
# param     img             The image to be processed
# param     sobel_kernel    kernel size of the sobel function
# param     thresh          Threshold of values to output
# returns                   The processed image
#
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    scaled_sobel = np.uint8(255 * mag / np.max(mag))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return sbinary


# -------------------------------------------------------------------
# Apply a gradient direction threshold
#
# param     img             The image to be processed
# param     sobel_kernel    kernel size of the sobel function
# param     thresh          Threshold of values to output
# returns                   The processed image
#
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# -------------------------------------------------------------------
# Apply color/gradient threshold to the image
#
# param     img             The RGB image to be processed
# returns                   The thresholded image
#
def threshold_image(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    r_channel = binary_threshold(img[:, :, 0], (220, 255))
    s_channel = binary_threshold(hls[:, :, 2], (170, 255))
    white = np.zeros_like(img)
    white.fill(255)
    v_channel = binary_threshold(white[:, :, 2] - yuv[:, :, 2], (150, 255))

    combined = np.zeros_like(gray)
    combined[((r_channel == 1) | (s_channel == 1) | (v_channel == 1))] = 1
    return combined


# ----------------------------------------------------------
# TODO : implement
# Returns whether or not the current frame
#
# Checking that they have similar curvature
# Checking that they are separated by approximately the right distance horizontally
# Checking that they are roughly parallel
#
def isFrameValid(leftLine, rightLine):
    # Checking that they have similar curvature
    bSimilarCurvature = True

    # Checking that they are separated by approximately the right distance horizontally
    bCorrectDistance = True

    # Checking that they are roughly parallel
    bParallel = True

    return bSimilarCurvature and bCorrectDistance and bParallel


# -------------------------------------------------------------------
# Sliding window technique
#
# param     img         Img to perform sliding window on
#
def slidingWindow(img):
    global leftLine, rightLine

    # if we haven't detected a line
    if leftLine.detected is False or rightLine.detected is False:
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)

        leftLine.line_base_pos = np.argmax(histogram[:midpoint])
        rightLine.line_base_pos = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftLine.line_base_pos - margin
            win_xleft_high = leftLine.line_base_pos + margin
            win_xright_low = rightLine.line_base_pos - margin
            win_xright_high = rightLine.line_base_pos + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftLine.line_base_pos = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightLine.line_base_pos = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    # If we already have line data from a previous frame
    # use that information for this frame
    # The new line should be somewhat near the previous line
    else:
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
        (nonzerox > (leftLine.best_fit[0] * (nonzeroy ** 2) + leftLine.best_fit[1] * nonzeroy + leftLine.best_fit[2] - margin)) & (
        nonzerox < (leftLine.best_fit[0] * (nonzeroy ** 2) + leftLine.best_fit[1] * nonzeroy + leftLine.best_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (rightLine.best_fit[0] * (nonzeroy ** 2) + rightLine.best_fit[1] * nonzeroy + rightLine.best_fit[2] - margin)) & (
        nonzerox < (rightLine.best_fit[0] * (nonzeroy ** 2) + rightLine.best_fit[1] * nonzeroy + rightLine.best_fit[2] + margin)))

    # Fit a second order polynomial to each
    leftLine.currentx = nonzerox[left_lane_inds]
    rightLine.currentx = nonzerox[right_lane_inds]

    leftLine.current_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
    rightLine.current_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)

    # if the frame is valid
    # add the data to the line history
    if isFrameValid(leftLine, rightLine):
        leftLine.updateData()
        rightLine.updateData()
        leftLine.detected = True
        rightLine.detected = True
    else:
        leftLine.detected = False
        rightLine.detected = False

    # Now get the average line from the line history
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)

    # Generate x and y values for plotting
    ploty = np.linspace(0, window_img.shape[0] - 1, window_img.shape[0])
    left_fitx = leftLine.best_fit[0] * ploty ** 2 + leftLine.best_fit[1] * ploty + leftLine.best_fit[2]
    right_fitx = rightLine.best_fit[0] * ploty ** 2 + rightLine.best_fit[1] * ploty + rightLine.best_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, perspectiveTransform(True), (img.shape[1], img.shape[0]))

    return newwarp


def pipeline(image):

    # Undistort the image
    undistorted = undistort(image)

    explore_color_spaces(undistorted)
    explore_gradients(undistorted)

    thresholded = threshold_image(undistorted)

    # Transform the image to a top down view
    top_down = transform_top_down(thresholded)

    # Detect the lane line using a sliding window technique
    newwarp = slidingWindow(top_down)

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    return result


# ---------------------------------------------
# Execute Code Here
# Run tests and image processing pipeline here
#
def main():
    images = []
    input_files = ['test_images/test6.jpg']

    # Read in an image
    for input_file in input_files:
        image = mpimg.imread(input_file)
        images.append(image)

    for image in images:
        pipeline(image)

    # output_video = 'output_video2.mp4'
    # input_clip = VideoFileClip("project_video.mp4")
    # output_clip = input_clip.fl_image(pipeline)  # NOTE: this function expects color images!!
    # output_clip.write_videofile(output_video, audio=False)


if __name__ == '__main__':
    main()