# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 08:38:16 2017

@author: Dharmesh Jani
"""

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
from tracker import tracker

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.

# Useful functions for producing the binary pixel of interest images to feed into the LaneTracker Algorithm
def abs_sobel_thresh(img, orient='x', sobel_kernel=3,thresh=(0,255)):
   # Gradient in x and y direction
    
    imghsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Use different approach for using Sobel Operator
    #Channels L and S from HLS
    sobelx1 = cv2.Sobel(imghsl[:,:,1], cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobelx2 = cv2.Sobel(imghsl[:,:,2], cv2.CV_64F, 1,0, ksize=sobel_kernel)
        
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelx1 = np.uint8(255*sobelx1/ np.max(sobelx1))
    scaled_sobelx2 = np.uint8(255*sobelx2/ np.max(sobelx2))

    # 5) Create a binary mask where mag thresholds are met
    binary_outputx1 = np.zeros_like(scaled_sobelx1)
    binary_outputx1[(scaled_sobelx1 >= thresh[0]) & (scaled_sobelx1 <= thresh[1])] = 1

    binary_outputx2 = np.zeros_like(scaled_sobelx2)
    binary_outputx2[(scaled_sobelx2 >= thresh[0]) & (scaled_sobelx2 <= thresh[1])] = 1

    binary_output = np.zeros_like(scaled_sobelx1)
    binary_output[(binary_outputx1 ==1) | (binary_outputx2 ==1)]=1
    # 6) Return this mask as your binary_output image
    return binary_output
    
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Check magnitude of gradient in both x and y directions
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output
    
def dir_threshold(img, sobel_kernel=15, thresh=(0.7,1.41)):
    # Check direction of gradient in x and y
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
   
    absgraddir = np.arctan2(abs_sobely,abs_sobelx)
    
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output
#def hls_color_thresh(img, threshH,threshL, threshS):
def hls_color_thresh(img, threshLow, threshHigh):
    # 1) Convert to HLS color space
    #imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #Hue (0,180) Light (0,255), satur (0,255)

   
    # 3) Return a binary image of threshold result
    binary_output = np.zeros((img.shape[0], img.shape[1]))
    #binary_output[(imgHLS[:,:,0] >= threshH[0]) & (imgHLS[:,:,0] <= threshH[1]) & (imgHLS[:,:,1] >= threshL[0]) & (imgHLS[:,:,1] <= threshL[1])  | ((imgHLS[:,:,2] >= threshS[0]) & (imgHLS[:,:,2] <= threshS[1]))] = 1
    binary_output[(imgHLS[:,:,0] >= threshLow[0]) & (imgHLS[:,:,0] <= threshHigh[0]) & (imgHLS[:,:,1] >= threshLow[1])  & (imgHLS[:,:,1] <= threshHigh[1])  & (imgHLS[:,:,2] >= threshLow[2]) & (imgHLS[:,:,2] <= threshHigh[2])] = 1
                 
    return binary_output

def color_thresh1(img, sthresh=(0, 255), vthresh=(0,255)):
    """
    check color saturation
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    # Based on lectures use red channel based thresholds for  yellow and white lanes filtering
    r_channel = img[:,:,0]
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1]) & (r_channel >= 200) & (r_channel <= 255)] = 1
    #s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    ## We can also use v channel from HSV to calibrate on color thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1]) ] = 1

    g_thresh = (250,255)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_binary = np.zeros_like(gray)
    gray_binary[(gray>g_thresh[0]) & (gray<= g_thresh[1])] = 1
    
    output = np.zeros_like(s_channel)
    output[(s_binary==1)&(v_binary==1)]=1
    return output

def color_thresh2(image, sthresh=(0,255), vthresh=(0,255), lthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,2]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= lthresh[0]) & (l_channel <= lthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1) & (l_binary == 1)] = 1
    return output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def fitlines(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)


    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
   
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
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
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    
    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit =[]
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
    
    if len(rightx) == 0:
        right_fit =[]
    else:
        right_fit = np.polyfit(righty, rightx, 2)
    

    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    return left_fit, right_fit,out_img

def fit_continuous(left_fit, right_fit, binary_warped):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit_updated =[]
    else:
        left_fit_updated = np.polyfit(lefty, leftx, 2)
    
    
    if len(rightx) == 0:
        right_fit_updated =[]
    else:
        right_fit_updated = np.polyfit(righty, rightx, 2)
        
    return  left_fit_updated, right_fit_updated

#Calc Curvature
def curvature(left_fit, right_fit, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    center = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +(right_fit[0]*720**2+right_fit[1]*720+right_fit[2]) ) /2 - 640)*xm_per_pix
    
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad, center

def sanity_check(left_fit, right_fit, minSlope, maxSlope):
    #Performs a sanity check on the lanes
    #Check 1: check if left and right fits exists
    #Check 2: Calculates the tangent between left and right in two points, and check if it is in a reasonable threshold
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    if len(left_fit) ==0 or len(right_fit) == 0:
        status = False
        d0=0
        d1=0
        #Previous fitlines routine returns empty list if not finds
    else:
        #Difference of slope
        L_0 = 2*left_fit[0]*460+left_fit[1]
        R_0 = 2*right_fit[0]*460+right_fit[1]
        d0 =  np.abs(L_0-R_0)

        L_1 = 2*left_fit[0]*720+left_fit[1]
        R_1 = 2*right_fit[0]*720+right_fit[1]
        d1 =  np.abs(L_1-R_1)

        
        if d0>= minSlope and d0<= maxSlope and d1>= minSlope and d1<= maxSlope:
            status = True
        else:
            status = False
            
    return(status, d0, d1)



global counter
counter=0
ref_left =np.array([-0.0001,0,400])
ref_right=np.array([-0.0001,0,1000])   
left_fit =np.array([-0.0001,0,400])
right_fit=np.array([-0.0001,0,1000])   


def process_image(img):
    
    global counter

    # undistort image
    img = cv2.undistort(img, mtx,dist,None,mtx)
    img_orig = img

    #2.Magnitude Threshold
    #Threshold color    
    yellow_low = np.array([0,100,100])
    yellow_high = np.array([50,255,255])
    white_low = np.array([18,0,180])
    white_high = np.array([255,80,255])
    global ref_left 
    global ref_right
    global left_fit
    global right_fit
    
   
    # Process image and generate binary pixel of interests
    imgThres_yellow = hls_color_thresh(img,yellow_low,yellow_high)
    imgThres_white = hls_color_thresh(img,white_low,white_high)
    gradx = abs_sobel_thresh(img,orient='x',sobel_kernel=9,thresh=(80,220)) 
    
    preprocessImage = np.zeros_like(img[:,:,0])
    preprocessImage[((gradx == 1) | (imgThres_yellow==1) | (imgThres_white==1))] = 255
    #preprocessImage[((gradx == 1) & (grady == 1) | (m_binary == 1) & (d_binary == 1) & (c_binary == 1))] = 255
    #c_binary = color_thresh1(img, sthresh=(100,255), vthresh=(50,255))
    #c_binary = color_thresh2(img, sthresh=(50,255), vthresh=(100,255), lthresh=(50,255))
   # m_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0,25))
    #d_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.8,1.5))
    
    bin_image = preprocessImage
    
    # Define the region parameters taken from https://github.com/wonjunee/Advanced-Lane-Finding
    # Masking of the binary preprocessed image
    imshape = img.shape
    left_bottom = (100, imshape[0])
    right_bottom = (imshape[1] - 20, imshape[0])
    apex1 = (610, 410)
    apex2 = (680, 410)

    inner_left_bottom = (310, imshape[0])
    inner_right_bottom = (1150, imshape[0])
    inner_apex1 = (700, 480)
    inner_apex2 = (650, 480)

    vertices = np.array([[left_bottom, apex1, apex2, \
                          right_bottom, inner_right_bottom, \
                          inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)
    # Masked area
    preprocessImage = region_of_interest(bin_image, vertices)
    
    
    # Definition for points to use for perspective transform
    img_size = (img.shape[1],img.shape[0])
    # img.shape[1] # This is 1280, Xmax
    # img.shape[0] # This is 720, Ymax
    # Generate Binary Warped Image using perspective transform
    #Tweaking source and destination for 4 corners of the trapezoid per reviewer comment
    src = np.float32([[585, 450], [203, 720], [1127, 720], [685, 450]])
    dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    ########
    
    if counter==0:
        left_fit, right_fit,out_imgfit = fitlines(warped)
    else:
        left_fit, right_fit = fit_continuous(left_fit, right_fit, warped)
    
#    left_fit,right_fit,out_img = fitlines(warped)

    status_sanity, d0, d1 =sanity_check(left_fit, right_fit, 0, .55)

    #Calc curvature and center
    if status_sanity  == True:        
        #Save as last reliable fit
        ref_left, ref_right = left_fit, right_fit        
        counter+=1
    else:        #Use the last realible fit
        left_fit, right_fit = ref_left, ref_right
        
    left_curv, right_curv, center_off = curvature(left_fit, right_fit, warped)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
    
    window_width =  25  #60
    window_height = 80  #80 based on Udacity classlass 
    # Use Offset for window sliding
    Offset = 25
    
    ## Set up the overall class to do all the tracking
    curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = Offset, My_ym = 30/720, My_xm = 3.7/700, Mysmooth_factor = 15)
    
    window_centroids = curve_centers.find_window_centroids(warped)
    
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    
    # points used to find the left and right lanes
    rightx = []
    leftx = []
    
    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        # add center value found in frame to the list of lane points per left, right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
	    # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255
   
    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green (R,G,B) with R and B being zeros
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
    green_boxes = result # debug holder

    ## Fitting the lane boundaries to the left, right and center positions found
    yvals = range(0,warped.shape[0])
    #yvals = np.linspace(0, 719, num=720)
    #print(warped.shape[0])
    #res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)


    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    middel_marker = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)

    cv2.fillPoly(road, [left_lane], color=[255,0,0])
    cv2.fillPoly(road, [right_lane], color=[0,0,255])
    cv2.fillPoly(road, [middel_marker], color=[0,255,0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255,255,255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255,255,255])

    road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img_orig, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)  # Setting up final overlaid image with lane markers

    #ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dim
    #xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dim

    # Track left lane curvature as we are using leftx
   # curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
    #curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
    
    
    # Calculate the offset of the car on the road
    #camera_center = (left_fitx[-1] + right_fitx[-1])/2
    #center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_off <= 0:
        side_pos = 'right'

    avg_curv = (left_curv+right_curv)/2
    # draw the text showingn curvature, offset and speed
    cv2.putText(result, 'Radius of Curvature = ' +str(round(avg_curv,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result,'Vehicle is '+str(abs(round(center_off,3))) +'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    # Final combination that works best includes
    # color_thresh1 based on s and v channel and preprocessing which includes gradx, grady and c_binary
    
    #result = bin_image
    #result = road
    #result = green_boxes
    #result = warped
    
    return result

Input_video = 'harder_challenge_video.mp4'
#Input_video = 'output1_tracked.mp4'
#Input_video = 'project_video.mp4'
#Input_video = 'challenge_video.mp4'

#Output_video = 'output1_new_updated.mp4'
#Output_video = 'output_challenge_updated.mp4'
Output_video = 'output_harder_challenge_updated.mp4'

#clip1 = VideoFileClip(Input_video).subclip(20,23)
clip1 = VideoFileClip(Input_video)

video_clip = clip1.fl_image(process_image)   # Function expects color images
video_clip.write_videofile(Output_video, audio=False)


