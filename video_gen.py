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
    
   gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

   if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
   if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
   scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
   binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
   binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])]=1
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

def process_image(img):
    
    # undistort image
    img = cv2.undistort(img, mtx,dist,None,mtx)
    img_orig = cv2.undistort(img, mtx,dist,None,mtx)

    ##  Experimented with Masking Addition
    # Define the vertices of a mask. Image size is (720,1280) with 3 numbers (RGB)for each element
    imgx = img.shape[1] # This is 1280, Xmax
    imgy = img.shape[0] # This is 720, Ymax

    # The area of interest is defined as ratio of the image
    left_bottomx = 0.15*imgx  
    right_bottomx = 0.85*imgx
    apexy = 0.60*imgy     # Height of the apex y coordinate
    d = 100           # Distance between the two apex points making rectangle
    # Step 4: Define the area of interest for lane detection
    quad_vertices = np.array([[(left_bottomx,imgy),(imgx/2-d/2+10,apexy),(imgx/2+d/2+10,apexy),
                          (right_bottomx,imgy)]],dtype=np.int32)
    
    #img = region_of_interest(img,quad_vertices)
    
    ##### End Masking Addition
    
    # Process image and generate binary pixel of interests
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img,orient='x',thresh=(20,100)) 
    grady = abs_sobel_thresh(img,orient='y',thresh=(25,255)) 
    #c_binary = color_thresh1(img, sthresh=(100,255), vthresh=(50,255))
    c_binary = color_thresh2(img, sthresh=(100,255), vthresh=(50,255), lthresh=(50,255))
    m_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0,25))
    d_binary = dir_threshold(img, sobel_kernel=15, thresh=(1,1.5))
    #preprocessImage[((gradx == 1) & (grady ==1) | (c_binary==1))] = 255
    
    preprocessImage[((gradx == 1) & (grady == 1) | (m_binary == 1) & (d_binary == 1) & (c_binary == 1))] = 255

    bin_image = preprocessImage
    
    # Definition for points to use for perspective transform
    img_size = (img.shape[1],img.shape[0])
    #print (img_size)
    
    # Setting source and destination for 4 corners of the trapezoid
    '''   
    bot_width = 0.76
    mid_width = 0.08
    height_pct = 0.625
    bottom_trim = 0.935
    src = np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],
        [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim], [img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
    offset = img_size[0]*0.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    '''
    
    
    # Trying updated Source and Destination Coordinates (based on feedback from the reviewer)
   src = np.float32([[585, 450], [203, 720], [1127, 720], [685, 450]])
   dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])


    # Setting source and destination for 4 corners of the trapezoid
    #src = np.float32([[220, 700], [1100, 700], [690, 450], [590, 450]])
    #dst = np.float32([[300, 720], [980, 720], [980, 0], [300, 0]])
   
    #src = np.float32([[230, 700], [1100, 700], [680, 450], [600, 450]])
    #dst = np.float32([[310, 710], [960, 710], [960, 10], [310, 10]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    # Mask warped image

    window_width =  25  #60
    window_height = 80  #80 based on Udacity class 
    # Use Offset for window sliding
    Offset = 25
    
    ## Set up the overall class to do all the tracking
    curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = Offset, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15)
    
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
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    #right_fit = np.polyfit(res_yvals, rightx, 3)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    #right_fitx = right_fit[0]*yvals*yvals*yvals + right_fit[1]*yvals*yvals + right_fit[2]*yvals + right_fit[3]
    right_fitx = np.array(right_fitx,np.int32)
   
    
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

    ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dim
    xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dim

    # Track left lane curvature as we are using leftx
    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
    
    # Calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # draw the text showingn curvature, offset and speed
    cv2.putText(result, 'Radius of Curvature = ' +str(round(curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3))) +'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    # Final combination that works best includes
    # color_thresh1 based on s and v channel and preprocessing which includes gradx, grady and c_binary
    
    #result = bin_image
    #result = road
    #result = green_boxes
    #result = warped
    
    return result

#Input_video = 'harder_challenge_video.mp4'
#Input_video = 'output1_tracked.mp4'
#Input_video = 'project_video.mp4'
Input_video = 'challenge_video.mp4'

#Output_video = 'output1_tracked.mp4'
Output_video = 'output_challenge.mp4'
#Output_video = 'output_harder_challenge.mp4'


clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image)   # Function expects color images
video_clip.write_videofile(Output_video, audio=False)


