#################################################################
# Copyright 2022 National Technology & Engineering Solutions of 
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 
# with NTESS, the U.S. Government retains certain rights in this 
# software.
#
# Sandia National Labs
# Date: 2021-08-31
# Authors: Kelsie Larson, Skyler Gray
# Contact: Kelsie Larson, kmlarso@sandia.gov
#
# Contains functions for finding features, computing optical
# flow, and checking solar angle for sunrise/sunset.
#################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import sys

import datetime
from pvlib.solarposition import get_solarposition
from .goes_io import (xy_to_lonlat)


def is_sun_set_rise_transition(rect, timestamp, proj, sat_height, scale_factor, offset):
    """Check whether sunrise or sunset is occurring.
    This algorithm uses the pvlib module to calculate the apparent solar altitude
    angle of every pixel on the perimeter of the tracked image of interest. The
    sunrise transition has reached the tracked image when the maximum altitude
    angle of the right half of the image perimeter is above -6. The transition
    officially ends when the minimum of the left half of the image perimeter pixels
    is greater than 6. Sunset is defined similarly, where it starts when the min
    of the right half is less than 6 and ends when the max of the left half is 
    less than 6.
    
    Parameters
    ----------
    rect : np.ndarray
        A 2D array delineating with 5 (x, y) points that create a closed rectangle
    timestamp : string
        The timestamp of the image of interest
    proj : pyproj.Proj object
        Projection object (outputted by :func:`goes_io.get_projection_from_nc`)
    sat_height : float
        Height of satellite above earth ellipsoid (outputted by
        :func:`goes_io.get_projection_from_nc`)
    scale_factor : tuple of float (x, y)
        The x and y scale factors for converting pixel coordinates
        to meters (outputted by :func:`goes_io.get_projection_from_nc`)
    offset : float
        The x and y offsets for converting pixel coordinates to
        meters (outputted by :func:`goes_io.get_projection_from_nc`)
    
    Returns
    -------
    bool
        An indicator for whether this image is part of a sunrise or
        sunset transition.
    """
    # Extract x and y max and mins
    xmin, ymin = rect.min(axis = 0).astype(int).tolist()
    xmax, ymax = rect.max(axis = 0).astype(int).tolist()
    
    # Split box into roughly left and right halves (flooring width calculation)
    width_half = int( (xmax - xmin)/2 )
    
    left = [[xmin, y] for y in range(ymin, ymax+1)] + \
        [[x, ymin] for x in range(xmin+1, xmin + width_half + 1)] + \
        [[x, ymax] for x in range(xmin+1, xmin + width_half + 1)]
    pixels_left = np.array(left)
    
    right = [[xmax, y] for y in range(ymin, ymax+1)] + \
        [[x, ymin] for x in range(xmax - width_half, xmax)] + \
        [[x, ymax] for x in range(xmax - width_half, xmax)]
    pixels_right = np.array(right)
    
    # Convert pixel locations to lat-long coordinates
    lonlat_left = xy_to_lonlat(pixels_left[:, 0],
                               pixels_left[:, 1], 
                               proj, sat_height, scale_factor, offset)
    
    lonlat_right = xy_to_lonlat(pixels_right[:, 0],
                                pixels_right[:, 1], 
                                proj, sat_height, scale_factor, offset)
    
    # Pull time of day information from timestamp
    time_of_day = datetime.datetime.strptime(timestamp + ' +0000', 
                                             '%Y-%m-%d %H:%M:%S %z')
    
    # Calculate apparent solar altitude (in degrees)
    apparent_altitude_left = []
    apparent_altitude_right = []
    for i in range(lonlat_left.shape[0]):
        alt = 90 - get_solarposition(time_of_day, 
                                     lonlat_left[i, 1],
                                     lonlat_left[i, 0])['apparent_zenith'].item()
        apparent_altitude_left.append(alt)
        
        alt = 90 - get_solarposition(time_of_day, 
                                     lonlat_right[i, 1],
                                     lonlat_right[i, 0])['apparent_zenith'].item()
        apparent_altitude_right.append(alt)
    
    # Detect as sunrise/sunset transition period if solar angle is within
    #  specified range
    # Heuristically, low-level clouds appear to brighten at sunrise after -1 degrees
    #  They also appear to darken at sunset after decreasing past 5 degrees
    #  The sunset/sunrise transition ending degrees are just meant as bounds to
    #  catch the transition without falsely marking a clear day/night frame as
    #  a transition frame
    
    # Calculate max/min solar angles on both sides of box
    maxalt_left = max(apparent_altitude_left)
    maxalt_right = max(apparent_altitude_right)
    minalt_left = min(apparent_altitude_left)
    minalt_right = min(apparent_altitude_right)

    # UNCOMMENT FOR VERBOSE DIURNAL TRANSITION
    #print(f"Checking Sunrise characteristics - maxalt_right = {round(maxalt_right, 2)}, minalt_left = {round(minalt_left, 2)}")
    #print(f"Checking Sunset characteristics - minalt_right = {round(minalt_right, 2)}, maxalt_left = {round(maxalt_left, 2)} \n")
    
    transition = False
    # If experiencing sunrise or sunset, respectively, around box of interest...
    if (maxalt_right > -6 and minalt_left < 6) or \
       (minalt_right < 6 and maxalt_left > -6):
        transition = True        

    return transition


def find_features(data: np.ndarray,
                  mask: np.ndarray = None,
                  feature_params: dict = None):
    """Get starting "good" feature points.
    This algorithm uses the OpenCV implementation of Shi-Tomasi 
    corner detection.
    
    Parameters
    ----------
    data : numpy.ndarray
        The image in which you wish to find corner-like features
    mask : numpy.ndarray, optional
        A np.uint8 array filled with only 0s and 1s.  Will not 
        search for features in data where mask=0.
    feature_params : dict, optional
        Dictionary of key-value inputs for cv2.goodFeaturesToTrack.
        Parameters include:
            - maxCorners: maximum number of features to return.  
                Default = 100.
            - qualityLevel: minimum quality of a feature 
                (quality = minimum_eigenval * qualityLevel).  
                Default = 0.2.
            - minDistance: minimum distance between returned 
                features.  Default = 3.  
            - blockSize: neighborhood size for computing
                covariation matrix of derivatives.  Default = 3.
            - useHarrisDetector: bool for whether to use Harris
                or Shi-Tomasi detection. Default = False.
            - k: Free float parameter of the Harris detector.
                Default = 0.04.
            - See the OpenCV documentation at 
              https://docs.opencv.org/3.4.15/dd/d1a/group__imgproc__feature.html 
              for more information on the feature_params inputs.
    
    Returns
    -------
    dict
        This is a dictionary with keys k in [0, 1, ..., m],
        i.e. there is m+1 total good features detected.  Each 
        dict[k] value is the [x,y] location of the feature in a 
        size-3 array i.e. [x, nil, y]
    """
    
    # Set up default parameters.
    # cv2.goodFeaturesToTrack uses Shi-Tomasi corner detection.
    # maxCorners is the maximum number of corners to return.
    # qualityLevel sets the minimum required 'quality' for a corner
    # to be returned (quality = minimum_eigenvalue * qualityLevel).
    # minDistance is the minimum distance between returned corners.
    # blockSize is the size of the neighborhood for computing 
    # covariation matrix of derivatives.
    f_params = dict(maxCorners = 100,
                    qualityLevel = 0.2,
                    minDistance = 3,
                    blockSize = 3)
    
    # Update the parameters with whatever the user inputted
    if feature_params is not None:
        f_params.update(feature_params)
        
    if mask is not None:
        feats = cv2.goodFeaturesToTrack(data,
                                        mask=mask,
                                        **f_params)
        
    else:
        feats = cv2.goodFeaturesToTrack(data,
                                        **f_params)

    outfeats = dict()
    for i in range(len(feats)):
        outfeats[i] = feats[i]
        
    return outfeats 


def compute_optical_flow(prev_frame: np.ndarray,
                         this_frame: np.ndarray,
                         prev_pts: dict,
                         lk_params: dict = None):
    """Compute the locations of the points in the current frame
    
    Parameters
    ----------
    prev_frame : numpy.ndarray
        The previous image (frame)
    this_frame : numpy.ndarray
        The current image (frame)
    prev_pts : dict
        This is a dictionary with keys k in [0, 1, ..., m], 
        where m is the number of good features detected.  Each dict[k] 
        value is the [x,y] location of the feature in the 
        previous frame in a size-3 array, i.e. [x, [], y]
    lk_params : dict
        This dictionary contains the key-value pairs for input
        parameters into the cv2.calcOpticalFlowPyrLK function.
        Parameters include:
            - winSize: tuple for window size.
                Default = (15, 15).
            - maxLevel: max depth of pyramids.  Default = 2.
            - criteria: tuple for stopping criteria.  Default
                EPS = 0.03 or max iterations = 10.
            - flags: operation flags
            - minEigThreshold: threshold for degraded features 
                during tracking.  Default = 1e-4.
            - See the documentation for calcOpticalFlowPyrLK() at
              https://docs.opencv.org/3.4.15/dc/d6b/group__video__track.html
              for more info on these.
    
    Returns
    -------
    dict
        This is a dictionary with keys k in [0, 1, ..., m], 
        where m is the number of good features detected.  Each dict[k] 
        value is the [x,y] location of the feature in the current
        frame in a size-3 array, i.e. [x, [], y]
    """
    
    # Set up the optical flow algorithm:
    # This is the iterative Lucas-Kanade algorithm with pyramids.
    # winSize is the size of the search window at each pyramid level.
    # maxLevel is the maximum 0-based pyramid level, e.g. if maxLevel=0 then
    # 1 level (level 0) is used; if maxLevel=1 then 2 levels (level 0 and
    # level 1) are used, etc.
    # criteria is the termination criteria for the iterative search algorithm;
    # in this case it is either 10 iterations or when the search window 
    # displacement (epsilon) is less than 0.03.
    if lk_params is None:
        lk_params = dict(winSize = (15, 15),
                         maxLevel = 2,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                     10, 0.03))

    ppts = np.array([v for v in prev_pts.values()])
    tmp_out, st, err = cv2.calcOpticalFlowPyrLK(prev_frame,
                                                this_frame,
                                                ppts,
                                                None,
                                                **lk_params)
    current_pts = dict()
    for i in range(len(prev_pts)):
        if st[i][0] == 1:
            current_pts[list(prev_pts.keys())[i]] = tmp_out[i]

    return current_pts
