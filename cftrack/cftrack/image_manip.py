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
# Functions for image value manipulation and equalization.
#################################################################

from skimage import exposure
import numpy as np


def to_uint8(data, stretch=True, mx=None, mn=None):
    """Scales array to unsigned int values between 0 and 255.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data array to map to int and scale.  Must be an integer dtype
        if stretch=False.
    stretch : bool, default=True
        Default is True.  
        Whether or not to 'stretch' the scale - to apply a linear
        map such that the minimum value in data is mapped to 0 
        and the maximum value in data is mapped to 255.  If stretch 
        is False, then data will be scaled by dividing by the 
        maximum possible value for that given data type (e.g. if 
        data is type int16 and stretch=False, then data will be 
        divided by 32767 and then scaled between 0 and 255).
    mx : float, optional
        The maximum value to clip the data to when linearly 
        rescaling.  This value will map to 255 after the linear 
        rescaling and anything above it will clipped to 255.
        Only applies when `stretch` is True.
    mn : float, optional
        The minimum value to clip the data to when linearly
        rescaling.  This value will map to 0 after the linear
        rescaling and anything below it will be clipped to 255.
        Only applies when `stretch` is True.
    
    Returns
    -------
    numpy.ndarray
        The scaled and mapped uint8 data.
    """
    if stretch:
        if mx is None:
            mx = data.max()
        if mn is None:
            mn = data.min()
        data = 255*(data.astype(np.float) - mn)/(mx - mn)
    else:
        if np.issubdtype(data.dtype, np.integer):
            data = 255*data.astype(np.float)/np.iinfo(data.dtype).max
        
    return data.astype(np.uint8)


def equalize_data(data):
    """Performs histogram equalization on an inputted image data
    array.  The image values are clipped to the 5th and 97th 
    percentils.  They are then re-binned to force a linear
    effective image CDF, thereby usually improving overall image 
    contrast.  This function uses the scikit-image implementation
    of exposure.equalize_hist() to perform equalization.
    
    Parameters
    ----------
    data : numpy.ndarray
        The input image.
    
    Returns
    -------
    numpy.ndarray
        The histogram-equalized image.
    """
    
    newdata = exposure.rescale_intensity(data.astype(np.float), 
                                         in_range=(np.percentile(data, 5), np.percentile(data, 97)),
                                         out_range=(-1, 1))
    newdata = exposure.equalize_hist(newdata)
    
    return newdata
