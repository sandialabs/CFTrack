#################################################################
# Copyright 2022 National Technology & Engineering Solutions of 
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 
# with NTESS, the U.S. Government retains certain rights in this 
# software.
#
# Sandia National Labs
# Date: 2021-08-31
# Authors: Kelsie Larson, Don Lyons, Skyler Gray
# Contact: Kelsie Larson, kmlarso@sandia.gov
#
# Functions for reading various information from NASA GOES-R ABI
# imagery and metadata
#################################################################
import platform
import glob
import os
import sys
import h5py
import datetime
import numpy as np
from pyproj import Proj

from .image_manip import to_uint8, equalize_data


def get_nc_files(custom_path: str,
                 band: str = 'C06',
                 year: int = 2019,
                 day: int = 169,
                 hour: int = None,
                 minute: int = None):
    """Get the list of nc file paths for a given band, year, day, hour,
    and minute.
    
    This function assumes a specific file structure beneath the
    GOES data directory.  It expects:
    .../yyyy/ddd/bnd/ where yyyy is the 4-digit year, ddd is the
    3-digit day of the year, and bnd is the 3-character GOES band.
    
    This function also assumes the following file naming convention, typical
    of NOAA GOES imagery:
    *yyyydddhhmm*.nc where * is a wildcard, yyyy is the 4-digit year, ddd is
    the 3-digit day, hh is the 2-digit hour in 24-hr format, and mm is the
    2-digit minute.
    
    Parameters
    ----------
    custom_path : str
        The path to the GOES_image_dir data directory file structure.
    band : str, default='C06'
        The GOES band.
    year : int, default=2019
        The year the data was taken.
    day : int, default=169
        The day the data was taken.
    hour : int, optional
        The hour the data was taken.  If None, will find the data for every
        hour in the day.
    minute : int, optional
        The minute the data was taken.  If None, will find the data for every
        minute.
    
    Returns
    -------
    list of str
        Alphabetically sorted list of nc file paths.
    """
    filestr = custom_path
    
    # Add the rest of the path
    filestr = os.path.join(filestr,
                           '{0:04d}'.format(year),
                           '{0:03d}'.format(day),
                           '{}'.format(band),
                           '*' + '{0:04d}'.format(year) + '{0:03d}'.format(day))
    
    # Filter by hour if it is inputted
    if hour is not None:
        filestr += '{0:02d}'.format(hour)
        
        # Filter by minute if it is inputted
        if minute is not None:
            filestr += '{0:02d}'.format(minute)
    
    # Finish the glob search string
    filestr += '*.nc'
    
    # Get all the possible files
    possible_files = glob.glob(filestr)
    possible_files.sort()
    return(possible_files)


def get_matching_files_from_other_band(input_files: list,
                                       new_band: str,
                                       custom_path: str):
    """Get the files from new_band that match the timestamps from input_files.
    Also remove the input_files which do not have a match in the new_band.
    
    Parameters
    ----------
    input_files : list of str
        The list of absolute filenames from a single band
    new_band : str
        The string name of the band you want to get matching files for
    custom_path : str
        The path to the GOES_image_dir directory file structure
    
    Returns
    -------
    list of str
        The updated list of input_files
    list of str
        The matched list of files from new_band
    """
    in_files = input_files
    new_files = []
    out_files_saved = []
    
    for f in in_files:
        search_str = os.path.basename(f).split('_')[3]
        year = int(search_str[1:5])
        day = int(search_str[5:8])
        hr = int(search_str[8:10])
        minute = int(search_str[10:12])
        tmp_files = get_nc_files(custom_path,
                                 new_band, year, day,
                                 hour=hr, minute=minute)
        
        if len(tmp_files) == 0:
            print ("WARNING: file {} does not have a match in {}.".format(
                search_str, new_band))
            del(f)
        
        else:
            new_files.append(tmp_files[0])
            out_files_saved.append(f)

    return (out_files_saved, new_files)


def get_timestamp_from_nc(filepath: str):
    """Get the timestamp from the nc file.
    
    Parameters
    ----------
    filepath : str
        Path to nc file.
    
    Returns
    -------
    datetime.datetime object
    """
    # The "epoch" according to GOES product guide is J2K
    startdate = datetime.datetime(2000, 1, 1, 12)

    # Read the relevant information from the file
    with h5py.File(filepath, 'r') as fid:
        seconds_since_epoch = fid['t'][()]

    # Convert timestamp to datetime format
    tmpdelta = datetime.timedelta(seconds=seconds_since_epoch)
    timestamp = startdate+tmpdelta

    return timestamp


def get_radiance_from_nc(filepath: str):
    """Get radiance data from nc file.
    
    This function will clip the data to the given range and
    apply the scale factor and offset as described in the
    GOES-R L1 product user guide.
    
    See https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf
    
    Parameters
    ----------
    filepath : str
        The path to the nc file.
    
    Returns
    -------
    numpy.ndarray object
        The radiance data.
    datetime.datetime object
        The timestamp from the nc file.
    float
        The mean radiance (of all valid pixels); this is the pre-computed
        value taken directly from nc file metadata
    
    Notes
    -----
    NC files are netcdf files, which are a specific format of
    HDF5.  The raw data is usually stored as 16-bit uint, but is
    sometimes stored as 16-bit int instead.  If unsigned_flag is
    true, then the data is 16-bit int and must be cast to 16-bit
    uint.  The data then must be converted to float values by
    multiplying by the scale_factor and adding the add_offset. To
    check your conversion, the file stores max, min, and mean
    radiance values in max_radiance_value_of_valid_pixels,
    min_radiance_value_of_valid_pixels, and
    mean_radiance_value_of_valid_pixels.
    
    The DQF, or data quality flag, has values between -1 and 4 to 
    denote how believable a certain pixel value is.  The DQF should
    be 0 for every pixel.
    
    The timestamp is stored in t as seconds from the epoch, where
    the epoch is 12 noon on 01-01-2000.
    """

    scaled_data, timestamp, mean_radiance = \
        read_and_scale_data(filepath,
                            key='Rad',
                            mean_key='mean_radiance_value_of_valid_pixels')
    
    return scaled_data, timestamp, mean_radiance


def get_cmi_from_nc(filepath: str):
    """Get CMI data from nc file.
    
    This function will clip the data to the given range and
    apply the scale factor and offset as described in the
    GOES-R L1 product user guide.
    
    See https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf
    
    Parameters
    ----------
    filepath : str
        The path to the nc file.
    
    Returns
    -------
    numpy.ndarray object
        The CMI data.
    datetime.datetime object
        The timestamp from the nc file.
    float
        The mean CMI (of all valid pixels); this is the pre-computed
        value taken directly from nc file metadata
    """

    if ('M6C01' in filepath or
        'M6C02' in filepath or
        'M6C03' in filepath or
        'M6C04' in filepath or
        'M6C05' in filepath or
        'M6C06' in filepath):
        mean_key = 'mean_reflectance_factor'
    else:
        mean_key = 'mean_brightness_temperature'
        
    scaled_data, timestamp, mean_cmi = \
        read_and_scale_data(filepath,
                            key='CMI',
                            mean_key=mean_key)
    
    return scaled_data, timestamp, mean_cmi


def read_and_scale_data(filepath: str, key: str, mean_key: str = None):
    """Read the data from a GOES-ABI nc file and apply the
    appropriate scale factor and offset.  Also get the mean
    value and the timestamp.
    
    Parameters
    ----------
    filepath : str
        The path to the nc file.
    key : str
        The key to access the raw data in the nc file format.
        For example, if you're reading radiance data, the key
        would be "Rad".
    mean_key : str, optional
        The key to access the mean value stored in the metadata.
        For example, if you're reading radiance data, the
        mean_key would be "mean_radiance_value_of_valid_pixels".
    
    Returns
    -------
    numpy.ndarray object
        The data.
    datetime.datetime object
        The timestamp from the nc file.
    float (only returned if mean_key is not None)
        The mean value (of all valid pixels); this is the pre-computed
        value taken directly from nc file metadata
    """
    
    # The "epoch" according to GOES product guide is J2K
    startdate = datetime.datetime(2000, 1, 1, 12)

    # Read the relevant information from the file
    with h5py.File(filepath, 'r') as fid:
        raw_data = fid[key][()]
        raw_range = fid[key].attrs['valid_range']
        scale_factor = fid[key].attrs['scale_factor']
        add_offset = fid[key].attrs['add_offset']
        unsigned_flag = fid[key].attrs['_Unsigned'].decode('utf-8')
        timestamp = fid['t'][()]

        if mean_key is not None:
            mean_val = fid[mean_key][()]
        
    # Convert timestamp to datetime format
    tmpdelta = datetime.timedelta(seconds=timestamp)
    timestamp = startdate+tmpdelta
    
    # Cast data if required
    if unsigned_flag == 'true':
        raw_data = raw_data.astype(np.uint16)
        raw_range = raw_range.astype(np.uint16)
    
    # Clip the raw data to the min and max
    raw_data[raw_data < raw_range[0]] = raw_range[0]
    raw_data[raw_data > raw_range[1]] = raw_range[1]
    
    # Apply the scale factor and offset
    scaled_data = raw_data*scale_factor + add_offset

    if mean_key is None:
        return scaled_data, timestamp
    
    return scaled_data, timestamp, mean_val


def check_dqf(filepath: str, percent_bad: float = 2.0):
    """Check that the percentage of non-zero values in the DQF
    is less than the requested number.
    
    DQF is the data quality flag.  Each pixel in the nc image is 
    assigned an integer "data quality" value, with 0 as "perfect
    quality" and other integer values holding specific meanings.
    
    The default fill value is 255.  255 appears in the DQF if
    the DQF is not set.  Other values are as follows:
      - 0: good pixel
      - 1: conditionally usable pixel (see the L1b PUG)
      - 2: out of range pixel
      - 3: no value pixel
    
    If the percentage of non-zero DQF values in the image is
    less than the inputted percent_bad, then the image
    passes the DQF check.  If the percentage of non-zero DQF
    values in the image is greater the inputted
    percent_bad or if there is an error reading the DQF,
    then the image fails the DQF check.
    
    Parameters
    ----------
    filepath : str
        The path to the nc file
    percent_bad : float
        The maximum allowed percentage of non-zero DQF values in
        the image
    
    Returns
    -------
    bool
        True if the file passes the DQF check.  False if the
        file fails the DQF check or if there is a read error
        on the file.
    """
    with h5py.File(filepath, 'r') as fid:
        try:
            dqf = fid['DQF'][()]
            dqf = dqf.astype(np.uint8)
        except OSError:
            print ("WARNING: Read error on {}".format(filepath))
            return False
    
    total_pixels = dqf.shape[0]*dqf.shape[1]
    num_bad = (dqf != 0).sum()
    if num_bad*100.0/total_pixels > percent_bad:
        return False
    
    return True


def get_projection_from_nc(filepath: str):
    """Get the projection info from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the nc file.
    
    Returns
    -------
    p : pyproj.Proj
        The projection object that goes from lon/lat in deg to x/y map
        projection coordinates in meters
    h : float
        The satellite position above the earth (ellipsoid)
    (sfx, sfy) : (float, float)
        The x and y scale factors required to convert from px to meters
    (osx, osy) : (float, float)
        The x and y offsets required to convert from px to meters
    """
    # Get the relevant info from the h5 header
    with h5py.File(filepath, 'r') as fid:
        h = fid['goes_imager_projection'].attrs['perspective_point_height'][0]
        lon_0 = fid['goes_imager_projection'].attrs['longitude_of_projection_origin'][0]
        sweep = fid['goes_imager_projection'].attrs['sweep_angle_axis'].decode('utf-8')
        
        sfx = fid['x'].attrs['scale_factor']
        osx = fid['x'].attrs['add_offset']
        
        sfy = fid['y'].attrs['scale_factor']
        osy = fid['y'].attrs['add_offset']
        
    # Set up the pyproj.Proj projection object
    p = Proj(proj='geos', h=h, lon_0=lon_0, sweep=sweep)
    return (p, h, (sfx, sfy), (osx, osy))


def lonlat_to_xy(lon, lat, proj, sat_height, scale_factor, offset):
    """Convert (lon, lat) to (x, y) pixel coordinates
    
    Parameters
    ----------
    lon : float or list of float
        Longitude or list of longitudes in degrees
    lat : float or list of float
        Corresponding latitude or list of latitudes in degrees
    proj : pyproj.Proj object
        Projection object (outputted by :func:`get_projection_from_nc`)
    sat_height : float
        Height of satellite above earth ellipsoid (outputted by
        :func:`get_projection_from_nc`)
    scale_factor : tuple of float (x, y)
        The x and y scale factors for converting pixel coordinates
        to meters (outputted by :func:`get_projection_from_nc`)
    offset : float
        The x and y offsets for converting pixel coordinates to
        meters (outputted by :func:`get_projection_from_nc`)
    
    Returns
    -------
    numpy.ndarray, dtype=int
        2D array of (x, y) pixel coordinates.  If the inputted
        lon and lat were single values, the 2D array will have
        just one row.  If the inputted lon and lat were lists,
        then the 2D array will be the same number of rows as the
        inputted lists.
    """

    # Get the x and y coordinates in meters and relative to the
    # satellite height.
    xtmp, ytmp = proj(lon, lat)

    # To get from meters to pixel locations, we must first divide
    # by the satellite height
    # (see https://proj.org/operations/projections/geos.html)
    # then convert from meters in (x,y) to pixels in (x,y).
    # This means subtract the "add_offset" and divide by the
    # "scale_factor"
    # (see https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf)
    x = (xtmp/sat_height - offset[0])/scale_factor[0]
    y = (ytmp/sat_height - offset[1])/scale_factor[1]
    
    return (np.array(list(zip(x, y))).astype(np.int))


def xy_to_lonlat(x, y, proj, sat_height, scale_factor, offset):
    """Convert (lon, lat) to (x, y) pixel coordinates
    
    Parameters
    ----------
    x : float or list of float
        X pixel location or list of x pixel locations
    y : float or list of float
        Y pixel location or list of y pixel locations
    proj : pyproj.Proj object
        Projection object (outputted by get_projection_from_nc)
    sat_height : float
        Height of satellite above earth ellipsoid (outputted by
        get_projection_from_nc)
    scale_factor : tuple of float (x, y)
        The x and y scale factors for converting pixel coordinates
        to meters (outputted by get_projection_from_nc)
    offset : float
        The x and y offsets for converting pixel coordinates to
        meters (outputted by get_projection_from_nc)
    
    Returns
    -------
    numpy.ndarray, dtype=float
        2D array of (lon, lat) coordinates in degrees.  If the 
        inputted x and y were single values, the 2D array will have
        just one row.  If the inputted lon and lat were lists,
        then the 2D array will be the same number of rows as the
        inputted lists.
    """
    # Convert (x, y) to meters relative to the satellite height.
    # According to GOES L1B PUG, multiply by scale_factor and add
    # the add_offset
    # (see https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf)
    # Then multiply by satellite height
    # (see https://proj.org/operations/projections/geos.html)
    xtmp = (x*scale_factor[0] + offset[0])*sat_height
    ytmp = (y*scale_factor[1] + offset[1])*sat_height

    # Convert to lon, lat in degrees
    lon, lat = proj(xtmp, ytmp, inverse=True)
    
    return(np.array(list(zip(lon, lat))))

def get_all_radiance_data(custom_path: str,
                          start_year: int = 2019,
                          start_day: int = 168,
                          start_hour: int = 0,
                          start_minute: int = 0,
                          end_year: int = None,
                          end_day: int = None,
                          end_hour: int = 23,
                          end_minute: int = 59):
    """Get the radiance data, timestamp, and mean radiance values from the
    GOES nc files and return as lists for the specified time interval.
    
    Parameters
    ----------
    custom_path : str
        The path to the GOES_data_dir directory structure.
    start_year : int, default=2019
        The year of the datetime which starts the time interval.
    start_day : int, default=168
        The day of the datetime which starts the time interval in 
        day-of-year format, e.g. Feb 20 would be day 51.
    start_hour : int, default=0
        The hour of the datetime which starts the time interval in 
        24-hour format, e.g. 4pm would be 16.  Must be between 0 
        and 23.
    start_minute : int, default=0
        The start minute of the time interval.  Must be between 0
        and 59.
    end_year : int, default=`start_year`
        The year of the datetime which ends the time interval.
        The end datetime is included in the time interval.
    end_day : int, default=`start_day`
        The day of the datetime which ends the time interval in
        day-of-year format, e.g. Feb 20 would be day 51.  The
        end datetime is included in the time interval.
    end_hour : int, default=23
        The hour of the datetime which ends the time interval in
        24-hour format, e.g. 4pm would be 16.  Must be between 0
        and 23.  The end datetime is included in the time 
        interval.
    end_minute : int, default=59
        The end minute of the time interval.  Must be between 0
        and 59.
    
    Returns
    -------
    list of numpy.ndarray of uint8 format
        list of arrays of combined radiance data; one array per file
    list of str
        List of image timestamps in format %Y-%m-%d %H:%M:%S
    list of float
        List of the mean radiance value for band C06
    """
    
    # Set up the default inputs
    if end_year is None:
        end_year = start_year
    if end_day is None:
        end_day = start_day
    
    print ("")
    print ("Deciding which C06 files to grab...")
    print ("Start: {}, {}, {:02d}:{:02d}".format(start_year, start_day, start_hour, start_minute))
    print ("End: {}, {}, {:02d}:{:02d}".format(end_year, end_day, end_hour, end_minute))
    
    if end_year < start_year:
        sys.exit("ERROR: End time must come after start time.")
        
    elif end_year == start_year:
        if end_day < start_day:
            sys.exit("ERROR: End time must come after start time.")
        elif end_day == start_day:
            if end_hour < start_hour:
                sys.exit("ERROR: End time must come after start time.")
            elif end_hour == start_hour:
                if end_minute < start_minute:
                    sys.exit("ERROR: End time must come after start time.")
    
    # Loop through the years, days, hours, minutes, etc
    what_to_read = dict()
    
    for year in range(start_year, end_year+1):
        what_to_read[year] = dict()
        
    # Collect all of the filenames for every day in the input
    if start_year != end_year:
        for day in range(start_day, 366):
            what_to_read[start_year][day] = get_nc_files(custom_path,
                                                         'C06', year, day)
        for day in range(1, end_day):
            what_to_read[end_year][day] = get_nc_files(custom_path,
                                                       'C06', year, day)
        for year in range(start_year+1, end_year):
            for day in range(1, 366):
                what_to_read[year][day] = get_nc_files(custom_path,
                                                       'C06', year, day)
    
    else:
        for day in range(start_day, end_day+1):
            what_to_read[start_year][day] = get_nc_files(custom_path,
                                                         'C06', start_year, day)
    
    # On the first day, if there is a start hour, remove every filename before
    # that hour
    first_ind = 0
    for i in range(len(what_to_read[start_year][start_day])):
        basename = os.path.basename(what_to_read[start_year][start_day][i])
        hr = int(basename.split('_')[-3][8:10])
        if hr >= start_hour:
            first_ind = i
            break
    
    # Uncomment for debug
    #print ("Deleting stuff before {} on day {}".format(hr, start_day))
    #print ("First index to keep: {}".format(first_ind))
    del(what_to_read[start_year][start_day][0:first_ind])
    
    # On the last day, if there is an end hour, remove every filename after
    # that hour
    last_ind = len(what_to_read[end_year][end_day])-1
    for i in range(len(what_to_read[end_year][end_day])-1, -1, -1):
        basename = os.path.basename(what_to_read[end_year][end_day][i])
        hr = int(basename.split('_')[-3][8:10])
        if hr <= end_hour:
            last_ind = i
            break
    
    # Uncomment for debug
    #print ("Deleting stuff after {} on day {}".format(hr, end_day))
    #print ("Last index to keep: {}".format(last_ind))
    del(what_to_read[end_year][end_day][last_ind+1:])
    
    # How many samples are in the first hour?
    first_ind = len(what_to_read[start_year][start_day])
    for i in range(len(what_to_read[start_year][start_day])):
        basename = os.path.basename(what_to_read[start_year][start_day][i])
        hr = int(basename.split('_')[-3][8:10])
        if hr > start_hour:
            first_ind = i
            break
    
    # On the first day, if there is a start minute, remove anything before
    # that start minute in the first hour
    for i in range(first_ind):
        basename = os.path.basename(what_to_read[start_year][start_day][i])
        mn = int(basename.split('_')[-3][10:12])
        if mn >= start_minute:
            first_ind = i
            break
    
    # Uncomment for debug
    #print ("Deleting stuff before minute {}".format(mn))
    #print ("First index to keep: {}".format(first_ind))
    del(what_to_read[start_year][start_day][0:first_ind])
    
    
    # How many samples are in the last hour?
    last_ind = -1
    for i in range(len(what_to_read[end_year][end_day])-1, -1, -1):
        basename = os.path.basename(what_to_read[end_year][end_day][i])
        hr = int(basename.split('_')[-3][8:10])
        if hr < end_hour:
            last_ind = i
            break
    
    # On the last day, if there is a end minute, remove anything after that
    # end minute in the last hour
    for i in range(len(what_to_read[end_year][end_day])-1, last_ind, -1):
        basename = os.path.basename(what_to_read[end_year][end_day][i])
        mn = int(basename.split('_')[-3][10:12])
        if mn <= end_minute:
            last_ind = i
            break
    # Uncomment for debug
    #print ("Deleting stuff after minute {}".format(mn))
    #print ("Last index to keep: {}".format(last_ind))
    del(what_to_read[end_year][end_day][last_ind+1:])
    
    # Flatten the dictionary into a list of filenames
    files_c06 = []
    for yr in what_to_read:
        for dy in what_to_read[yr]:
            files_c06.extend(what_to_read[yr][dy])
    print ("")
    
    ####################################################
    # Get the matching C07 files and remove the C06 files that have
    # no match in C07
    print ("Getting matching files for C07...")
    files_c06, files_c07 = get_matching_files_from_other_band(files_c06, 'C07',
                                                              custom_path=custom_path)
    
    num_samples = len(files_c06)
    
    ####################################################
    # Read all the radiance data from the C06 and C07 files
    print ("")
    print ("Reading radiance data from files...")
    timestamps = []
    data_combined = []
    mean_rads = []
    
    k = 0
    for f6, f7 in zip(files_c06, files_c07):
        print ("Reading and combining bands for frame {} of {}".format(k+1, num_samples))
        
        if not check_dqf(f6):
            print ("WARNING: Data quality error {}. Skipping.".format(os.path.basename(f6)))
            k += 1
            continue
        
        if not check_dqf(f7):
            print ("WARNING: Data quality error {}. Skipping.".format(os.path.basename(f7)))
            k += 1
            continue
        
        data_c06, timestamp, mean_rad = get_radiance_from_nc(f6)
        timestamps.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        
        data_c07, _, _ = get_radiance_from_nc(f7)
        
        # Combine the data by:
        # 1. C06 - C07
        # 2. Equalize the histogram of the entire combined image
        #    (i.e. 'linearizing' the effective cdf)
        # 3. Convert to uint8 - the feature-selection algorithm uses
        #    only uint8 or float32 data
        tmp = data_c06-data_c07
        tmp = equalize_data(tmp)
        
        # The data must be uint8 for input to opencv LK
        data_combined.append(to_uint8(tmp, mx=1.0, mn=0.0))
        
        # Add the mean radiance to array
        mean_rads.append(mean_rad)
    
        # Increment our frame counter
        k += 1
    
    return data_combined, timestamps, mean_rads, files_c06, files_c07
    
