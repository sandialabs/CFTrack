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
# Run optical flow on series of GOES-R ABI radiance images.
#################################################################

import argparse
import sys
import json
import os

# ------------------------------------------------------------ #
# IMPORTS
# ------------------------------------------------------------ #
import numpy as np
import datetime
import cv2

from .goes_io import (get_all_radiance_data, get_projection_from_nc,
                      xy_to_lonlat)
from .optical_flow import (find_features, compute_optical_flow,
                           is_sun_set_rise_transition)
from .plot_support import (plot_frame, plot_animation,
                           compute_LK_rigid_transform,
                           apply_eis)

# ------------------------------------------------------------ #
# SHOW START PLOTS
# ------------------------------------------------------------ #
def show_intermediate_plots(custom_path: str,
                            start_year: int,
                            start_day: int,
                            start_hour: int,
                            start_minute: int,
                            end_year: int,
                            end_day: int,
                            end_hour: int,
                            end_minute: int,
                            mask_ylo: int,
                            mask_yhi: int,
                            mask_xlo: int,
                            mask_xhi: int):
    """Plot the first frame in the given series with the bounding
    box and the discovered Shi-Tomasi features.
    
    This function will generate 3 matplotlib plots to show the
    user's choice in bounding box over the first available image
    in the selected time frame.  The first plot shows the full 
    image with the selected tracking box overlaid.  The second 
    plot shows a zoomed-in view of the area surrounding the 
    tracking box plus the Shi-Tomasi features.  The third plot
    shows a further zoomed-in view.
    
    Parameters
    ----------
    custom_path : str
        The path to the GOES_data_dir directory structure.
    start_year : int
        Start year for goes data series.  Should be four-digit
        number.
    start_day : int
        Start day for goes data series in day-of-year format.
    start_hour : int
        Start hour for goes data series in 24-hour format, e.g.
        23 for 11 pm GMT.
    start_minute : int
        Start minute for goes data series.  Must be between 0 and
        59, inclusive.
    end_year : int
        End year for goes data series.  Should be four-digit
        number.  The data collector is inclusive, e.g. if 
        end_year is 2019, then 2019 is included in the data 
        series.
    end_day : int
        End day for goes data series in day-of-year format.  The
        data collector is inclusive, e.g. if end_day is 139, then
        day 139 is included in the data series.
    end_hour : int
        End hour for goes data series in 24-hour format.  The 
        data collector is inclusive, e.g. if end_hour is 23, then
        hour 23 data is included in the data series.
    end_minute : int
        End minute for goes data series.  Must be between 0 and 
        59.  Data collector is inclusive, e.g. if end_minute is
        55, then minute 55 data (if available) is included in the
        data series.
    mask_ylo : int
        Tracking box mask lower y boundary.  Note that 'lower' 
        refers to 'closer to zero,' which in Python is typically
        toward the top of the image.
    mask_yhi : int
        Tracking box mask upper y boundary.  Note that 'upper' 
        refers to 'farther from zero,' which in Python is 
        typically toward the bottom of the image.
    mask_xlo : int
        Tracking box mask lower x boundary.
    mask_ylo : int
        Tracking box mask upper x boundary.
    
    Returns
    -------
    None
    """
    
    def is_leapyear(year):
        if year % 4 == 0:
            if year % 100 == 0:
                if year % 400 == 0:
                    leapyear = True
                else:
                    leapyear = False
            else:
                leapyear = True
        else:
            leapyear = False
        return leapyear
    
    def time_incrementer(minute, hour, day, year):
        # How many days in the year?
        if is_leapyear(year): max_days = 366
        else: max_days = 365
        
        # if minute exceeds 59, increment hour
        if minute / 60 >= 1:
            minute = minute % 60
            hour = hour + 1
        # if hour exceeds 23, increment day
        if hour / 24 == 1:
            hour = 0
            day = day + 1
        # if day exceeds # in year, increment year
        if day / max_days > 1:
            day = 1
            year = year + 1
        
        return (minute, hour, day, year)
    
    i = 0
    data_combined = []
    while len(data_combined) == 0:
        print(f'Call {i+1} to image loader...')
        s_minute, s_hour, s_day, s_year = \
            time_incrementer(start_minute + i * 5,
                             start_hour,
                             start_day,
                             start_year)
        e_minute, e_hour, e_day, e_year = \
            time_incrementer(start_minute + (i+1) * 5,
                             start_hour,
                             start_day,
                             start_year)
        data_combined, timestamps, _, _, _ = \
            get_all_radiance_data(custom_path = custom_path,
                                  start_year = s_year,
                                  start_day = s_day,
                                  start_hour = s_hour,
                                  start_minute = s_minute,
                                  end_year = e_year,
                                  end_day = e_day,
                                  end_hour = e_hour,
                                  end_minute = e_minute)
    
    tmpmask = [np.array(((mask_xlo, mask_ylo),
                         (mask_xhi, mask_ylo),
                         (mask_xhi, mask_yhi),
                         (mask_xlo, mask_yhi),
                         (mask_xlo, mask_ylo)))]

    # Plot the first frame with the mask over top
    plot_frame(data=data_combined[0], 
               rects=tmpmask,
               title = timestamps[0])

    # Create the mask for the Shi-Tomasi
    mask = np.zeros(data_combined[0].shape, dtype=np.uint8)
    mask[mask_ylo:mask_yhi, mask_xlo:mask_xhi] += 1
    
    # Find starting Shi-Tomasi features using the mask
    p_out = dict()
    p_out[0] = find_features(data_combined[0],
                             mask=mask)
    
    # Plot the first frame, zoomed in to an area immediately
    # surrounding the mask
    ts = timestamps[0].replace(' ', '_').replace(':', '')
    plot_frame(data=data_combined[0], 
               features = p_out[0],
               rects = tmpmask,
               data_bounds = (mask_ylo-100, mask_yhi+100,
                              mask_xlo-100, mask_xhi+100),
               title = timestamps[0],
               feature_size = 3,
               size_inches = (5, 5))
    
    # Plot the first frame, zoomed in further to show a detailed
    # view of the features
    plot_frame(data=data_combined[0], 
               features = p_out[0],
               data_bounds = (mask_ylo, mask_yhi,
                              mask_xlo, mask_xhi),
               title = timestamps[0],
               feature_size = 10,
               size_inches = (3, 3))


# ------------------------------------------------------------ #
# CODE FOR FULL RUN
# ------------------------------------------------------------ #
# If the user chose to continue with the full run, then start
# the full run!
def run_optical_flow(custom_path: str,
                     start_year: int,
                     start_day: int,
                     start_hour: int,
                     start_minute: int,
                     end_year: int,
                     end_day: int,
                     end_hour: int,
                     end_minute: int,
                     mask_ylo: int,
                     mask_yhi: int,
                     mask_xlo: int,
                     mask_xhi: int,
                     crop_bounds: list,
                     output_loc: str):
    """
    Main code for running optical flow on a series of GOES-R ABI
    radiance products.
    
    This function depends on the GOES-R ABI Band C06 and C07
    radiance products.  All radiance products used must be 
    downloaded and stored in a specific file structure for
    automated reading based on inputted start/stop dates/times.
    Currently, this function points to the Cloud-Aerosols share
    drive, where GOES-R data is downloaded and stored as-needed.
    
    This will output a series of files to the specified output
    location.  The file series includes a .json file, which 
    contains tracking box corner coordinates (both pixel space
    and in lon/lat), the feature locations (in both pixel space
    and in lon/lat), and eis information for each pair of C06 and
    C07 GOES-R ABI radiance products.  It will also output a .mp4
    video file and a collection of .png image files.  The .mp4
    shows the video result of optical flow tracking, while the 
    .png images (one png is generated per hour of tracking) are
    still frames of the full-scene video as well as the zoomed-in
    analysis focused on the tracking box.
    
    Parameters
    ----------
    custom_path : str
        The path to the GOES/G17/ directory structure.  We assume
        a specific directory structure for where the GOES data is
        stored.
        
        It expects:
        .../GOES/G17/yyyy/ddd/bnd/ where yyyy is the 4-digit year, ddd is the
        3-digit day of the year, and bnd is the 3-character GOES band.
        
        This function also assumes the following file naming convention, typical
        of NOAA GOES imagery:
        *yyyydddhhmm*.nc where * is a wildcard, yyyy is the 4-digit year, ddd is
        the 3-digit day, hh is the 2-digit hour in 24-hr format, and mm is the
        2-digit minute.
    start_year : int
        Start year for goes data series.  Should be four-digit
        number.
    start_day : int
        Start day for goes data series in day-of-year format.
    start_hour : int
        Start hour for goes data series in 24-hour format, e.g.
        23 for 11 pm GMT.
    start_minute : int
        Start minute for goes data series.  Must be between 0 and
        59, inclusive.
    end_year : int
        End year for goes data series.  Should be four-digit
        number.  The data collector is inclusive, e.g. if 
        end_year is 2019, then 2019 is included in the data 
        series.
    end_day : int
        End day for goes data series in day-of-year format.  The
        data collector is inclusive, e.g. if end_day is 139, then
        day 139 is included in the data series.
    end_hour : int
        End hour for goes data series in 24-hour format.  The 
        data collector is inclusive, e.g. if end_hour is 23, then
        hour 23 data is included in the data series.
    end_minute : int
        End minute for goes data series.  Must be between 0 and 
        59.  Data collector is inclusive, e.g. if end_minute is
        55, then minute 55 data (if available) is included in the
        data series.
    mask_ylo : int
        Tracking box mask lower y boundary.  Note that 'lower' 
        refers to 'closer to zero,' which in Python is typically
        toward the top of the image.
    mask_yhi : int
        Tracking box mask upper y boundary.  Note that 'upper' 
        refers to 'farther from zero,' which in Python is 
        typically toward the bottom of the image.
    mask_xlo : int
        Tracking box mask lower x boundary.
    mask_xhi : int
        Tracking box mask upper x boundary.
    crop_bounds : List or tuple
        Bounds at which the image data is cropped for 
        display. This list/tuple is length-4, with bounds given in 
        numpy order:
            (y-lo, y-hi, x-lo, x-hi)
    output_loc : str
        Output directory to store outputs.  `output_loc` does not
        need to already exist; it will be created.  Another 
        directory, named by the start and end timestamps for the
        optical flow series, will be created below `output_loc`.
        The optical flow results will be written there.
    
    Returns
    -------
    None
    """

    # Get the data by hour, day, year, etc.
    data_combined, timestamps, mean_rads, files_c06, files_c07 = \
        get_all_radiance_data(custom_path = custom_path,
                              start_year = start_year,
                              start_day = start_day,
                              start_hour = start_hour,
                              start_minute = start_minute,
                              end_year = end_year,
                              end_day = end_day,
                              end_hour = end_hour,
                              end_minute = end_minute)
    
    # Set sample count
    num_samples = len(data_combined)

    # If there are no samples read, error
    if num_samples == 0:
        raise NoFilesReadError("No nc files found.")
    
    # Create the timestamp string to be used in the save filename
    filename_timestamp = datetime.datetime. \
        strptime(timestamps[0], '%Y-%m-%d %H:%M:%S'). \
        strftime('%Y%m%d%H%M%S')
    filename_timestamp += '-'
    filename_timestamp += datetime.datetime. \
        strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S'). \
        strftime('%Y%m%d%H%M%S')
    
    # Get the data bounds from the file
    first_timestamp = datetime.datetime. \
        strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
    
    # Create mask over region of interest
    mask = np.zeros(data_combined[0].shape, dtype=np.uint8)
    mask[mask_ylo:mask_yhi, mask_xlo:mask_xhi] += 1
    
    # Compute starting features using the mask
    p_out = dict()
    p_out[0] = find_features(data_combined[0],
                             mask=mask)
    
    # Set up the rectangles dictionary
    # The rects are the four-corners bounding boxes, starting with
    # the initial user-inputted mask, which are added to the
    # dictionary as the features are tracked across the frames.
    # There is one rect per frame.
    rects = dict()
    rects[0] = np.array(((mask_xlo, mask_ylo),
                         (mask_xhi, mask_ylo),
                         (mask_xhi, mask_yhi),
                         (mask_xlo, mask_yhi),
                         (mask_xlo, mask_ylo)))
    
    # Get the initial mean feature location and the initial rectangle
    # width and height
    lp = np.array([v for v in p_out[0].values()])[:, 0, :]
    prev_mean = lp.mean(axis=0)
    loc_means = []
    loc_means.append(prev_mean)
    
    # Crop the first frame to the size of the "zoomed-in" image and set up the
    # cropped frames dictionary.  These cropped frames are the zoomed-in, eis-
    # smoothed portions of what's in the rectangles in each frame.
    cropped_frames = dict()
    mask_pad = 30
    cropped_frames[0] = data_combined[0][
        int(rects[0][0, 1])-mask_pad:int(rects[0][2, 1])+mask_pad,
        int(rects[0][0, 0])-mask_pad:int(rects[0][2, 0])+mask_pad]
    transforms = np.zeros((len(data_combined)-1, 2))
    
    # Get projection info for calculating pixel solar angles at each frame
    proj, ht, sf, ofs = get_projection_from_nc(files_c06[0])
    
    # Added frame color
    frame_color = [[0.0, 0.0, 1.0]]
    # Added transition list of bools to know how to color animation square
    transition_frame = [0]

    
    # Uncomment to compute optical flow through the transitions instead of
    # "jumping over" them
    #transitions = []
    
    # Compute optical flow over frames before first transition
    print ("")
    print ("Computing flow...")
    end_frame = num_samples
    
    # =============================================================================
    
    i = 1
    while(i < end_frame):
        transition = is_sun_set_rise_transition(rects[i-1], timestamps[i-1],
                                                proj, ht, sf, ofs)
        if(transition == True):
            
            # Need at least 5 images to project velocity during transition
            if(i < 6):
                raise ValueError("Timeframe must start at least 5 frames before apparent sunrise/sunset.")
            
# =============================================================================
#             # Set the transition to take 24 frames (2 hours)
#             trans_end = i + 24
#             
#             # Make sure we don't pass the end of our frames
#             if trans_end > num_samples:
#                 trans_end = num_samples
# =============================================================================
            
            # Project motion over next n frames using the average velocity over
            # the last 5
            print ("")
            print ("Projecting motion forward...")
            vels = []
            accs = []
            for j in range(i-6, i-1):
                vels.append(loc_means[j+1] - loc_means[j])
            
            for j in range(len(vels)-1):
                accs.append(vels[j+1] - vels[j])
            
            start_vel = np.array(vels).mean(axis=0)
            start_loc = loc_means[i-1]
            mean_acc = np.array(accs).mean(axis=0)
            
            ######################################################
            # Project motion through the transition by assuming a
            # constant velocity
            proj_means = []
            while(transition == True):
                # Transition frame should be colored differently in animation
                transition_frame.append(1)
                frame_color.append([0.0, 1.0, 0.0])
                
                print ("Projecting motion for image {} of {}".format(i+1, num_samples))
                #new_mean = start_loc + start_vel*t + 0.5*mean_acc*(t**2)
                new_mean = loc_means[i-1] + start_vel
                loc_means.append(new_mean)
                #proj_means.append(new_mean)
                shift = new_mean-prev_mean
                rects[i] = rects[i-1] + shift
                prev_mean = new_mean
                
                cropped_frames[i] = data_combined[i][
                        int(rects[i][0, 1])-mask_pad:int(rects[i][2, 1])+mask_pad,
                        int(rects[i][0, 0])-mask_pad:int(rects[i][2, 0])+mask_pad]
    
                # Get transform matrix for eis
                transform_mat = compute_LK_rigid_transform(
                    cropped_frames[i-1],
                    cropped_frames[i])
    
                # We only care about translation, so ditch what we don't need
                transform_mat = transform_mat[0:2, 2]
                
                #print (transform_mat)
                transforms[i-1] = transform_mat
                
                i += 1
                
                # If we've come to the end of our frames, just break
                # out of the loop
                if i == num_samples:
                    break
                
                # Are we still in the middle of sunrise or sunset?
                transition = is_sun_set_rise_transition(rects[i-1], timestamps[i-1],
                                                        proj, ht, sf, ofs)
            
            ######################################################
            
            # If we've come to the end of our frames, just break out 
            # of the loop
            if i == num_samples:
                break
            
            # Create new mask over region of interest
            mask_xlo = int(np.round(rects[i-1][0][0]))
            mask_ylo = int(np.round(rects[i-1][0][1]))
            mask_xhi = int(np.round(rects[i-1][2][0]))
            mask_yhi = int(np.round(rects[i-1][2][1]))
            mask = np.zeros(data_combined[i].shape, dtype=np.uint8)
            mask[mask_ylo:mask_yhi, mask_xlo:mask_xhi] += 1
            
            # Compute starting new features using the mask
            p_out[i-1] = find_features(data_combined[i-1],
                                               mask=mask)
            
            lp = np.array([v for v in p_out[i-1].values()])[:, 0, :]
            prev_mean = lp.mean(axis=0)
            
            print('Sunlight/sunset period ended. Running optical flow...')
            
            continue
        
        # Frame will be colored normally in animation
        transition_frame.append(0)
        frame_color.append([0.0, 0.0, 1.0])
        
        print ("Computing flow for image {} of {}".format(i+1, num_samples))
        p_out[i] = compute_optical_flow(data_combined[i-1],
                                        data_combined[i],
                                        p_out[i-1])
        
        # Compute the new mean and std deviation of the LK features
        lp = np.array([v for v in p_out[i].values()])[:, 0, :]
        new_mean = lp.mean(axis=0)
        loc_means.append(new_mean)
        
        # Find and apply the new shift to the rectangle
        shift = new_mean-prev_mean
        rects[i] = rects[i-1] + shift
        prev_mean = new_mean
        
        # Crop the frame to the size of the rectangle plus buffer
        cropped_frames[i] = data_combined[i][
                int(rects[i][0, 1])-mask_pad:int(rects[i][2, 1])+mask_pad,
                int(rects[i][0, 0])-mask_pad:int(rects[i][2, 0])+mask_pad]
        
        # Get the transform for the eis application on this frame right now
        transform_mat = compute_LK_rigid_transform(cropped_frames[i-1],
                                                   cropped_frames[i])
        # We only care about translation right now, so ditch the part of
        # the transform matrix we don't need
        transform_mat = transform_mat[0:2, 2]
        #print (transform_mat)
        transforms[i-1] = transform_mat
        i += 1
    

# =============================================================================
    
    # ------------------------------------------------------------ #
    # DONE COMPUTING FLOW, NOW APPLY EIS
    # ------------------------------------------------------------ #
    
    # Apply transforms to cropped frames to decrease "jitter"
    # The window width for rolling average for smoothing motion is:
    # window = traj_pad*2 + 1
    traj_pad = 2
    smoothed_frames, transforms, smooth_transforms = \
        apply_eis(transforms, cropped_frames, traj_pad)
    
    
    # Crop the borders
    """
    for i in range(len(smoothed_frames)):
        #smoothed_frames[i] = smoothed_frames[i][mask_pad:-mask_pad, mask_pad:-mask_pad]
        smoothed_frames[i] = smoothed_frames[i][10:-10, 10:-10]
    """
    
    
    # ------------------------------------------------------------ #
    # SET UP THE OUTPUT DIRECTORY
    # ------------------------------------------------------------ #
    output_loc = os.path.join(output_loc, filename_timestamp)
    if not os.path.isdir(output_loc):
        print ("Warning: directory {} does not exist. "
               "Making directory...".format(output_loc))
        os.makedirs(output_loc)
    
    
    # ------------------------------------------------------------ #
    # OUTPUT OPTICAL FLOW RESULTS TO JSON FILE
    # ------------------------------------------------------------ #
    
    # Reformat the optical flow results to output to file
    proj, ht, sf, ofs = get_projection_from_nc(files_c06[0])
    
    out_dict = dict()
    out_dict['files_c06'] = []
    out_dict['files_c07'] = []
    out_dict['features_xy'] = dict()
    out_dict['features_lonlat'] = dict()
    out_dict['tracking_box_corners_xy'] = []
    out_dict['tracking_box_corners_lonlat'] = []
    
    # Convert features x, y to lon, lat
    for frame_num in range(num_samples):
        if frame_num in p_out.keys():
            out_dict['features_xy'][frame_num] = dict()
            out_dict['features_lonlat'][frame_num] = dict()
            for track_num in p_out[frame_num].keys():
                xy_val = list(p_out[frame_num][track_num][0].astype(float))
                out_dict['features_xy'][frame_num][track_num] = xy_val
                out_dict['features_lonlat'][frame_num][track_num] = \
                    xy_to_lonlat([xy_val[0]], [xy_val[1]], proj, ht, sf, ofs).tolist()
    
    #for k in p_out.keys():
    for k in range(num_samples):
        out_dict['files_c06'].append(os.path.basename(files_c06[k]))
        out_dict['files_c07'].append(os.path.basename(files_c07[k]))
        
        # Convert rect corners to lon,lat and add to dict
        xvals = [c[0] for c in rects[k]]
        yvals = [c[1] for c in rects[k]]
        lonlatvals = xy_to_lonlat(xvals, yvals, proj, ht, sf, ofs)
        out_dict['tracking_box_corners_xy'].append(rects[k].tolist())
        out_dict['tracking_box_corners_lonlat'].append(lonlatvals.tolist())
    
    # Save the transforms
    out_dict['eis_pad'] = mask_pad
    out_dict['eis_transforms'] = transforms.tolist()
    out_dict['eis_transforms_smoothed'] = smooth_transforms.tolist()
        
    # Output optical flow results to JSON file
    output_file = os.path.join(output_loc, "flow_features_") + filename_timestamp + ".json"
    with open(output_file, 'w') as fid:
        json.dump(out_dict, fid, indent=2)
    
    # ------------------------------------------------------------ #
    # PLOT STILL FRAMES FOR EACH HOUR
    # ------------------------------------------------------------ #
    
    # Make plots of the still frames every hour
    print ("")
    print ("Plotting one still frame per hour...")
    interval = datetime.timedelta(hours = 1.0)
    ts = first_timestamp
    last_timestamp = datetime.datetime. \
        strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S')
    frames_to_plot = []
    while ts <= last_timestamp:
        if datetime.datetime.strftime(ts, '%Y-%m-%d %H:%M:%S') in timestamps:
            frames_to_plot.append(timestamps.index(
                    datetime.datetime.strftime(ts, '%Y-%m-%d %H:%M:%S')))
        ts += interval
        print (ts)
    
    print ("Frames to plot: {}".format(frames_to_plot))
    
    output_file = os.path.join(output_loc, "stillframe_flow_")
    
    # Plot the stillframe images
    for i in frames_to_plot:
        ts = timestamps[i].replace(' ', '_').replace(':', '')
        # Plot the full image
        plot_frame(data = data_combined[i],
                   rects = [rects[i]],
                   colors = [[0.0, 0.0, 1.0]],
                   title = timestamps[i],
                   filename = output_file + "full_" + ts + ".png",
                   data_bounds = crop_bounds,
                   show_plot = False)
        # Plot the cropped image
        plot_frame(data = cropped_frames[i],
                   title = timestamps[i],
                   size_inches = (4, 4),
                   filename = output_file + "detail_" + ts + ".png",
                   show_plot = False)
    
    
    # ------------------------------------------------------------ #
    # PLOT THE ANIMATION
    # ------------------------------------------------------------ #
    # Plot the animation
    output_file = os.path.join(output_loc, "video_flow_") + filename_timestamp
    plot_animation(data_combined, timestamps,
                   rects=rects,
                   cropped_frames=smoothed_frames,
                   filename=output_file,
                   data_bounds=crop_bounds,
                   colors=frame_color,
                   with_tracks=False,
                   save_gif=False,
                   save_mp4=True)


def main():
    # ------------------------------------------------------------ #
    # PARSE INPUT ARGUMENTS
    # ------------------------------------------------------------ #
    
    config_file = None
    parser = argparse.ArgumentParser(description="Run optical flow for specified time range.")
    
    parser.add_argument("config_file", type=str,
                        help="Config file path.")
    
    parser.add_argument("-o", "--output_dir", type=str, default="./",
                        help="Location to save output files to.")
    
    parser.add_argument("--show_intermediate_plots", action="store_true",
                        help="Show plots of intermediate steps")
    
    args = parser.parse_args()
    
    #print (args)
    config_file = args.config_file
    #if config_file is not None:
    if not os.path.isfile(config_file):
        parser.error("Cannot find config file.")
    
    # Read the config file
    with open(config_file, 'r') as fid:
        cd = json.load(fid)
    
    # Set defaults
    start_hour = 0
    start_minute = 0
    end_hour = 23
    end_minute = 59
    
    # Check for missing arguments
    if ("data_path" not in cd.keys()):
        parser.error("Missing required option 'data_path' from config file.")
        
    if ("start_year" not in cd.keys() or
        "start_day" not in cd.keys() or
        "mask" not in cd.keys()):
        parser.error("Missing required option from config file.")
        
    if ("xlo" not in cd["mask"].keys() or
        "xhi" not in cd["mask"].keys() or
        "ylo" not in cd["mask"].keys() or
        "yhi" not in cd["mask"].keys()):
        parser.error("Missing bound(s) from config file mask.")
        
    # Add cropping bounds if included, else None
    crop_bounds = None
    if ("crop" in cd.keys()):
        if ("xlo" not in cd["crop"].keys() or
            "xhi" not in cd["crop"].keys() or
            "ylo" not in cd["crop"].keys() or
            "yhi" not in cd["crop"].keys()):
            parser.error("Missing bound(s) from config file crop.")
        else:
            crop_bounds = [cd['crop']['ylo'],
                           cd['crop']['yhi'],
                           cd['crop']['xlo'],
                           cd['crop']['xhi']]

    # Set the data path
    data_path = cd['data_path']
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(config_file), data_path))
    else:
        data_path = os.path.abspath(data_path)
    
    # Set the start datetime options
    start_year = cd['start_year']
    start_day = cd['start_day']
    if 'start_hour' in cd.keys():
        start_hour = cd['start_hour']
    if 'start_minute' in cd.keys():
        start_minute = cd['start_minute']
    
    # Set the end datetime options
    end_year = start_year
    end_day = start_day
    if 'end_year' in cd.keys():
        end_year = cd['end_year']
    if 'end_day' in cd.keys():
        end_day = cd['end_day']
    if 'end_hour' in cd.keys():
        end_hour = cd['end_hour']
    if 'end_minute' in cd.keys():
        end_minute = cd['end_minute']
    
    # Set the mask bounds
    mask_xlo = cd['mask']['xlo']
    mask_xhi = cd['mask']['xhi']
    mask_ylo = cd['mask']['ylo']
    mask_yhi = cd['mask']['yhi']
    
    # Parse the rest of the arguments
    output_loc = args.output_dir
    
    # ------------------------------------------------------------ #
    # SHOW START PLOTS
    # ------------------------------------------------------------ #
    # Run the plot stuff
    if args.show_intermediate_plots:
        show_intermediate_plots(data_path,
                                start_year, start_day, start_hour, start_minute,
                                end_year, end_day, end_hour, end_minute,
                                mask_ylo, mask_yhi, mask_xlo, mask_xhi)
        answer = input("Would you like to continue? (y/n):")
        if answer == "n":
            print ("Exiting...")
            sys.exit()

        print ("Continuing with run...")
    
    # ------------------------------------------------------------ #
    # START THE FULL RUN
    # ------------------------------------------------------------ #
    # Run the optical flow stuff
    run_optical_flow(data_path,
                     start_year, start_day, start_hour, start_minute,
                     end_year, end_day, end_hour, end_minute,
                     mask_ylo, mask_yhi, mask_xlo, mask_xhi,
                     crop_bounds, output_loc)
 

class NoFilesReadError(Exception):
    """Exception for when no nc files are found.
    
    Attributes
    ----------
    message: str
      explanation of the error
    """
    def __init__(self, message):
        self.message = message
