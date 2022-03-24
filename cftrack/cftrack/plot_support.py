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
# Functions for plotting single frame and video, getting HYSPLIT
# results from file, and for doing EIS.
#################################################################

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import datetime
import cv2
import json
import os

from .optical_flow import (find_features,
                           compute_optical_flow)
from .goes_io import (get_projection_from_nc, 
                      lonlat_to_xy,
                      get_radiance_from_nc, 
                      get_timestamp_from_nc)
from .image_manip import (to_uint8, 
                          equalize_data)

def plot_frame(data: np.ndarray,
               title: str = None,
               features: dict = None,
               rects: list = None,
               colors: list = None,
               data_bounds = None,
               show_plot: bool = True,
               filename: str = None,
               rect_linewidth: float = 2.0,
               title_size: float = 12.0,
               x_label: str = None,
               x_label_size: float = 10.0,
               y_label: str = None,
               y_label_size: float = 10.0,
               legend: list = None,
               legend_font_size: float = 8.0,
               size_inches: tuple = (15, 11),
               feature_size: int = 5,
               show_axes: bool = True,
               tick_font_size: float = 8.0,
               flip_y: bool = False,
               axis_units: str = 'pixels'):
    """Plot the given features on a still frame
    
    Parameters
    ----------
    data : numpy.ndarray
        The image to plot
    title : str, optional
        Image plot title. Usually used to display the image timestamp.
    features : dict, optional
        The dictionary of features to plot over the image.  This will have the
        structure of the dictionary outputted from find_features(), that is,
        it will have keys k in [0, 1, ..., m], where m is the number of
        features.  Each features[k] value will be the feature location as
        a length-3 array formatted as [x, nil, y].  Features are
        plotted as red dots.
    rects : list of 2D numpy.ndarray, optional
        Each numpy.ndarray should contain the
        vertices of the rectangle to plot, in order of
        top_left, top_right, bottom_right, bottom_left, top_left.  Should be
        a 2D array containing the [x, y] value for each vertex.
    colors : list of triplets, optional
        List of triplets specifying RGB in float values between 0 and 1.  Each
        triplet corresponds to one rectangle in rects.
    data_bounds : list or tuple, optional
        Bounds at which the image data is cropped for 
        display. This list/tuple is length-4, with bounds given in 
        numpy order:
            (y-lo, y-hi, x-lo, x-hi)
    show_plot : bool, default=True
        Whether to show the plot after saving.
    filename : str, optional
        Path to save the output file (without extension).  Will not
        save unless specified.
    rect_linewidth : float, default=2.0
        The width of the included rectangle.
    title_size : float, default=12.0
        The font size of each frame's title.
    x_label : str, optional
        The string label for the x axis.
    x_label_size : float, default=10.0
        The font size of the x axis label.
    y_label : str, optional
        The string label for the y axis.
    y_label_size : float, default=10.0
        The font size of the y axis.
    legend : list of str, optional
        List of strings to populate a legend.  Each string will 
        correspond to one rectangle in rects (in order).  If
        omitted, a legend will not be shown.
    legend_font_size : float, default=8.0
        The font size of legend strings.
    size_inches : tuple, default=(15, 11)
        The desired size to render the image in inches.  Default is
        15" x 11"
    feature_size : int, default=5
        The size of added feature points.
    show_axes : bool, default=True
        Determines whether to show the axes.
    tick_font_size : float, default=8.0
        The font size of axis tick labels.
    flip_y : bool, default=False
        Determines whether to flip just the y axis labels.  Default
        is False, meaning the origin will be on the upper left side
        of the plot.  If True, the origin will be moved to the bottom
        left side of the plot.
    axis_units : {'pixels', 'km'}, default='pixels'
        String representing the desired units.  Default is pixels.
        Currently, the only other supported unit is kilometers ('km').
        The GOES-R satellite image pixels cover about 2 km length-
        and height-wise, so axes ticks are displayed in km using 
        a 2-km pixel size.
    """
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(size_inches[0], size_inches[1])
    
    # Set up the data bounds
    if data_bounds is not None:
        rylo, ryhi, rxlo, rxhi = data_bounds
    else:
        rylo = 0
        rxlo = 0
        ryhi = data.shape[0]
        rxhi = data.shape[1]
    
    # Show the image
    ax.imshow(data[rylo:ryhi, rxlo:rxhi], cmap='gist_gray')
    
    # Plot the features, if they were inputted
    if features is not None:
        scatter_vals = np.array([v for v in features.values()])
        to_delete = []
        for i in range(len(scatter_vals)):
            if (scatter_vals[i][0, 0] < rxlo or
                scatter_vals[i][0, 0] > rxhi or
                scatter_vals[i][0, 1] < rylo or
                scatter_vals[i][0, 1] > ryhi):
                to_delete.append(i)
                #scatter_vals = np.delete(scatter_vals, i, axis=0)
        scatter_vals = np.delete(scatter_vals, 
                                 np.array(to_delete, dtype=np.int), 
                                 axis=0)
        ax.scatter(scatter_vals[:, 0, 0]-rxlo, 
                   scatter_vals[:, 0, 1]-rylo, 
                   s=feature_size, 
                   c='r')
    if rects is not None:
        if colors is None:
            colors = cm.rainbow(np.linspace(0, 1, len(rects)))
        for i in range(len(rects)):
            width = rects[i][2, 0] - rects[i][0, 0]
            height = rects[i][2, 1] - rects[i][0, 1]
            r = patches.Rectangle((rects[i][0, 0]-rxlo, rects[i][0, 1]-rylo),
                                  width, height, 
                                  edgecolor=colors[i],
                                  facecolor=None,
                                  fill=False,
                                  linewidth=rect_linewidth)
            ax.add_patch(r)
            
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=x_label_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=y_label_size)
    if legend is not None:
        ax.legend(legend, prop={"size": legend_font_size})
    if not show_axes:
        ax.axis('off')
    else:
        for l in (ax.get_xticklabels() + ax.get_yticklabels()):
            l.set_fontsize(tick_font_size)
    fig.tight_layout()
    #fig.canvas.draw_idle()
    
    if flip_y:
        ts = ax.get_yticks()
        yticklabelstmp = [str(int(l)) for l in ts]
        
        yticks = []
        yticklabels = []
        for t in range(len(ts)):
            if ts[t] >= 0 and ts[t] < ryhi-rylo:
                yticks.append(ryhi-rylo-1 - ts[t])
                yticklabels.append(yticklabelstmp[t])
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    
    if axis_units == 'km':
        # Note: Axes ticks are displayed in km using 2-km pixel size
        
        # Convert y axis
        if flip_y:
            ls = [str(2*int(l)) for l in yticklabels]
            ax.set_yticks(ax.get_yticks().tolist())
            ax.set_yticklabels(ls)
        else:
            ts = ax.get_yticks()
            yticklabelstmp = [str(2*int(l)) for l in ts]
            
            yticks = []
            yticklabels = []
            for t in range(len(ts)):
                if ts[t] >= 0 and ts[t] < ryhi-rylo:
                    yticks.append(ts[t])
                    yticklabels.append(yticklabelstmp[t])
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        
        # Convert x axis
        ts = ax.get_xticks()
        xticklabelstmp = [str(2*int(l)) for l in ts]
        
        xticks = []
        xticklabels = []
        for t in range(len(ts)):
            if ts[t] >= 0 and ts[t] < rxhi-rxlo:
                xticks.append(ts[t])
                xticklabels.append(xticklabelstmp[t])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        
    if filename is not None:
        fig.savefig(filename, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_animation(image_data: list,
                   timestamps: list,
                   lk_pts: dict = None,
                   rects: dict = None,
                   colors: list = None,
                   data_bounds = None,
                   cropped_frames = None,
                   with_tracks: bool = False,
                   filename: str = 'tmp',
                   save_gif: bool = True,
                   save_mp4: bool = False,
                   fps: int = 10,
                   rect_linewidth: float = 2.0,
                   title_size: float = 12.0,
                   x_label: str = None,
                   x_label_size: float = 10.0,
                   y_label: str = None,
                   y_label_size: float = 10.0,
                   size_inches: tuple = (15, 11),
                   feature_size: int = 5,
                   show_axes: bool = True,
                   tick_font_size: float = 8.0,
                   flip_y: bool = False,
                   axis_units: str = 'pixels'):
    """Plot and save an animation given a series of images,
    timestamps, and other items to plot on top.
    
    Parameters
    ----------
    image_data : list of numpy.ndarray
        Each numpy array contains one frame to be displayed.
    timestamps : list of str
        Each timestamp is included in the title above the 
        associated frame.
    lk_pts : dict of dicts of numpy.ndarray, optional
        The top-level dict has frame numbers for keys.  The
        following-level dict has feature numbers for keys.  For
        example, lk_pts[0][3] contains the pixel location of
        feature #3 in frame #0, and the pixel location is saved
        in a size-3 array, i.e. [x, [], y].
        lk_pts features are plotted as red dots.
    rects : dict of numpy.ndarray, optional
        The keys of this dict are the frame numbers.  Each value
        contains a 2D numpy array of the corner pixel locations
        for a rectangle, in order clockwise from the upper left
        and repeating the upper left location, i.e.
        [[ULx, ULy], [URx, URy], [LRx, LRy], [LLx, LLy], [ULx, ULy]]. 
        These are plotted as blue rectangles.
    colors : list of triplets, optional
        List of triplets specifying RGB in float values between 0 and 1.  Each
        triplet corresponds to one rectangle in rects.
    data_bounds : list or tuple, optional
        Bounds at which image_data (the larger image) is cropped for 
        display. This list/tuple is length-4, with bounds given in 
        numpy order:
            (y-lo, y-hi, x-lo, x-hi)
    cropped_frames : dict of numpy.ndarray, optional
        The keys of this dict are the frame numbers.  Each value
        contains a 2D numpy array of an image.  The image is meant
        to be a cropped & zoomed piece of the corresponding 
        image_data numpy array and is displayed side-by-side
        with the image_data numpy array.
    with_tracks : bool, default=False
        If True, plot the tracks created by the moving features.
        Must also have the lk_pts input.
    filename : str, optional
        Path to save the output file (without extension).  Will 
        default to 'tmp' in the current directory.
    save_gif : bool, default=True
        If True, save the animation as a gif.
        Note: save_gif and save_mp4 are NOT mutually exclusive.
        You may save both a gif and an mp4.
    save_mp4 : bool, default=False
        If True, save the animation as an mp4.
        Note: save_gif and save_mp4 are NOT mutually exclusive.
        You may save both a gif and an mp4.
    fps : int, default=10
        The frame rate (frames per second) of the saved video
        files.
    rect_linewidth : float, default=2.0
        The width of the included rectangle.
    title_size : float, default=12.0
        The font size of each frame's title.
    x_label : str, optional
        The string label for the x axis.
    x_label_size : float, default=10.0
        The font size of the x axis label.
    y_label : str, optional
        The string label for the y axis.
    y_label_size : float, default=10.0
        The font size of the y axis.
    size_inches : tuple, default=(15, 11)
        The desired size to render the video in inches.  Default is
        15" x 11"
    feature_size : int, default=5
        The size of added feature points.
    show_axes : bool, default=True
        Determines whether to show the axes.
    tick_font_size : float, default=8.0
        The font size of axis tick labels.
    flip_y : bool, default=False
        Determines whether to flip just the y axis labels.  Default
        is False, meaning the origin will be on the upper left side
        of the plot.  If True, the origin will be moved to the bottom
        left side of the plot.
    axis_units : {'pixels', 'km'}, default='pixels'
        String representing the desired units.  Default is pixels.
        Currently, the only other supported unit is kilometers ('km').
        The GOES-R satellite image pixels cover about 2 km length-
        and height-wise, so axes ticks are displayed in km using 
        a 2-km pixel size.
    """
    print ("")
    print ("Building animation...")
    num_frames = len(image_data)
    if rects is not None and cropped_frames is not None:
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])
        ax = []
        ax.append(plt.subplot(gs[0]))
        ax.append(plt.subplot(gs[1]))
        
    else:
        ax = []
        fig, axs = plt.subplots(1, 1)
        ax.append(axs)
    
    fig.set_size_inches(size_inches[0], size_inches[1])
    
    # Set up the data bounds
    if data_bounds is not None:
        rylo, ryhi, rxlo, rxhi = data_bounds
    else:
        rylo = 0
        rxlo = 0
        ryhi = image_data[0].shape[0]
        rxhi = image_data[0].shape[1]
    
    # Set up the tracks
    # tracks is a dict where the key is the track number associated with 
    # a given lk feature and the value is an array the length of the number
    # of frames, where each entry of the array is the location of that lk
    # feature in that frame.
    if with_tracks and lk_pts is not None:
        tracks = dict()
        num_pts = len(lk_pts[0])
        for i in range(num_pts):
            tracks[i] = []

        # For each frame, get the lk_pts. Append each point in lk_pts to its
        # appropriate track.
        for frame in range(num_frames):
            for k in lk_pts[frame].keys():
                tracks[k].append(lk_pts[frame][k][0])

    # Function to plot each frame of the animation
    def plot_data(i):
        
        # Clear the previous frame
        ax[0].clear()
        
        # Set the title and the image for the next frame
        print("Building frame {} of {}".format(i+1, num_frames))
        ax[0].set_title(timestamps[i], fontsize=title_size)
        ax[0].imshow(image_data[i][rylo:ryhi, rxlo:rxhi], cmap='gist_gray')
        
        # Set additional aesthetics if needed
        if x_label is not None:
            ax[0].set_xlabel(x_label, fontsize=x_label_size)
        if y_label is not None:
            ax[0].set_ylabel(y_label, fontsize=y_label_size)
        if not show_axes:
            ax[0].axis('off')
        else:
            for l in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                l.set_fontsize(tick_font_size)
        
        
        # Plot the features and the tracks, if requested
        if lk_pts is not None:
            if with_tracks:
                for j in range(num_pts):
                    if len(tracks[j]) < i+1:
                        tr = np.array(tracks[j])
                    else:
                        tr = np.array(tracks[j][:i+1])
                    ax[0].plot(tr[:, 0]-rxlo, tr[:, 1]-rylo, c='b')
            if i in lk_pts.keys():
                lp = np.array([v for v in lk_pts[i].values()])
                ax[0].scatter(lp[:, 0, 0]-rxlo,
                              lp[:, 0, 1]-rylo,
                              s=feature_size, 
                              c='r')
        
        if flip_y:
            ts = ax[0].get_yticks()
            yticklabelstmp = [str(int(l)) for l in ts]
            
            yticks = []
            yticklabels = []
            for t in range(len(ts)):
                if ts[t] >= 0 and ts[t] < ryhi-rylo:
                    yticks.append(ryhi-rylo-1 - ts[t])
                    yticklabels.append(yticklabelstmp[t])
            ax[0].set_yticks(yticks)
            ax[0].set_yticklabels(yticklabels)
        
        if axis_units == 'km':
            # Note: Axes ticks are displayed in km using 2-km pixel size
            
            # Convert y axis
            if flip_y:
                ls = [str(2*int(l)) for l in yticklabels]
                ax[0].set_yticks(ax[0].get_yticks().tolist())
                ax[0].set_yticklabels(ls)
            else:
                ts = ax[0].get_yticks()
                yticklabelstmp = [str(2*int(l)) for l in ts]
                
                yticks = []
                yticklabels = []
                for t in range(len(ts)):
                    if ts[t] >= 0 and ts[t] < ryhi-rylo:
                        yticks.append(ts[t])
                        yticklabels.append(yticklabelstmp[t])
                ax[0].set_yticks(yticks)
                ax[0].set_yticklabels(yticklabels)
            
            # Convert x axis
            ts = ax[0].get_xticks()
            xticklabelstmp = [str(2*int(l)) for l in ts]
            
            xticks = []
            xticklabels = []
            for t in range(len(ts)):
                if ts[t] >= 0 and ts[t] < rxhi-rxlo:
                    xticks.append(ts[t])
                    xticklabels.append(xticklabelstmp[t])
            ax[0].set_xticks(xticks)
            ax[0].set_xticklabels(xticklabels)
                
        # Plot the rectangles if included
        if rects is not None:
            # Plot the rects
            if i in rects.keys():
                if colors is not None:
                    ax[0].plot(rects[i][:, 0]-rxlo,
                               rects[i][:, 1]-rylo,
                               c = colors[i],
                               linewidth=rect_linewidth)
                #if transition_frame is not None:
                #    frame_color = [[0.0, val, 1.0 - val] for val in transition_frame]
                #    ax[0].plot(rects[i][:, 0]-rxlo, 
                #               rects[i][:, 1]-rylo, 
                #               c=frame_color[i])
                else:
                    ax[0].plot(rects[i][:, 0]-rxlo, 
                               rects[i][:, 1]-rylo, 
                               c=[0.0, 0.0, 1.0],
                               linewidth=rect_linewidth)
            
            # Plot the cropped frames next to the whole image if
            # they were inputted
            if cropped_frames is not None:
                ax[1].clear()
                ax[1].set_title(timestamps[i], fontsize=title_size)
                ax[1].imshow(cropped_frames[i], cmap='gist_gray')
                
                # Set additional aesthetics if needed
                if x_label is not None:
                    ax[1].set_xlabel(x_label, fontsize=x_label_size)
                if y_label is not None:
                    ax[1].set_ylabel(y_label, fontsize=y_label_size)
                if not show_axes:
                    ax[1].axis('off')
                else:
                    for l in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
                        l.set_fontsize(tick_font_size)
                
                yhi = cropped_frames[i].shape[0]
                ylo = 0
                xhi = cropped_frames[i].shape[1]
                xlo = 0
                if flip_y:
                    ts = ax[1].get_yticks()
                    yticklabelstmp = [str(int(l)) for l in ts]
                    
                    yticks = []
                    yticklabels = []
                    for t in range(len(ts)):
                        if ts[t] >= 0 and ts[t] < yhi-ylo:
                            yticks.append(yhi-ylo-1 - ts[t])
                            yticklabels.append(yticklabelstmp[t])
                    ax[1].set_yticks(yticks)
                    ax[1].set_yticklabels(yticklabels)
                
                if axis_units == 'km':
                    # Note: Axes ticks are displayed in km using 2-km pixel size
                    
                    # Convert y axis
                    if flip_y:
                        ls = [str(2*int(l)) for l in yticklabels]
                        ax[1].set_yticks(ax[1].get_yticks().tolist())
                        ax[1].set_yticklabels(ls)
                    else:
                        ts = ax[1].get_yticks()
                        yticklabelstmp = [str(2*int(l)) for l in ts]
                        
                        yticks = []
                        yticklabels = []
                        for t in range(len(ts)):
                            if ts[t] >= 0 and ts[t] < yhi-ylo:
                                yticks.append(ts[t])
                                yticklabels.append(yticklabelstmp[t])
                        ax[1].set_yticks(yticks)
                        ax[1].set_yticklabels(yticklabels)
                    
                    # Convert x axis
                    ts = ax[1].get_xticks()
                    xticklabelstmp = [str(2*int(l)) for l in ts]
                    
                    xticks = []
                    xticklabels = []
                    for t in range(len(ts)):
                        if ts[t] >= 0 and ts[t] < xhi-xlo:
                            xticks.append(ts[t])
                            xticklabels.append(xticklabelstmp[t])
                    ax[1].set_xticks(xticks)
                    ax[1].set_xticklabels(xticklabels)

    # Build the animation
    ani = animation.FuncAnimation(fig, plot_data,
                                  frames=num_frames,
                                  interval=50,
                                  blit=False,
                                  repeat=True)

    # Save the animation
    if save_gif:
        print ("Saving animation to file...")
        ani.save(filename + '.gif', writer = 'imagemagick', fps = fps)
    
    if save_mp4:
        print ("Saving mp4...")
        ani.save(filename + '.mp4', writer = 'ffmpeg', fps = fps, dpi = 160)
    
    # If default mp4 codec isn't available...
    #ani.save(filename + '.mp4', writer = 'ffmpeg', codec='mpeg4',
    #         fps=5, dpi=160)
    


def unpack_json(image_path: str, 
                json_path: str):
    """Unpacking vital data from a cftrack-produced .json file for plotting.
    
    Parameters
    ----------
    image_path : str
        Path to GOES_data_dir directory
    json_path : str
        Path to .json file
    
    Returns
    -------
    list of 2D numpy.ndarray of uint8 format
        List of arrays of combined radiance data; one array per file
    list of str
        List of image timestamps in format %Y-%m-%d %H:%M:%S
    dict of 2D numpy.ndarray
        Dictionary of arrays of the corner locations of the tracking box; 
        one array per file; dictionary keys are integers
    dict of dictionaries of 2D numpy.ndarray
        local feature locations in [x,y] coordinates within a local frame; 
        e.g. lk_pts[i][k] is the kth point of frame i; all dict keys are integers
    """
    
    # Read in the optical flow results file
    with open (json_path, 'r') as file:
        data = json.load(file)

    # Extract image filenames, frame tracking boxes, and local features
    f_c06 = data['files_c06']
    f_c07 = data['files_c07']
    
    # tracking boxes
    # Convert .json's string keys to integer keys
    num_frames = len(data['tracking_box_corners_xy'])
    box_corners = dict()
    for frame in range(num_frames):
        box_corners[frame] = np.asarray(data['tracking_box_corners_xy'][frame])
    
    # local features
    # Convert .json's string keys to integer keys
    lk_features = data['features_xy']
    lk_pts = dict()
    for frame in lk_features.keys():
        lk_pts[int(frame)] = dict()
        for feat in lk_features[frame].keys():
            lk_pts[int(frame)][int(feat)] = np.asarray([[lk_features[frame][feat][0],
                                                         lk_features[frame][feat][1]]])
    
    # Get all the full paths to the files
    files_c06 = []
    files_c07 = []
    for fn6, fn7 in zip(f_c06, f_c07):
        dt = fn6.split('_')[3]
        year = dt[1:5]
        day = dt[5:8]
        files_c06.append(os.path.join(image_path, year, day, 'C06', fn6))
        files_c07.append(os.path.join(image_path, year, day, 'C07', fn7))
    
    # Get the timestamps for each set of images
    timestamps = []
    for f6 in files_c06:
        ts = get_timestamp_from_nc(f6)
        timestamps.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Read in the data
    k = 0
    data_combined = []
    for f6, f7 in zip(files_c06, files_c07):
        print ("Reading and combining bands for frame {} of {}".format(k+1, num_frames))
    
        data_c06, _, _ = get_radiance_from_nc(f6)    
        data_c07, _, _ = get_radiance_from_nc(f7)
        
        # Combine the data by:
        # 1. C06 - C07
        # 2. Equalize the histogram of the entire combined image
        #    (i.e. 'linearizing' the effective cdf)
        # 3. Convert to uint8 - the feature-selection algorithm uses
        #    only uint8 or float32 data
        
        tmp = data_c06 - data_c07
        tmp = equalize_data(tmp)
        # The data must be uint8 for input to opencv LK
        data_combined.append(to_uint8(tmp, mx=1.0, mn=0.0))
    
        # Increment frame counter
        k += 1
    
    # Return image data, timestamps, tracking boxes, and feature points
    return (data_combined, timestamps, box_corners, lk_pts)


def compute_LK_rigid_transform(frame1: np.ndarray,
                               frame2: np.ndarray):
    """Compute a rigid transform between two frames by finding
    and matching shi-tomasi features in both.
    
    Parameters
    ----------
    frame1 : numpy.ndarray
        The first image frame in a sequence.
    frame2 : numpy.ndarray
        The second image frame in a sequence.
    
    Returns
    -------
    numpy.ndarray
        The transform matrix.
    """
    # Find features in frame1
    prev_pts_dict = find_features(frame1)

    # Compute flow to find the frame1 features in frame2
    new_pts_dict = compute_optical_flow(frame1,
                                        frame2,
                                        prev_pts_dict)

    #print ("Prev eis pts: {}, new eis pts: {}".format(len(prev_pts_dict), len(new_pts_dict)))
    # Get rid of any feature points that were not located in
    # frame2
    if len(new_pts_dict) != len(prev_pts_dict):
        todel = []
        for k in prev_pts_dict.keys():
            if k not in new_pts_dict.keys():
                todel.append(k)
        for k in todel:
            del(prev_pts_dict[k])

    # Convert from dict to array
    new_pts = np.array([new_pts_dict[k] for k in new_pts_dict.keys()])[:, 0, :]
    prev_pts = np.array([prev_pts_dict[k] for k in prev_pts_dict.keys()])[:, 0, :]

    # Get the transform matrix
    transform_mat = cv2.estimateRigidTransform(prev_pts, new_pts, fullAffine=False)
    if transform_mat is None:
        transform_mat = np.zeros((2, 3))
    
    return transform_mat


def apply_eis(transforms: np.ndarray,
              frames: dict,
              window_hw: int = 2,
              time_axis: int = 0):
    """Apply a series of transforms to image frames after
    smoothing the motion via a rolling average.
    
    Parameters
    ----------
    transforms : numpy.ndarray
        An array containing all of the transform matrices
        (or vectors if translation only).  Each transformation
        matrix in the array must be either shape (2, 3) for a
        full affine transform or shape (2, 1) for just a
        translation.
    frames : dict
        A dictionary with sequentially increasing integers as its
        keys and the corresponding numpy.ndarray frames, in 
        order, as its values.
    window_hw : int, default=2
        The window half-width, in pixels, for the rolling
        average.
    time_axis : int, default=0
        The axis of the multi-dimensional transforms array
        which corresponds to frame number or the time dimension.
    
    Returns
    -------
    dict
        The motion-smoothed image frames, stored as the values
        in the dictionary.  The frame indices are the keys.
    numpy.ndarray
        The inputted `transforms` array
    numpy.ndarray
        The smoothed transforms that were applied to create the
        motion-smoothed frames
    """
    
    # The creation of this smoothing function was guided, in part,
    # by a tutorial at learnopencv.com (Abhishek Singh Thakur, 
    # "Video Stabilization Using Point Feature Matching in OpenCV,"
    # 2019, url: https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/)
    
    # Get the cumulative sum of the transforms for all the frames
    trajectory = np.cumsum(transforms, axis=time_axis)
    smoothed_trajectory = np.zeros(trajectory.shape)

    # Pad the trajectory so we can apply the rolling average
    # Zeros in front, copies of last value at end
    end_pad = []
    for i in range(window_hw):
        end_pad.append(trajectory[-1])
    end_pad = np.array(end_pad)
    padded_trajectory = np.concatenate(
        (np.zeros((window_hw, 2)), trajectory, end_pad))

    # Apply rolling average with window width window_hw*2
    for i in range(window_hw, len(padded_trajectory)-window_hw):
        smoothed_trajectory[i-window_hw] = \
            sum(padded_trajectory[i-window_hw:i+window_hw+1, :])/(2.0*window_hw+1.0)

    # Compute and apply the smoothed transforms
    tmp = smoothed_trajectory - trajectory
    smooth_transforms = transforms + tmp

    # Create a dictionary for the transformed frames and apply
    # the smoothed transforms to each frame
    smoothed_frames = dict()
    out_shape = (frames[0].shape[1], frames[0].shape[0])
    
    # If you just have the translation array, add the 
    # identity for the rotation to make it shape 2x3
    if (smooth_transforms[0].shape == (2,) or
        smooth_transforms[0].shape == (2, 1)):
        for i in range(len(smooth_transforms)):
            T = np.array([[1, 0, smooth_transforms[i, 0]],
                          [0, 1, smooth_transforms[i, 1]]])
            # Apply the transformation matrix
            smoothed_frames[i] = cv2.warpAffine(frames[i],
                                                T,
                                                out_shape,
                                                flags = cv2.INTER_CUBIC)
    
    # Otherwise just apply the transformation matrix as inputted
    else:
        for i in range(len(smooth_transforms)):
            T = smooth_transforms[i]
            smoothed_frames[i] = cv2.warpAffine(frames[i],
                                                T,
                                                out_shape,
                                                flags = cv2.INTER_CUBIC)

    # The last frame does not have a transform applied to it
    smoothed_frames[len(frames)-1] = frames[len(frames)-1]

    # Crop the borders
    for i in range(len(smoothed_frames)):
        smoothed_frames[i] = smoothed_frames[i][10:-10, 10:-10]

    return smoothed_frames, transforms, smooth_transforms
