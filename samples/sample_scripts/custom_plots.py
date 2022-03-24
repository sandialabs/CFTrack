#################################################################
# Copyright 2022 National Technology & Engineering Solutions of 
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 
# with NTESS, the U.S. Government retains certain rights in this 
# software.
#
# Sandia National Labs
# Date: 2021-09-01
# Authors: Skyler Gray, Kelsie Larson
# Contact: Kelsie Larson, kmlarso@sandia.gov
#
# Sample code for customizing frame and animation plots from
# cftrack-run .json output file.
#################################################################

import os
import argparse

from cftrack.plot_support import (plot_frame,
                                  plot_animation,
                                  unpack_json)

def main():
    
    # ARGUMENT CODE; DO NOT CHANGE
    parser = argparse.ArgumentParser(description='This is sample script to generate custom plots.')
    parser.add_argument("image_path", 
                        help = "The custom path of the GOES image directory structure.")
    parser.add_argument("json_path", 
                        help = "The custom path of the .json file generated from the cftrack-run command.")
    args = parser.parse_args()
    image_path = args.image_path
    json_path = args.json_path
    # END OF ARGUMENT CODE
    
    # Get image data, timestamps, tracking boxes, and feature points
    data_combined, timestamps, rects, lk_pts = unpack_json(image_path, json_path)
    
    
    ####################################################################
    # PLOT CODE: EDIT THIS CODE TO CHANGE PLOT AESTHETICS AND FEATURES #
    ###################################################################
    # default save path
    save_path = os.getcwd()
    
    # Plot first image with cloud feature points
    i = 0
    plot_frame(data_combined[i],
               title = timestamps[i],
               features = lk_pts[i],
               rects = [rects[i]],
               colors = [(1, 0, 0)],
               data_bounds = (300, 600, 1000, 1500),
               show_plot = False,
               filename = os.path.join(save_path, 'frame_temp'),
               rect_linewidth = 4.0,
               title_size = 24.0,
               x_label = 'km',
               x_label_size = 18.0,
               y_label = 'km',
               y_label_size = 18.0,
               tick_font_size = 12.0,
               flip_y = True,
               axis_units = 'km')
    
    # plot local feature points
    plot_animation(data_combined,
                   timestamps = timestamps,
                   lk_pts = lk_pts,
                   rects = rects,
                   save_gif = True,
                   with_tracks = True,
                   filename = os.path.join(save_path, 'temp_pts'),
                   data_bounds = (300, 600, 1000, 1500),
                   rect_linewidth = 4.0,
                   title_size = 24.0,
                   x_label = 'km',
                   x_label_size = 18.0,
                   y_label = 'km',
                   y_label_size = 18.0,
                   tick_font_size = 12.0,
                   flip_y = True,
                   axis_units = 'km')
    
    ####################
    # END OF PLOT CODE #
    ####################


if __name__ == '__main__':
    main()
