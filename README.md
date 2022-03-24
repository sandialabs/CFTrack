# CFTrack

Sandia Cloud Feature Tracker for following low-cloud motion in GOES-R imagery.

_Please note that this tool was developed for GOES-17 CONUS data and has not been thoroughly tested on other imagery._

![Tracking exmple](/preview/jun_sample_flow.mp4 "Tracking example")*Video example showing the gradual dispersal of aerosol injections over 29 hours.*

<!-- blank line -->
<br>
<!-- blank line -->

## Table of Contents

- [Prerequisites](#prerequisites)
- [CFTrack Installation](#cftrack-installation)
- [Basic Use](#basic-use)
- [Advanced Use](#advanced-use)
- [Auto-generated API Documentation](#auto-generated-api-documentation)

## Prerequisites

#### Python
Please install:
- python distribution with pip 
    - python version >= 3.6 and < 3.9

If you are new to Python, my recommendation is to use [Anaconda Python](https://www.anaconda.com/products/individual).  See [here](https://docs.anaconda.com/anaconda/install/) for further installation instructions.

_Note: if you are behind a firewall with a proxy server (like at some companies and universities), you may have to set proxy variables for pip and Anaconda in order for them to have internet access.  Contact your system administrators to determine the appropriate proxy settings. To apply them, you can find instructions [here](https://pip.pypa.io/en/stable/user_guide/#using-a-proxy-server) for pip and instructions [here](https://docs.anaconda.com/anaconda/user-guide/tasks/proxy/) for Anaconda._

#### Python Packages
There's no need to install these now.  As long as you have an internet connection, the following Python packages will be installed automatically when you install CFTrack:
- opencv-python-headless = 3
- numpy >= 1.15
- matplotlib >= 3.1
- h5py >= 2.8
- pyproj >= 2.4
- scikit-image >= 0.16
- ffmpeg
- pvlib

#### GOES-17 Data
The GOES-17 continental US (CONUS) L1b radiance band 6 and 7 satellite image data must be available via path reference through your computer.  The image data is available and can be downloaded either via [AWS](https://registry.opendata.aws/noaa-goes/) or the [NOAA CLAss](https://www.avl.class.noaa.gov/saa/products/search?datatype_family=GRABIPRD) website.

All image files should be located in a GOES data directory. This tool expects the image files to be in a specific directory structure with certain file naming standards.  The images must be organized by year, day, and band, e.g. `<GOES_image_dir_name>/yyyy/ddd/bnd/`, where yyyy is the 4-digit year, ddd is the 3-digit day of the year, and bnd is the 3-charactor GOES band (e.g. 'C07' for band 7).  It also expects the following file naming convention, typical of NOAA GOES imagery: `*yyyydddhhmm*.nc` where yyyy is the 4-digit year, ddd is the 3-digit day, hh is the 2-digit hour in 24-hr format, and mm is the 2-digit minute (* is a wildcard and can be any characters).  Here is a sample directory structure containing some image files for days 168 and 169 in 2019:

```bash
GOES_image_dir
├── 2019
    ├── 168
        ├── C06
            ├── OR_ABI-L1b-RadC-M6C06_G17_s20191680501197_e20191680503575_c20191680504004.nc
            ├── OR_ABI-L1b-RadC-M6C06_G17_s20191680506197_e20191680508575_c20191680509004.nc
            ├── OR_ABI-L1b-RadC-M6C06_G17_s20191680511197_e20191680513576_c20191680514005.nc
        ├── C07
            ├── OR_ABI-L1b-RadC-M6C07_G17_s20191680501197_e20191680503581_c20191680504014.nc
            ├── OR_ABI-L1b-RadC-M6C07_G17_s20191680506197_e20191680508581_c20191680509014.nc
            ├── OR_ABI-L1b-RadC-M6C07_G17_s20191680511197_e20191680513581_c20191680514015.nc
    ├── 169
        ├── C06
            ├── OR_ABI-L1b-RadC-M6C06_G17_s20191690501198_e20191690503576_c20191690504005.nc
            ├── OR_ABI-L1b-RadC-M6C06_G17_s20191690506198_e20191690508576_c20191690509005.nc
            ├── OR_ABI-L1b-RadC-M6C06_G17_s20191690511198_e20191690513576_c20191690514007.nc
        ├── C07
            ├── OR_ABI-L1b-RadC-M6C07_G17_s20191690501198_e20191690503582_c20191690504017.nc
            ├── OR_ABI-L1b-RadC-M6C07_G17_s20191690506198_e20191690508581_c20191690509016.nc
            ├── OR_ABI-L1b-RadC-M6C07_G17_s20191690511198_e20191690513582_c20191690514014.nc
```


## CFTrack Installation

0. Install Python

If you are new to Python, I recommend installing [Anaconda Python](https://www.anaconda.com/products/individual) and following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) to create and activate a new environment with Python 3.8.

Otherwise, use the distribution of your choice, but make sure it comes with pip.

1. Clone the repo

```bash
git clone git@cee-gitlab.sandia.gov:cloudaerosols/cftrack.git CFTRACK
```

2. Change to top-level directory

```bash
cd CFTRACK
```

The directory structure should look like:

```bash
CFTRACK
├── cftrack/
├── CHANGELOG.md
├── docs/
├── preview/
├── README.md
├── samples/
...
```

3. Install CFTrack

This package is set up to be installed via pip:

``` bash
pip install ./cftrack
```

Pip will search for and install any missing prerequisite Python packages before installing CFTrack.

4. Download and set up the GOES data to run the example.
   - Download all of the GOES G17 CONUS band 6 and band 7 images between 0500 and 0707 UTC on June 17, 2019 (day 168).
   - Set up and populate your directory structure according to the instructions above using the imagery you just downloaded.

5. Update the samples/test_config_jun.json file.  Make sure the **data_path** argument is set to point to the data directory structure from step 4.  In the example above, this would be the path to the "GOES_image_dir" directory.  

_Note: this path can either be absolute or relative to the test_config_jun.json file.  When specifying absolute paths, avoid using the "\~" home directory shortcut and instead use the whole path (e.g. "/Users/janedoe/" instead of "\~/")._

6. To run CFTrack, call cftrack-run from the command line with the config file as input:

``` bash
cftrack-run samples/test_config_jun.json
```

This will create a new folder in the current working directory to write the results, including imagery and video.  It will be named by the start and end dates and times you chose in the config file, e.g. "<start date-time>_<end date-time>".
If you wish to output the files to a different location instead of the current working directory, you can use the following call (make sure to replace `<output file path>` with your own desired output path):

``` bash
cftrack-run samples/test_config_jun.json -o <output file path>
```

_Note: Your terminal may appear to freeze for about 30 seconds before running the first time.  This is normal - the program is importing dependencies._

## Basic Use

To track cloud features of your choice:

### 1. Create a custom .json file

For writing your own .json file, include at least the following arguments in your config file:

**Data Path Argument**

- **data_path**:
    Path to the data directory structure.  Both relative and absolute paths are supported.  In the example above, this would be the path to "GOES_image_dir."
    

**Date Range Arguments**

- Start Date and Time
    
    - **start_year**:
    Start year for GOES data series.  Should be four-digit
    number.

    - **start_day**:
    Start day for GOES data series in day-of-year format.
    
    - **start_hour**:
    Start hour for GOES data series in 24-hour format, e.g.
    23 for 11 pm.
    
    - **start_minute**:
    Start minute for GOES data series.  Must be between 0 and
    59, inclusive.

- End Date and Time
    
    _Note: The data collector is inclusive for year, day, hour, and minute settings; e.g. if end\_day is 139, then day 139 is included in the data series._

    - **end_year**:
    End year for GOES data series.  Should be four-digit
    number.
    
    - **end_day**:
    End day for GOES data series in day-of-year format.
    
    - **end_hour**:
    End hour for GOES data series in 24-hour format.
    
    - **end_minute**:
    End minute for GOES data series.

**Image Mask Arguments**

The mask bounds the area of clouds you wish to track.  Features will be detected only within the mask boundaries.  The mask boundaries must be specified in pixel space.  Keep in mind that the origin (x, y)=(0, 0) is located at the top left corner of the image.

_Note: Do not worry about setting the mask location correctly at first. The next step details how to find the right location._

_Note: a box size of around 50x50 pixels is granular enough to follow specific cloud features while reducing the time necessary to run the algorithm._

- **mask**: {
    - **xlo**:
        Tracking box mask lower (closer to zero) x boundary.

    - **xhi**:
        Tracking box mask upper x boundary.

    - **ylo**:
        Tracking box mask lower (closer to zero) y boundary.

    - **yhi**:
        Tracking box mask upper y boundary.
}

Optionally, you may also specify crop boundaries for the final video footage to zoom into. This can give you a more detailed view of the tracking box movement across time. The following arguments must be specified:

- **crop**: {
    - **xlo**:
        Crop window lower (closer to zero) x boundary.

    - **xhi**:
        Crop window upper x boundary.

    - **ylo**:
        Crop window lower (closer to zero) y boundary.

    - **yhi**:
        Crop window upper y boundary.
}


### 2. Check Mask Location

To locate the cloud features you wish to track, run

``` bash
cftrack-run <config file path> -o <output file path> --show_intermediate_plots
```

This will return three plots from the first relevant image of the image series. The first is the entire image with your mask rectangle outlined in blue. Second is a zoomed-in image centered on the masked rectangle. Third is the image bounded by the mask with the local feature points the algorithm will begin tracking.

When prompted whether to continue, stop if you need to relocate the mask. If necessary, revise your config file and check that the mask is placed correctly using the same code again before continuing.

After verifying the mask location is correct, continue running the algorithm.

### 3. Run CFTrack

Run the CFTrack algorithm (without showing the plots from Step 2) via the command

``` bash
cftrack-run <config file path> -o <output file path>
```

This will output a series of files to the specified output location or the current working directory by default.  The file series includes:

- a .json file, which contains tracking box corner coordinates (both pixel space and in lon/lat), the feature locations (in both pixel space and in lon/lat), and eis information for each pair of C06 and C07 GOES-R ABI radiance products

- a .mp4 video file showing the video result of the optical flow tracking

- a collection of .png image files, one still frame per hour of the full-scene video as well as the zoomed-in analysis focused on the tracking box

The .json file is useful for customizing image or video outputs. See the **Advanced Use** section below for more details.

## Advanced Use

You can customize frame and animation plots as needed after running the CFTrack algorithm. The sample/sample_scripts directory contains sample code for plotting a frame as well as an animation. Use this code to display the tracked local cloud feature points as well as change plot aesthetics. See the documentation on `plot_frame()` and `plot_animation()` for arguments you can add to these functions.

The simplest way to run the code is to run

``` bash
python sample/sample_scripts/custom_plots.py <GOES_image_dir_name> <json_file_path>
```

substituting <GOES_image_dir_name> for the directory path of your GOES imagery database. Substitute <json_file_path> for the file path to the .json file created by the CFTrack algorithm.

## Auto-generated API Documentation

API documentation is auto-generated as a series of html files using Sphinx; however, the html files must be built by the user.  Follow the instructions in the README in the docs/ directory to build and view the html API documentation.

