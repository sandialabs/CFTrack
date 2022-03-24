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
# Python package setup for SILT application.
#################################################################

from setuptools import setup

# Grab the version number stored in main.py
with open("cftrack/main.py", 'r') as fid:
    lines = fid.readlines()

for line in lines:
    if "__version__" in line:
        tmp = line.split()
        version = tmp[-1][1:-1]
        break

setup(
    name="cftrack",
    description="Sandia Python tools for following cloud motion via sparse optical flow in NASA GOES-17 CONUS imagery.",
    version=version,
    url="https://cee-gitlab.sandia.gov/cloudaerosols/cftrack",
    author_email="kmlarso@sandia.gov",
    install_requires=[
        "opencv-python-headless>=3,<4",
        "numpy",
        "matplotlib",
        "h5py",
        "pyproj",
        "scikit-image",
        "ffmpeg",
        "pvlib"
    ],
    packages=["cftrack",],
    entry_points={"console_scripts": ['cftrack = cftrack.main:main',
                                      'cftrack-run = cftrack.run_optical_flow:main']},
)
