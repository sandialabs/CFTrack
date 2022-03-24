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
# Entry point to print version and contact info.
# Also store version #.
#################################################################

__version__ = "1.0.0"


def main():
    global __version__

    print ("CFTrack version {}".format(__version__))
    print ("Sandia National Labs")
    print ("08/31/2021")
    print ("Authors: Kelsie Larson, Skyler Gray, Don Lyons")
    print ("Contact: Kelsie Larson, kmlarso@sandia.gov")

    print ("For usage info, type cftrack-run -h")
