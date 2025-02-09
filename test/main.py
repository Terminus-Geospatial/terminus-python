#!/usr/bin/env python3
# **************************** INTELLECTUAL PROPERTY RIGHTS ****************************#
#*                                                                                    *#
#*                           Copyright (c) 2025 Terminus LLC                          *#
#*                                                                                    *#
#*                                All Rights Reserved.                                *#
#*                                                                                    *#
#*          Use of this source code is governed by LICENSE in the repo root.          *#
#*                                                                                    *#
#**************************** INTELLECTUAL PROPERTY RIGHTS ****************************#
#

#  Python Libraries
import argparse
import logging
import os
import sys
import unittest

def parse_command_line():

    parser = argparse.ArgumentParser( description = 'Run Unit-Tests for Terminus Python libraries' )

    parser.add_argument( '-v', 
                         dest = 'log_level',
                         default = logging.INFO,
                         action = 'store_const',
                         const = logging.DEBUG,
                         help = 'Use verbose logging.' )
    
    return parser.parse_args()

if __name__ == '__main__':

    cmd_args = parse_command_line()

    #  Define the start directory
    test_dir = os.path.dirname( sys.argv[0] )

    #  Initialize the system logger
    logging.basicConfig( level = cmd_args.log_level )

    #  Remove particularly obnoxious loggers
    logging.getLogger( 'rasterio._base' ).disabled     = True
    logging.getLogger( 'rasterio._env' ).disabled      = True
    logging.getLogger( 'rasterio._filepath' ).disabled = True
    logging.getLogger( 'rasterio._io' ).disabled       = True
    logging.getLogger( 'rasterio.env' ).disabled       = True

    loader = unittest.TestLoader()

    tests = loader.discover( pattern   = '*',
                             start_dir = test_dir )
    
    runner = unittest.runner.TextTestRunner()

    runner.run( tests )
