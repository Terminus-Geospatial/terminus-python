#**************************** INTELLECTUAL PROPERTY RIGHTS ****************************#
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
import logging
import os
import sys
import unittest

if __name__ == '__main__':

    #  Define the start directory
    test_dir = os.path.dirname( sys.argv[0] )

    #  Initialize the system logger
    logging.basicConfig( level = logging.debug )

    loader = unittest.TestLoader()

    tests = loader.discover( pattern   = '*',
                             start_dir = test_dir )
    
    runner = unittest.runner.TextTestRunner()

    runner.run( tests )
