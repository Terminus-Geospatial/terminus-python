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
import unittest

#  Project Libraries
from tmns.dem.fixed import Fixed_DEM

class proj_fixed( unittest.TestCase ):

    def test_model(self):

        logger = logging.getLogger( 'proj_fixed.test_model' )

        model = Fixed_DEM( elevation_meters = 12345.6789 )
        logger.debug( model )

        #  Check corner points
        self.assertIsNone( model.min_corner_lla() )
        self.assertIsNone( model.max_corner_lla() )

        #  Check the elevation 
        self.assertAlmostEqual( model.elevation_meters( [ -105.210739, 39.609503 ] ), 12345.6789, 0.001 )
        self.assertAlmostEqual( model.elevation_meters( 'hello world' ), 12345.6789, 0.001 )
        self.assertAlmostEqual( model.elevation_meters( 12345 ), 12345.6789, 0.001 )
        
        #  Check stats
        self.assertTrue( model.has_stats() )
        self.assertAlmostEqual( model.stats.mean, 12345.6789, 1 )
        self.assertAlmostEqual( model.stats.std,  0, 1 )