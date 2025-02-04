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
import os
import unittest

#  Project Libraries
import tmns.dem.gtiff as gt

class proj_gtiff( unittest.TestCase ):

    def test_model(self):

        dem_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                   '../data/Denver_SRTM_GL1.tif' ) )
        self.assertTrue( os.path.exists( dem_path ) )

        #  Load model
        model = gt.DEM_File( dem_path )
        self.assertFalse( model.load() )

        #  Check corner points
        min_corner = model.min_corner_lla()
        self.assertAlmostEqual( min_corner[0], -105.29680555554567, 0.0001 )
        self.assertAlmostEqual( min_corner[1],   39.96736111110846, 0.0001 )

        max_corner = model.max_corner_lla()
        self.assertAlmostEqual( max_corner[0], -104.65958333332337, 0.0001 )
        self.assertAlmostEqual( max_corner[1],   39.4209722222195,  0.0001 )

        #  Check the elevation 
        elev_m = model.elevation_meters( [ -105.210739, 39.609503 ] )
        self.assertAlmostEqual( elev_m, 2377.7, 1 )
        elev_m = model.elevation_meters( [ -104.764170, 39.947794 ] )
        self.assertAlmostEqual( elev_m, 1557, 1 )
        
        #  Check stats
        self.assertTrue( model.has_stats() )
        self.assertAlmostEqual( model.stats.mean, 1789.82460, 1 )
        self.assertAlmostEqual( model.stats.std,   225.55451, 1 )