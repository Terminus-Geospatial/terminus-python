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
import unittest

#  Numpy
import numpy as np

#  Terminus Libraries
from tmns.core.types import GCP
from tmns.dem.gtiff import DEM_File as DEM
import tmns.proj.RPC00B as RPC00B
import tmns.proj.RSM 

def load_rpc_file( rpc_path ):

    data = {}
    with open( rpc_path, 'r' ) as fin:
        for line in fin.readlines():
            parts = line.replace(' ','').strip().split(':')

            term = RPC00B.Term.from_str(parts[0])
            value = float(parts[1])
            data[term] = value

    #  Create RPC object from data
    return RPC00B.RPC00B.from_dict( data )

class proj_RSM( unittest.TestCase ):

    def setUp(self):
    
        #  Make sure the GeoTiff example file exists
        self.tif_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                        '../data/Denver_SRTM_GL1.tif' ) )
        
        self.assertTrue( os.path.exists( self.tif_path ) )
        self.dem = DEM( self.tif_path )

        return super().setUp()
    
    def test_planet1(self):

        logger = logging.getLogger( 'test_RSM.planet1' )

        #  Make sure the RPC example file exists
        rpc_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                   '../data/20240717_171045_18_24af_1B_AnalyticMS_RPC.TXT' ) )
        
        self.assertTrue( os.path.exists( rpc_path ) )

        #  Load RPC Model
        rpc_model = load_rpc_file( rpc_path )

        #  Create GCPs
        img_size = rpc_model.image_size_pixels().astype('int32')

        #  Compute Elevation Range
        elevations = []
        

        # Iterate over pixels
        index = 0
        gcps = []
        for r in range( 0, img_size[1], 100 ):
            for c in range( 0, img_size[0], 100 ):

                # Pixel value
                pixel = np.array( [ c, r ], dtype = np.float64 )

                #  World coordinate
                lla = rpc_model.pixel_to_world( pixel,
                                                dem_model = self.dem,
                                                logger = logger )

                #  Add to gcp list
                gcp = GCP( id = index,
                           pixel = pixel,
                           coordinate = lla )
                print( gcp )
                index += 1

                gcps.append( gcp )


    