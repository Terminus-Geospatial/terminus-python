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

#  Python libraries
import logging
import os
import unittest

#  Numpy 
import numpy as np

#  Terminus Libraries
from tmns.core.types import GCP
from tmns.dem.fixed import Fixed_DEM
from tmns.dem.gtiff import DEM_File as DEM
from tmns.io.kml import ( Folder,
                          Label_Style, 
                          Placemark,
                          Point,
                          Style,
                          Writer ) 
import tmns.proj.RPC00B as RPC00B
            

class proj_RPC00B( unittest.TestCase ):

    @staticmethod
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


    def setUp(self):
    
        #  Make sure the GeoTiff example file exists
        self.tif_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                        '../data/Denver_SRTM_GL1.tif' ) )
        
        self.assertTrue( os.path.exists( self.tif_path ) )
        self.dem = DEM( self.tif_path )

        return super().setUp()
    
    def verify_projection( self, model, logger, gcps ):
        '''
        
        '''
        image_size = model.image_size_pixels()

        #  Iterate over each GCP
        for id in gcps.keys():
                
            ref_coord = gcps[id]['lla']
            ref_pixel = np.array( gcps[id]['pix'], dtype = np.float64 )

            #  Convert world back to pixel coordinate
            pix = model.world_to_pixel( ref_coord )
            lla = model.pixel_to_world( ref_pixel,
                                        dem_model = Fixed_DEM( 1625 ),
                                        logger = logger )

            pix_delta = np.abs(pix - ref_pixel)
            self.assertLess( pix_delta[0], 1 )
            self.assertLess( pix_delta[1], 1 )
            

    
    def test_planet1(self):

        logger = logging.getLogger( 'test_RPC00B.test_planet1' )

        #  Make sure the RPC example file exists
        rpc_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                   '../data/20240717_171045_18_24af_1B_AnalyticMS_RPC.TXT' ) )
        self.assertTrue( os.path.exists( rpc_path ) )

        #  Load RPC file
        rpc_model = proj_RPC00B.load_rpc_file( rpc_path )

        #  Check image size
        img_size = rpc_model.image_size_pixels()
        self.assertAlmostEqual( img_size[0], 8880, 1 )
        self.assertAlmostEqual( img_size[1], 5304, 1 )

        #  GCP Reference
        gcps = { 1: { 'lla': [ -104.967822, 39.735651, 1625], 'pix': [6328, 834] } }

        #  Verify some key values
        self.assertTrue( np.sum( rpc_model.center_pixel() - np.array([ 4440 , 2652 ])) < 0.1 )
        self.assertTrue( np.sum( rpc_model.center_coord() - np.array([ -105.0657, 39.69, 1970 ])) < 0.1 )
        
        #  Run Projection Test
        self.verify_projection( rpc_model, logger, gcps )

        logger.debug(rpc_model)

    def test_planet1_solver(self):

        logger = logging.getLogger( 'test_RPC00B.planet1' )
        logger.debug( 'Logger Initialized' )

        #  Make sure the RPC example file exists
        rpc_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                   '../data/20240717_171045_18_24af_1B_AnalyticMS_RPC.TXT' ) )
        self.assertTrue( os.path.exists( rpc_path ) )

        #  Load RPC Model
        rpc_model = proj_RPC00B.load_rpc_file( rpc_path )

        #  Create GCPs
        img_size = rpc_model.image_size_pixels().astype('int32')
        logger.debug( f'Image Size: {img_size[0]} x {img_size[1]} pixels' )

        #  Compute Elevation Range
        elevations = []

        # Iterate over pixels
        index = 0
        gcps = []
        kml_points = []
        for r in range( 0, img_size[1], 500 ):
            for c in range( 0, img_size[0], 500 ):

                # Pixel value
                pixel = np.array( [ c, r ], dtype = np.float64 )

                #  World coordinate
                lla = rpc_model.pixel_to_world( pixel,
                                                dem_model = self.dem,
                                                logger = logger,
                                                method = 'B' )
                print( 'Pixel: ', pixel, ', LLA: ', lla )

                #  Add to gcp list
                gcp = GCP( id = index,
                           pixel = pixel,
                           coordinate = lla )
                index += 1
                gcps.append( gcp )

                new_point = Placemark( name = f'GCP {index}',
                                       styleUrl='#mainStyle',
                                       geometry = Point( lat  = lla[1],
                                                         lon  = lla[0],
                                                         elev = lla[2] ) )
                kml_points.append( new_point )

        writer = Writer()

        style = Style( id = 'mainStyle',
                       label_style = Label_Style( color = 'ff0000ff' ) )

        folder = Folder( 'pixel2world',
                         features=kml_points )
        writer.add_node( folder )
        writer.write( 'output' )

        