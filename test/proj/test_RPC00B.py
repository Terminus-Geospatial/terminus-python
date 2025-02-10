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
np.set_printoptions(precision=6, floatmode='fixed')

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
from tmns.proj import RPC00B
            

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

        self.flat_dem = Fixed_DEM( elevation_meters = 0 )

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

            lla_delta = np.abs(lla - ref_coord)
            self.assertLess( lla_delta[0], 1 )
            self.assertLess( lla_delta[1], 1 )
            self.assertLess( lla_delta[2], 1 )

            logger.debug( f'Ref Coord: {ref_coord}, Ref Pixel: {ref_pixel}, Pix Out: {pix}, LLA: {lla}, Pix Delta: {pix_delta}, LLA Delta: {lla_delta}' )
            

    def test_term( self ):
        '''
        Verify we can perform lookups on the RPC00B coefficients.
        '''

        #  I'm being lazy and just enumerating the whole thing
        terms = { 1: { 0: RPC00B.Term.SAMP_NUM_COEFF_1,  1: RPC00B.Term.SAMP_DEN_COEFF_1,
                       2: RPC00B.Term.LINE_NUM_COEFF_1,  3: RPC00B.Term.LINE_DEN_COEFF_1 },
                  2: { 0: RPC00B.Term.SAMP_NUM_COEFF_2,  1: RPC00B.Term.SAMP_DEN_COEFF_2,
                       2: RPC00B.Term.LINE_NUM_COEFF_2,  3: RPC00B.Term.LINE_DEN_COEFF_2 },
                  3: { 0: RPC00B.Term.SAMP_NUM_COEFF_3,  1: RPC00B.Term.SAMP_DEN_COEFF_3,
                       2: RPC00B.Term.LINE_NUM_COEFF_3,  3: RPC00B.Term.LINE_DEN_COEFF_3 },
                  4: { 0: RPC00B.Term.SAMP_NUM_COEFF_4,  1: RPC00B.Term.SAMP_DEN_COEFF_4,
                       2: RPC00B.Term.LINE_NUM_COEFF_4,  3: RPC00B.Term.LINE_DEN_COEFF_4 },
                  5: { 0: RPC00B.Term.SAMP_NUM_COEFF_5,  1: RPC00B.Term.SAMP_DEN_COEFF_5,
                       2: RPC00B.Term.LINE_NUM_COEFF_5,  3: RPC00B.Term.LINE_DEN_COEFF_5 },
                  6: { 0: RPC00B.Term.SAMP_NUM_COEFF_6,  1: RPC00B.Term.SAMP_DEN_COEFF_6,
                       2: RPC00B.Term.LINE_NUM_COEFF_6,  3: RPC00B.Term.LINE_DEN_COEFF_6 },
                  7: { 0: RPC00B.Term.SAMP_NUM_COEFF_7,  1: RPC00B.Term.SAMP_DEN_COEFF_7,
                       2: RPC00B.Term.LINE_NUM_COEFF_7,  3: RPC00B.Term.LINE_DEN_COEFF_7 },
                  8: { 0: RPC00B.Term.SAMP_NUM_COEFF_8,  1: RPC00B.Term.SAMP_DEN_COEFF_8,
                       2: RPC00B.Term.LINE_NUM_COEFF_8,  3: RPC00B.Term.LINE_DEN_COEFF_8 },
                  9: { 0: RPC00B.Term.SAMP_NUM_COEFF_9,  1: RPC00B.Term.SAMP_DEN_COEFF_9,
                       2: RPC00B.Term.LINE_NUM_COEFF_9,  3: RPC00B.Term.LINE_DEN_COEFF_9 },
                 10: { 0: RPC00B.Term.SAMP_NUM_COEFF_10, 1: RPC00B.Term.SAMP_DEN_COEFF_10,
                       2: RPC00B.Term.LINE_NUM_COEFF_10, 3: RPC00B.Term.LINE_DEN_COEFF_10 },
                 11: { 0: RPC00B.Term.SAMP_NUM_COEFF_11, 1: RPC00B.Term.SAMP_DEN_COEFF_11,
                       2: RPC00B.Term.LINE_NUM_COEFF_11, 3: RPC00B.Term.LINE_DEN_COEFF_11 },
                 12: { 0: RPC00B.Term.SAMP_NUM_COEFF_12, 1: RPC00B.Term.SAMP_DEN_COEFF_12,
                       2: RPC00B.Term.LINE_NUM_COEFF_12, 3: RPC00B.Term.LINE_DEN_COEFF_12 },
                 13: { 0: RPC00B.Term.SAMP_NUM_COEFF_13, 1: RPC00B.Term.SAMP_DEN_COEFF_13,
                       2: RPC00B.Term.LINE_NUM_COEFF_13, 3: RPC00B.Term.LINE_DEN_COEFF_13 },
                 14: { 0: RPC00B.Term.SAMP_NUM_COEFF_14, 1: RPC00B.Term.SAMP_DEN_COEFF_14,
                       2: RPC00B.Term.LINE_NUM_COEFF_14, 3: RPC00B.Term.LINE_DEN_COEFF_14 },
                 15: { 0: RPC00B.Term.SAMP_NUM_COEFF_15, 1: RPC00B.Term.SAMP_DEN_COEFF_15,
                       2: RPC00B.Term.LINE_NUM_COEFF_15, 3: RPC00B.Term.LINE_DEN_COEFF_15 },
                 16: { 0: RPC00B.Term.SAMP_NUM_COEFF_16, 1: RPC00B.Term.SAMP_DEN_COEFF_16,
                       2: RPC00B.Term.LINE_NUM_COEFF_16, 3: RPC00B.Term.LINE_DEN_COEFF_16 },
                 17: { 0: RPC00B.Term.SAMP_NUM_COEFF_17, 1: RPC00B.Term.SAMP_DEN_COEFF_17,
                       2: RPC00B.Term.LINE_NUM_COEFF_17, 3: RPC00B.Term.LINE_DEN_COEFF_17 },
                 18: { 0: RPC00B.Term.SAMP_NUM_COEFF_18, 1: RPC00B.Term.SAMP_DEN_COEFF_18,
                       2: RPC00B.Term.LINE_NUM_COEFF_18, 3: RPC00B.Term.LINE_DEN_COEFF_18 },
                 19: { 0: RPC00B.Term.SAMP_NUM_COEFF_19, 1: RPC00B.Term.SAMP_DEN_COEFF_19,
                       2: RPC00B.Term.LINE_NUM_COEFF_19, 3: RPC00B.Term.LINE_DEN_COEFF_19 },
                 20: { 0: RPC00B.Term.SAMP_NUM_COEFF_20, 1: RPC00B.Term.SAMP_DEN_COEFF_20,
                       2: RPC00B.Term.LINE_NUM_COEFF_20, 3: RPC00B.Term.LINE_DEN_COEFF_20 } }
        
        for x in range( 1, 21 ):

            self.assertEqual( RPC00B.Term.get_sample_num( x ), terms[x][0] )
            self.assertEqual( RPC00B.Term.get_sample_den( x ), terms[x][1] )
            self.assertEqual( RPC00B.Term.get_line_num( x ), terms[x][2] )
            self.assertEqual( RPC00B.Term.get_line_den( x ), terms[x][3] )



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

    def test_planet1_solver_dem(self):

        logger = logging.getLogger( 'test_RPC00B.test_planet1_solver_dem' )
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
        method = 'B'

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
                                                method = method )
                #print( 'Pixel: ', pixel, ', LLA: ', lla )

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

        #  Solve for model
        logger.debug( 'Solving RPC00B model' )
        new_model = RPC00B.RPC00B.solve( gcps,
                                         dem    = self.dem,
                                         method = method,
                                         logger = logger )
        logger.info( new_model )

        #  Verify the model
        res_points = []
        counter = 0
        for r in range( 0, img_size[1], 500 ):
            for c in range( 0, img_size[0], 500 ):

                # Pixel value
                pixel = np.array( [ c, r ], dtype = np.float64 )

                #  World coordinate
                lla = new_model.pixel_to_world( pixel,
                                                dem_model = self.dem,
                                                logger = logger,
                                                method = method )

                new_point = Placemark( name = f'GCP {index}',
                                       styleUrl='#mainStyle',
                                       geometry = Point( lat  = lla[1],
                                                         lon  = lla[0],
                                                         elev = lla[2] ) )
                res_points.append( new_point )

                self.assertLess( np.sum( np.abs( lla - gcps[counter].coordinate ) ), 0.0001 )

                counter += 1




        # Write Results to KML
        writer = Writer()

        style = Style( id = 'mainStyle',
                       label_style = Label_Style( color = 'ff0000ff' ) )

        folder = Folder( 'loaded_model',
                         features=kml_points )
        writer.add_node( folder )

        folder = Folder( 'solution',
                         features=res_points )
        writer.add_node( folder )

        writer.write( f'output_1_method_{method}' )
    
    def test_planet1_solver_flat(self):

        logger = logging.getLogger( 'test_RPC00B.test_planet1_solver_flat' )
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
        method = 'B'
        for r in range( 0, img_size[1], 500 ):
            for c in range( 0, img_size[0], 500 ):

                # Pixel value
                pixel = np.array( [ c, r ], dtype = np.float64 )

                #  World coordinate
                lla = rpc_model.pixel_to_world( pixel,
                                                dem_model = self.flat_dem,
                                                logger = logger,
                                                method = method )
                logger.debug( 'Pixel: ', pixel, ', LLA: ', lla )

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
        writer.write( f'output_2_method_{method}' )

        