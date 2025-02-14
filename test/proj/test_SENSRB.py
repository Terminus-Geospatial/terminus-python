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

#  Pandas
import pandas as pd

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
from tmns.proj import SENSRB
            

class proj_SENSRB( unittest.TestCase ):

    def load_usgs_gcps( self ):

        logger = logging.getLogger( 'test_SENSRB.load_usgs_gcps' )

        #  Make sure the RPC example file exists
        gcp_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                   '../data/USGS_GCPs.csv' ) )
        self.assertTrue( os.path.exists( gcp_path ) )

        #  Load Ground Control Points
        gcp_df = pd.read_csv( gcp_path )
        
        gcps = []
        kml_points = []

        # Iterate over each row
        counter = 0
        for row in gcp_df.itertuples():

            pixel = np.array( [ float(row.PX), float( row.PY) ],
                              dtype = np.float64 )

            coord = np.array( [ float(row.Longitude),
                                float(row.Latitude),
                                float(row.Elevation) ],
                              dtype = np.float64 )

            gcps.append( GCP( id = counter,
                              pixel = pixel,
                              coordinate = coord ) )
            
            new_point = Placemark( name = f'Actual GCP {counter}',
                                       styleUrl='#mainStyle',
                                       geometry = Point( lat  = coord[1],
                                                         lon  = coord[0],
                                                         elev = coord[2] ) )
            kml_points.append( new_point )
            counter += 1

        return gcps, kml_points

    def setUp(self):
    
        #  Make sure the GeoTiff example file exists
        self.tif_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                        '../data/Denver_SRTM_GL1.tif' ) )
        
        self.assertTrue( os.path.exists( self.tif_path ) )
        self.dem = DEM( self.tif_path )

        self.flat_dem = Fixed_DEM( elevation_meters = 0 )

        return super().setUp()
    
    def verify_projection( self, model, logger, gcps ):
        

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
            
    
    def test_usgs_model(self):

        logger = logging.getLogger( 'test_SENSRB.load_usgs_gcps' )

        gcps, kml_points = self.load_usgs_gcps()
        
        image_size = np.array( [2829.0, 2964.0], dtype = np.float64 )

        #  Create model from GCPs
        new_model = SENSRB.SENSRB.solve( gcps       = gcps,
                                         image_size = image_size )
        logger.debug( new_model )
        
        #  Verify the model
        res_points = []
        counter = 0
        for r in range( 0, int(image_size[1]), 500 ):
            for c in range( 0, int(image_size[0]), 500 ):

                # Pixel value
                pixel = np.array( [ c, r ], dtype = np.float64 )

                #  World coordinate
                lla = new_model.pixel_to_world( pixel,
                                                dem_model = self.dem,
                                                logger = logger )
                new_point = Placemark( name = f'Computed GCP {counter}',
                                       styleUrl='#mainStyle',
                                       geometry = Point( lat  = lla[1],
                                                         lon  = lla[0],
                                                         elev = lla[2] ) )
                res_points.append( new_point )

                #self.assertLess( np.sum( np.abs( lla - gcps[counter].coordinate ) ), 1 )

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

        writer.write( f'SENSRB_output_1' )
        




    