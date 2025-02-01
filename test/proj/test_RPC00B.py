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
import tmns.proj.RPC00B as RPC00B

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
            

class proj_RPC00B( unittest.TestCase ):

    def verify_projection( self, model, logger, gcps ):
        '''
        Project 
        '''
        image_size = model.image_size_pixels()

        #  Iterate over each GCP
        for id in gcps.keys():
                
            ref_coord = gcps[id]['lla']
            ref_pixel = np.array( gcps[id]['pix'], dtype = np.float64 )

            #  Convert world back to pixel coordinate
            pix = model.world_to_pixel( ref_coord )
            lla = model.pixel_to_world( ref_pixel,
                                        ellipsoid_height = 1625,
                                        logger = logger )

            pix_delta = np.sum( pix - ref_pixel )
                
            print( f'Ground: {ref_coord}, Pix Out: {pix}, Pix Delta: {pix_delta}, LLA: {lla}' )

    
    def test_planet1(self):

        logger = logging.getLogger( 'test_RPC00B.planet1' )

        #  Make sure the RPC example file exists
        rpc_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                   '../data/20240717_171045_18_24af_1B_AnalyticMS_RPC.TXT' ) )
        
        self.assertTrue( os.path.exists( rpc_path ) )

        #  Load RPC file
        model = load_rpc_file( rpc_path )

        #  GCP Reference
        gcps = { 1: { 'lla': [ -104.967822, 39.735651, 1625], 'pix': [6328, 834] } }

        #  Verify some key values
        self.assertTrue( np.sum( model.center_pixel() - np.array([ 4440 , 2652 ])) < 0.1 )
        self.assertTrue( np.sum( model.center_coord() - np.array([ -105.0657, 39.69, 1970 ])) < 0.1 )
        
        #  Run Projection Test
        self.verify_projection( model, logger, gcps )

        print(model)

        