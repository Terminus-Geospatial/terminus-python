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
import math

# Numpy Libraries
import numpy as np

#  Terminus Libraries
from tmns.nitf.RSMIDA import ( RSMIDA,
                               Tag as RSMIDA_TAG )
from tmns.nitf.RSMPCA import ( RSMPCA,
                               Tag as RSMPCA_TAG )
from tmns.nitf.RSMPIA import ( RSMPIA,
                               Tag as RSMPIA_TAG )

class RSM:

    def __init__(self):

        self.rsmida = RSMIDA()
        self.rsmpca = [RSMPCA()]
        

    def get( self, key ):
        return self.data[key]

    def set( self, key, value ):
        self.data[key] = value

    def center_pixel(self):
        raise NotImplementedError()

    
    def center_coord(self):
        raise NotImplementedError()

    
    def image_size_pixels(self):
        raise NotImplementedError()

    
    def world_to_pixel( self, 
                        coord, 
                        skip_elevation = False, 
                        logger = None ):

        if logger == None:
            logger = logging.getLogger( 'RSM.world_to_pixel' )

        #  Adjust longitude depending on Ground Domain Form
        grndd = self.rsmida.get(RSMIDA_TAG.GRNDD)

        x = math.radians( coord[0] )
        if grndd == 'H' and coord[0] < 0:
            x += 2 * math.pi

        y = math.radians( coord[1] )

        #  Figure out height
        z = 0
        if skip_elevation == False:
            z = coord[2]
        
        #  Fetch appropriate PCA Index
        pca_idx = self.get_pca_index( lla_rad = np.array( [ x, y, z ] ) )

        #  Normalize the coordinate
        self.rsmpca[pca_idx].normalize_lla( x, y, z, skip_elevation )

        #  Apply the polynomials
    
    def pixel_to_world( self,
                        pixel,
                        dem_model = None,
                        logger = None ):

        if logger == None:
            logger = logging.getLogger( 'RSM.pixel_to_world' )

        pass

    
    def get_pca_index( self,
                       pixel = None,
                       lla_rad = None,
                       shift_point = True ):

        if pixel == None:
            pixel = self.rsmpia.low_order_polynomial( lla_rad )
        
        #   Great note in OSSIM:
        #     RSM (0,0) is upper left corner of pixel(0,0). OSSIM (0,0) is
        #     center of the pixel; hence, the shift 0.5 if coming from ossim.
        shift = 0.5 if shift_point else 0.0
        
        #  Get some required keys
        minr  = self.rsmida.get( RSMIDA_TAG.MINR )
        minc  = self.rsmida.get( RSMIDA_TAG.MINC )
        rssiz = self.rsmpia.get( RSMPIA_TAG.RSSIZ )
        cssiz = self.rsmpia.get( RSMPIA_TAG.CSSIZ )
        rnis  = self.rsmpia.get( RSMPIA_TAG.RNIS )
        cnis  = self.rsmpia.get( RSMPIA_TAG.CNIS )

        #  Row Section Number
        rsn = math.floor( pixel[1] + shift - minr ) / rssiz
        rsn = 0 if rsn < 0 else rsn
        rsn = rnis - 1 if rsn > rnis - 1 else rsn
        
        #  Column Section Number
        csn = math.floor( pixel[0] + shift - minc ) / cssiz
        csn = 0 if csn < 0 else csn
        csn = cnis - 1 if csn > cnis - 1 else csn

        #  Apply to return proper offset
        return int( rsn * cnis + csn )


    def __str__(self):
        output  =  'RSM:\n'
        for k in self.data.keys():
            output += f'   - {k.name}:  {self.data[k]}\n'
        return output
    

    @staticmethod
    def from_components( center_pixel,
                         center_lla,
                         image_width,
                         image_height,
                         rsmida_grndd = 'G' ):
        
        #  Construct the sensor model
        rsm = RSM()

        #  Update a few key flags prior to doing math
        rsm.set( RSMIDA_TAG.GRNDD, rsmida_grndd )
        
        return rsm
        

    