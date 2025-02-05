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

#  Rasterio Libraries
import rasterio

#  Terminus Libraries
from tmns.image.utils import ( interp_index_nd,
                               interp_image )
from tmns.math.geometry import Rectangle

class DEM_File:

    def __init__( self,
                  pathname,
                  auto_load = True,
                  compute_stats = True,
                  band_id = 1 ):
        '''
        Constructor given a path to a GeoTiff image.
        '''
        self.pathname = pathname
        self.dataset  = None
        self.bbox = None

        if auto_load:
            self.load( band_id )

        if compute_stats:
            self.stats = self.compute_stats()
        else:
            self.stats = None

    def is_loaded(self):
        return self.dataset != None

    def raster_size(self):
        return [self.dataset.width, 
                self.dataset.height]
    
    
    def min_corner_lla(self):
        return self.dataset.transform * (0,0)

    
    def max_corner_lla(self):
        rs = self.raster_size()
        return self.dataset.transform * ( rs[0]-1, rs[1]-1 )
    
    def has_stats(self):
        return (self.stats != None)

    def compute_stats( self ):
        
        #  Get the band stats
        return self.dataset.stats( indexes = [self.band_id] )[0]
    
    def load(self, band_id = 1 ):

        if self.dataset != None:
            return False
        
        #  Open the file and keep the raster
        self.band_id = band_id
        self.dataset = rasterio.open( self.pathname )
        self.raster  = self.dataset.read( band_id )

        self.bbox = Rectangle.from_minmax( 0, 0,
                                           self.dataset.width - 1,
                                           self.dataset.height - 1 )

        return True

    def elevation_meters( self, lla, logger = None ):

        if logger is None:
            logger = logging.getLogger( 'gtiff.DEM_File.elevation_meters' )


        #  Convert the LLA to pixel coordinates
        xform = self.dataset.transform
        pixel = ~xform * (lla[0], lla[1])

        
        indices = interp_index_nd( pixel )
        #logger.debug( f'LLA: {lla}, Pixel: {pixel}, Indices: {indices}' )

        # Top left
        p00 = indices[0]
        if not self.bbox.inside( p00 ):
            return None
        
        p10 = indices[1]
        if not self.bbox.inside( p10 ):
            return None
        
        p01 = indices[2]
        if not self.bbox.inside( p01 ):
            return None
        
        p11 = indices[3]
        if not self.bbox.inside( p11 ):
            return None

        dx = pixel[0] - p00[0]
        dy = pixel[1] - p00[1]

        #  Use interpolation
        return interp_image( self.raster,
                              dx =  dx,  dy =  dy, 
                             p00 = p00, p10 = p10,
                             p01 = p01, p11 = p11 )
 
    
    def __str__(self):

        output  =  'DEM_File:\n'
        output += f'  - Pathname: {self.pathname}\n'
        output += f'  - Is Loaded: {self.is_loaded()}\n'

        if self.is_loaded():
            rs = self.raster_size()
            output += f'  - Band Size: {rs[0]} x {rs[1]} pixels\n'

            bl = self.min_corner_lla()
            tr = self.max_corner_lla()
            output += f'  - Band BL (LLA): {bl[0]} {bl[1]}\n'
            output += f'  - Band TR (LLA): {tr[0]} {tr[1]}\n'\
        
        return output
    