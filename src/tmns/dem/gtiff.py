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

#  Rasterio Libraries
import rasterio

#  Terminus Libraries
from tmns.image.utils import ( interp_index_nd,
                               interp_image )

class DEM_File:

    def __init__( self, pathname, auto_load = True ):
        '''
        Constructor given a path to a GeoTiff image.
        '''
        self.pathname = pathname
        self.dataset  = None

        if auto_load:
            self.load()


    def raster_size(self):
        return [self.dataset.width, 
                self.dataset.height]
    
    
    def min_corner_lla(self):
        return self.dataset.transform * (0,0)

    
    def max_corner_lla(self):
        rs = self.raster_size()
        return self.dataset.transform * ( rs[0]-1, rs[1]-1 )
    
    
    def load(self):

        if self.dataset != None:
            return False
        
        #  Open the file and keep the raster
        self.dataset = rasterio.open( self.pathname )
        self.raster  = self.dataset.read(1)

        return True

    def elevation_meters( self, lla ):

        #  Convert the LLA to pixel coordinates
        xform = self.dataset.transform
        pixel = ~xform * (lla[0], lla[1])

        indices = interp_index_nd( pixel )
        p00 = indices[0]
        p10 = indices[1]
        p01 = indices[2]
        p11 = indices[3]

        dx = pixel[0] - p00[0]
        dy = pixel[1] - p00[1]

        #  Use interpolation
        return interp_image( self.raster,
                              dx =  dx,  dy =  dy, 
                             p00 = p00, p10 = p10,
                             p01 = p01, p11 = p11 )
 
        