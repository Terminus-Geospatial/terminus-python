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
from collections import namedtuple

#  Rasterio API
import rasterio

class Fixed_DEM:

    def __init__( self,
                  elevation_meters = 0 ):
        '''
        Create a terrain model with a single fixed value
        '''
        self._elevation_meters = elevation_meters

        self.stats = rasterio.Statistics( self._elevation_meters,
                                          self._elevation_meters,
                                          self._elevation_meters,
                                          0 )

    def is_loaded(self):
        return True

    def raster_size(self):
        return None
    
    
    def min_corner_lla(self):
        return None

    
    def max_corner_lla(self):
        return None
    
    def has_stats(self):
        return True

    def compute_stats( self ):
        return self.stats
    

    def elevation_meters( self, lla = None ):
        return self._elevation_meters
 
    
    def __str__(self):

        output  =  'Fixed_DEM:\n'
        output += f'  - Elevation Meters: {self._elevation_meters}\n'
        output += f'  - Stats: {self.stats}'
        
        return output
    