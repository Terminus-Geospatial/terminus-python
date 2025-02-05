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

class BaseTransformer:

    def __init__( self ):
        pass

    def center_pixel(self):
        raise Exception('Not implemented for base class')

    def center_coordinate(self):
        raise Exception('Not implemented for base class')

    def world_to_pixel( self, coord, logger = None ):
        raise Exception('Not implemented for base class')

    def pixel_to_world( self, pixel, dem, logger = None ):
        raise Exception('Not implemented for base class')

    def gsd(self):
        raise Exception( 'Not implemented for base class' ) 