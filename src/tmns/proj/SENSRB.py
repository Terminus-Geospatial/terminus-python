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
from enum import Enum
import logging
import math

# Numpy Libraries
import numpy as np


#  Project Libraries
from tmns.core.types       import GCP
from tmns.math.geometry    import Rectangle
from tmns.math.solver      import pseudoinverse
from tmns.proj.transformer import BaseTransformer

class Term(Enum):
    
    GENERAL_DATA               =   1
    SENSOR                     =   2
    SENSOR_URI                 =   3
    PLATFORM                   =   4
    PLATFORM_URI               =   5
    OPERATION_DOMAIN           =   6
    CONTENT_LEVEL              =   7
    GEODETIC_SYSTEM            =   8
    GEODETIC_TYPE              =   9
    ELEVATION_DATUM            =  10
    LENGTH_UNIT                =  11
    ANGULAR_UNIT               =  12
    START_DATE                 =  13
    START_TIME                 =  14
    END_DATE                   =  15
    END_TIME                   =  16
    GENERATION_COUNT           =  17
    GENERATION_DATE            =  18
    GENERATION_TIME            =  19
    SENSOR_ARRAY_DATA          =  20
    DETECTION                  =  21
    ROW_DETECTORS              =  22
    COLUMN_DETECTORS           =  23
    ROW_METRIC                 =  24
    COLUMN_METRIC              =  25
    FOCAL_LENGTH               =  26
    ROW_FOV                    =  27
    COLUMN_FOV                 =  28
    CALIBRATED                 =  29
    SENSOR_CALIBRATED_DATA     =  30
    CALIBRATION_UNIT           =  31
    PRINCIPAL_POINT_OFFSET_X   =  32
    PRINCIPAL_POINT_OFFSET_Y   =  32
    RADIAL_DISTORT_1           =  33
    RADIAL_DISTORT_2           =  34
    RADIAL_DISTORT_3           =  35
    RADIAL_DISTORT_LIMIT       =  36
    DECENT_DISTORT_1           =  37
    DECENT_DISTORT_2           =  38
    AFFINITY_DISTORT_1         =  39
    AFFINITY_DISTORT_2         =  40
    CALIBRATION_DATE           =  41
    IMAGE_FORMATION_DATA       =  42
    METHOD                     =  43
    MODE                       =  44
    ROW_COUNT                  =  45
    COLUMN_COUNT               =  46
    ROW_SET                    =  47
    COLUMN_SET                 =  48
    ROW_RATE                   =  49
    COLUMN_RATE                =  50
    FIRST_PIXEL_ROW            =  51
    FIRST_PIXEL_COLUMN         =  52
    TRANSFORM_PARAMS           =  53
    TRANSFORM_PARAM_1          =  54
    TRANSFORM_PARAM_2          =  55
    TRANSFORM_PARAM_3          =  56
    TRANSFORM_PARAM_4          =  57
    TRANSFORM_PARAM_5          =  58
    TRANSFORM_PARAM_6          =  59
    TRANSFORM_PARAM_7          =  60
    TRANSFORM_PARAM_8          =  61
    REFERENCE_TIME             =  62
    REFERENCE_ROW              =  63
    REFERENCE_COLUMN           =  64
    LATITUDE_OR_X              =  65
    LONGITUDE_OR_Y             =  66
    ALTITUDE_OR_Z              =  67
    SENSOR_X_OFFSET            =  68
    SENSOR_Y_OFFSET            =  69
    SENSOR_Z_OFFSET            =  70
    ATTITUDE_EULER_ANGLES      =  71
    SENSOR_ANGLE_MODEL         =  72
    SENSOR_ANGLE_1             =  73
    SENSOR_ANGLE_2             =  74
    SENSOR_ANGLE_3             =  75
    PLATFORM_RELATIVE          =  76
    PLATFORM_HEADING           =  77
    PLATFORM_PITCH             =  78
    PLATFORM_ROLL              =  79
    ATTITUDE_UNIT_VECTORS      =  80
    ICX_NORTH_OR_X             =  81
    ICX_EAST_OR_Y              =  82
    ICX_DOWN_OR_Z              =  83
    ICY_NORTH_OR_X             =  84
    ICY_EAST_OR_Y              =  85
    ICY_DOWN_OR_Z              =  86
    ICZ_NORTH_OR_X             =  87
    ICZ_EAST_OR_Y              =  88
    ICZ_DOWN_OR_Z              =  89
    ATTITUDE_QUATERNION        =  90
    ATTITUDE_Q1                =  91
    ATTITUDE_Q2                =  92
    ATTITUDE_Q3                =  93
    ATTITUDE_Q4                =  94
    SENSOR_VELOCITY_DATA       =  95
    VELOCITY_NORTH_OR_X        =  96
    VELOCITY_EAST_OR_Y         =  97
    VELOCITY_DOWN_OR_Z         =  98
    POINT_SET_DATA             =  99
    POINT_SET_TYPE             = 100
    POINT_COUNT                = 101
    P_ROW                      = 102
    P_COLUMN                   = 103
    P_LATITUDE                 = 104
    P_LONGITUDE                = 105
    P_ELEVATION                = 106
    P_RANGE                    = 107
    TIME_STAMPED_DATA_SETS     = 108
    TIME_STAMP_TYPE            = 109
    TIME_STAMP_COUNT           = 110
    TIME_STAMP_TIME            = 111
    TIME_STAMP_VALUE           = 112
    PIXEL_REFERENCED_DATA_SETS = 113
    PIXEL_REFERENCE_TYPE       = 114
    PIXEL_REFERENCE_COUNT      = 115
    PIXEL_REFERENCE_ROW        = 116
    PIXEL_REFERENCE_COLUMN     = 117
    PIXEL_REFERENCE_VALUE      = 118
    UNCERTAINTY_DATA           = 119
    UNCERTAINTY_FIRST_TYPE     = 120
    UNCERTAINTY_SECOND_TYPE    = 121
    UNCERTAINTY_VALUE          = 122
    ADDITIONAL_PARAMTER_DATA   = 123
    PARAMETER_NAME             = 124
    PARAMETER_SIZE             = 125
    PARAMETER_COUNT            = 126
    PARAMETER_VLAUE            = 127


    @staticmethod
    def from_str( key ):
        return Term[key]
    

class SENSRB(BaseTransformer):

    def __init__(self, data = None ):
        '''
        Constructor for Sensor Model, Rev B object
        '''
        self.data = SENSRB.defaults()
        if not data is None:

            for k in data.keys():
                self.data[k] = data[k]

    
    def get( self, key ):
        return self.data[key]

    def set( self, key, value ):
        self.data[key] = value
    
    def center_pixel(self):
        '''
        Assuming this is the center of the image size
        '''
        return np.array( [ self.get(Term.COLUMN_COUNT) / 2.0,
                           self.get(Term.ROW_COUNT) / 2.0 ],
                         dtype = np.float64 )

    
    def center_coord(self):
        raise NotImplementedError()

    
    def image_size_pixels(self):
        return np.array( [ self.get(Term.COLUMN_COUNT),
                           self.get(Term.ROW_COUNT) ],
                         dtype = np.float64 )

    
    def world_to_pixel( self,
                        coord, 
                        logger = None ):

        if logger == None:
            logger = logging.getLogger( 'SENSRB.world_to_pixel' )

        # Get the terms
        pix = None
        terms = self.get_transform_array()

        if len(terms) == 2:
            raise NotImplementedError()
        elif len(terms) == 4:
            raise NotImplementedError()
        elif len(terms) == 5:
            raise NotImplementedError()
        
        elif len(terms) == 6:
            
            A = np.array( [ [ terms[0], terms[1], terms[2]],
                            [ terms[3], terms[4], terms[5]],
                            [     0   ,     0   ,     1   ] ],
                            dtype = np.float64 )

            A_inv = np.linalg.pinv( A )
            pix = A_inv @ np.array( [ [ coord[0] ],
                                      [ coord[1] ],
                                      [ coord[2] ] ])

        elif len(terms) == 8:
            raise NotImplementedError()
        else:
            raise Exception( f'Unsupported number of terms: {len(terms)}' )

        return pix
    
    def pixel_to_world( self,
                        pixel,
                        dem_model = None,
                        logger = None ):
        '''
        Convert a Pixel coordinate into World coordinates.
        '''

        if logger == None:
            logger = logging.getLogger( 'SENSRB.pixel_to_world' )

        xform = self.get_transform_array()
        coord = None

        if len(xform) == 0:
            return pixel
        
        elif len(xform) == 2:
            coord = pixel + xform
        
        elif len(xform) == 4:
            s = xform[0]
            a = xform[1]
            dx = xform[2]
            dy = xform[3]

            M = np.array( [[ s * math.cos(a), s * math.sin(a)],
                           [-s * math.sin(a), s * math.cos(a)]],
                           dtype = np.float64 )
            coord = M @ pixel.resize(1,2) + np.array( [[dx],[dy]])
        
        elif len(xform) == 5:
            s1 = xform[0]
            s2 = xform[1]
            a  = xform[2]
            dx = xform[3]
            dy = xform[4]

            S = np.array( [[s1, 0], [0, s2]] )
            M = np.array( [[ math.cos(a), math.sin(a)],
                           [-math.sin(a), math.cos(a)]],
                           dtype = np.float64 )
            coord = S @ M @ pixel.reshape(2,1) + np.array( [[dx],[dy]])

        elif len(xform) == 6:
            S = np.array( [[ xform[0], xform[1] ], 
                           [ xform[2], xform[3] ]] )
        
            print(S)
            print(pixel.reshape(2,1))
            coord = S @ pixel.reshape(1,2) + np.array( [[xform[4]],[xform[5]]])
            print( f'coord: {coord}' )
        
        elif len(xform) == 8:
            
            Mx = np.array( [[ xform[0], xform[1], xform[2] ], 
                            [ xform[6], xform[7], 1 ]] )
            My = np.array( [[ xform[3], xform[4], xform[5] ], 
                            [ xform[6], xform[7], 1 ]] )
        
            raise NotImplementedError()
            return S @ pixel.resize(1,2) + np.array( [[dx],[dy]])

        else:
            raise Exception( f'Unsupported Number of Coefficients: {len(xform)}' )

        coord = coord.resize( 3, 1 )
        if dem_model != None:
            coord[2] = dem_model.elevation_meters( coord )
        return coord
    

    def __str__(self):
        output  =  'SENSRB:\n'
        for k in self.data.keys():
            output += f'   - {k.name}:  {self.data[k]}\n'
        return output
    
    def get_transform_array(self):
        '''
        Return the transform as an array
        '''
        n_params = self.get( Term.TRANSFORM_PARAMS )
        params = []
        for x in range( n_params ):
            params.append( self.get( Term(Term.TRANSFORM_PARAM_1.value + x) ) )
        return params
    
    @staticmethod
    def defaults():
        data = {  }

        return data   

    
    @staticmethod
    def from_components( center_pixel,
                         center_lla,
                         image_width,
                         image_height,
                         max_delta_lla ):

        model = SENSRB()
        

        
        return model
    

    @staticmethod
    def solve( gcps: list[GCP],
               image_size = None,
               logger = None ):
        
        if logger == None:
            logger = logging.getLogger( 'SENSRB.solve' )

        #  Primary Matrix
        A_pnts = []
        Bx_pnts = []
        By_pnts = []
        for gcp in gcps:
            l = list(gcp.pixel)
            l.append( 1 )
            A_pnts.append( l )

            Bx_pnts.append( [gcp.coordinate[0]] )
            By_pnts.append( [gcp.coordinate[1]] )

        A = np.array( A_pnts )
        A_inv = np.linalg.pinv( A )
        
        #  Result Matrices
        B_x = np.array( Bx_pnts )
        B_y = np.array( By_pnts )
        
        coeffA = A_inv @ B_x
        coeffB = A_inv @ B_y
        
        new_model = SENSRB()

        new_model.set( Term.COLUMN_COUNT, image_size[0] )
        new_model.set( Term.ROW_COUNT, image_size[1] )

        new_model.set( Term.TRANSFORM_PARAMS, len(coeffA) + len(coeffB) )
        new_model.set( Term.TRANSFORM_PARAM_1, coeffA[0] )
        new_model.set( Term.TRANSFORM_PARAM_2, coeffA[1] )
        new_model.set( Term.TRANSFORM_PARAM_3, coeffB[0] )
        new_model.set( Term.TRANSFORM_PARAM_4, coeffB[1] )
        new_model.set( Term.TRANSFORM_PARAM_5, coeffA[2] )
        new_model.set( Term.TRANSFORM_PARAM_6, coeffB[2] )

        return new_model