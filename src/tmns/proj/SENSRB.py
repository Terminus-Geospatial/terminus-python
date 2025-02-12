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

    
    def get( self, key, type = None ):
        if type == None:
            return self.data[key]
        else:
            return type(self.data[key])

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
        return np.array( [ self.get(Term.SAMP_SCALE)*2,
                           self.get(Term.LINE_SCALE)*2],
                         dtype = np.float64 )

    
    def world_to_pixel( self,
                        coord, 
                        logger = None ):

        if logger == None:
            logger = logging.getLogger( 'SENSRB.world_to_pixel' )

        #  Get the number of terms
        n_terms = self.get( Term.TRANSFORM_PARAMS, int )

        raise NotImplementedError()


    
    def pixel_to_world( self,
                        pixel,
                        dem_model = None,
                        logger = None ):
        '''
        Convert a Pixel coordinate into World coordinates.
        '''

        if logger == None:
            logger = logging.getLogger( 'SENSRB.pixel_to_world' )

        
        

    
    def __str__(self):
        output  =  'SENSRB:\n'
        for k in self.data.keys():
            output += f'   - {k.name}:  {self.data[k]}\n'
        return output

    
    

    
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
        
        model.set( Term.SAMP_OFF,     center_pixel[0] )
        model.set( Term.SAMP_SCALE,   image_width / 2.0 )
        
        model.set( Term.LINE_OFF,     center_pixel[1] )
        model.set( Term.LINE_SCALE,   image_height / 2.0 )
        
        model.set( Term.LON_OFF,      center_lla[0] )
        model.set( Term.LON_SCALE,    max_delta_lla[0] )

        model.set( Term.LON_OFF,      center_lla[1] )
        model.set( Term.LAT_SCALE,    max_delta_lla[1] )

        model.set( Term.HEIGHT_OFF,   center_lla[2] )
        model.set( Term.HEIGHT_SCALE, max_delta_lla[2] )
        
        return model
    

    @staticmethod
    def solve( gcps: list[GCP],
               image_size = None,
               dem = None,
               method = 'B',
               logger = None,
               cheat_x_num = None,
               cheat_x_den = None,
               cheat_y_num = None,
               cheat_y_den = None ):
        '''
        Todo:  Use a better spread function to better model min/max. 
               How does our "spread" method work across the equator?

        The "Cheat" coefficient sets allow verification of this API if you want to hard-code the results
        '''
        
        if logger == None:
            logger = logging.getLogger( 'RPC00B.solve' )

        # Determine image bounds from points
        image_bbox = None
        if image_size is None:
            image_bbox = Rectangle( point = gcps[0].pixel )
        else:
            image_bbox = Rectangle( point = np.array( [0,0], dtype = np.float64 ), 
                                    size  = image_size )
            
        # Determine scene bounds from points
        lla_bbox = Rectangle( np.copy( gcps[0].coordinate ) )
        for gcp in gcps[1:]:
            image_bbox.add_point( gcp.pixel )
            lla_bbox.add_point( np.copy( gcp.coordinate ) )

        logger.debug( f'Image BBox: {image_bbox}' )
        logger.debug( f'Coord BBox: {lla_bbox}' )
        
        #  Compute the optimal center image point
        center_pix = None
        if image_size is None:
            center_pix = image_bbox.mean_point()
        else:
            center_pix = image_size * 0.5
        
        #  Solve the center coordinate point
        center_lla = lla_bbox.mean_point()

        # Storage after we normalize
        fx = np.zeros( len( gcps ) )
        fy = np.zeros( len( gcps ) )
        
        x = np.zeros( len( gcps ) )
        y = np.zeros( len( gcps ) )
        z = np.zeros( len( gcps ) )
        
        # We need to know the direction of flow for each axis so we can estimate 
        # Scale properly
        maxDeltaLLA = np.zeros( len( center_lla ) )
        minPnt_pair = None
        maxPnt_pair = None

        #------------------------------------------#
        #-        Normalize all coordinates       -#
        #------------------------------------------#
        for idx in range( len( gcps ) ):

            #  First point only
            if minPnt_pair == None:
                minPnt_pair = [ np.copy( gcps[idx].pixel ), np.copy( gcps[idx].coordinate ) ]
                maxPnt_pair = [ np.copy( gcps[idx].pixel ), np.copy( gcps[idx].coordinate ) ]
            
            #  Check if min point  (likely pixel (0,0) )
            if np.linalg.norm( minPnt_pair[0] ) > np.linalg.norm( gcps[idx].pixel ):
                minPnt_pair = [ np.copy( gcps[idx].pixel ), np.copy( gcps[idx].coordinate ) ]
                
            #  Check if max point  (likely pixel (cols,rows) )
            if np.linalg.norm( maxPnt_pair[0] ) < np.linalg.norm( gcps[idx].pixel ):
                maxPnt_pair = [ np.copy( gcps[idx].pixel ), np.copy( gcps[idx].coordinate ) ]

            #  Normalize geographic coordinates
            d_lla = gcps[idx].coordinate - center_lla
            
            x[idx] = d_lla[0]
            y[idx] = d_lla[1]
            z[idx] = d_lla[2]

            #  Normalize pixel coordinates
            fx[idx] = (gcps[idx].pixel[0] - center_pix[0]) / (image_bbox.size[0] / 2.0)
            fy[idx] = (gcps[idx].pixel[1] - center_pix[1]) / (image_bbox.size[1] / 2.0)

            #  Update the delta for the geographic coordinates
            if abs( d_lla[0] ) > maxDeltaLLA[0]:
                maxDeltaLLA[0] = abs( d_lla[0] )

            if abs( d_lla[1] ) > maxDeltaLLA[1]:
                maxDeltaLLA[1] = abs( d_lla[1] )
            
            if abs( d_lla[2] ) > maxDeltaLLA[2]:
                maxDeltaLLA[2] = abs( gcps[idx].coordinate[2] )
        
        #  Solve direction
        pnt_dlt = (maxPnt_pair[1] - minPnt_pair[1])
        pnt_mag = np.ones( len( pnt_dlt ) )
        for idx in range( len( pnt_dlt ) ):
            if pnt_dlt[idx] < 0:
                maxDeltaLLA[idx] *= -1
        logger.debug( f'Point Delta: {pnt_dlt}, Mag: {pnt_mag}' )

        for idx in range( len( gcps ) ):

            x[idx] /= maxDeltaLLA[0]
            y[idx] /= maxDeltaLLA[1]
            z[idx] /= maxDeltaLLA[2]

        #--------------------------------------------------------------------------#
        #-        Create the new model and solve for each pixel row/col set       -#
        #--------------------------------------------------------------------------#
        rpc_model = RPC00B()
        rpc_model.set( Term.SAMP_OFF,   center_pix[0] )
        rpc_model.set( Term.LINE_OFF,   center_pix[1] )

        rpc_model.set( Term.SAMP_SCALE, image_bbox.size[0] / 2 )
        rpc_model.set( Term.LINE_SCALE, image_bbox.size[1] / 2 )
        
        rpc_model.set( Term.LON_SCALE,    maxDeltaLLA[0] )
        rpc_model.set( Term.LAT_SCALE,    maxDeltaLLA[1] )
        rpc_model.set( Term.HEIGHT_SCALE, maxDeltaLLA[2] )

        rpc_model.set( Term.LON_OFF,    center_lla[0] )
        rpc_model.set( Term.LAT_OFF,    center_lla[1] )
        rpc_model.set( Term.HEIGHT_OFF, center_lla[2] )

        if dem == None:
            rpc_model.set( Term.HEIGHT_SCALE, 0.0 )


        #  Perform Least-Squares Fit
        samp_num_coeffs = cheat_x_num
        samp_den_coeffs = cheat_x_den
        line_num_coeffs = cheat_y_num
        line_den_coeffs = cheat_y_den

        #  Update the coefficients
        for x in range( 20 ):
            rpc_model.set( Term.get_sample_num( x + 1 ), samp_num_coeffs[x] )
            rpc_model.set( Term.get_sample_den( x + 1 ), samp_den_coeffs[x] )
            rpc_model.set( Term.get_line_num( x + 1 ),   line_num_coeffs[x] )
            rpc_model.set( Term.get_line_den( x + 1 ),   line_den_coeffs[x] )


        #  Compute RMSE for errors
        sumSquareError = 0
        maxResidual = 0
        for gcp in gcps:

            npix = rpc_model.world_to_pixel( gcp.coordinate,
                                             skip_elevation = False,
                                             logger = logger,
                                             method = method )
            
            delta = npix - gcp.pixel
            val = math.sqrt( np.dot( delta, delta ) )
            if val > maxResidual:
                maxResidual = val
            sumSquareError += val
        
        rms_error = math.sqrt( sumSquareError / len(gcps) )

        rpc_model.set( Term.MAX_ERROR, maxResidual )
        rpc_model.set( Term.RMS_ERROR, rms_error )

        return rpc_model

    @staticmethod
    def fitness( self, C, model: RPC00B, fx, fy, x, y, z, method ):

        #  Get the core variables we are fixed on
        lon_off = model.get( Term.LON_OFF )
        lat_off = model.get( Term.LAT_OFF )
        hgt_off = model.get( Term.HEIGHT_OFF )

        lon_scale = model.get( Term.LON_SCALE )
        lat_scale = model.get( Term.LAT_SCALE )
        hgt_scale = model.get( Term.HEIGHT_SCALE )

        #  Make sure al inputs are the same size
        assert( len(fx) == len(fy) )
        assert( len(fx) == len(x) )
        assert( len(fx) == len(y) )
        assert( len(fx) == len(z) )

        #  Iterate over each GCP
        for idx in range( len( fx ) ):

            plh_vec = self.get_plh_vector( P = y[idx],
                                           L = x[idx],
                                           H = z[idx],
                                           method = method )

            r_n_n = np.dot( self.get_line_numerator_coefficients(),     plh_vec )
            r_n_d = np.dot( self.get_line_denominator_coefficients(),   plh_vec )
            c_n_n = np.dot( self.get_sample_numerator_coefficients(),   plh_vec )
            c_n_d = np.dot( self.get_sample_denominator_coefficients(), plh_vec )

            gcp_lla = np.array( [ (c_n_n/c_n_d) * self.get(Term.SAMP_SCALE) + self.get(Term.SAMP_OFF),
                                  (r_n_n/r_n_d) * self.get(Term.LINE_SCALE) + self.get(Term.LINE_OFF) ],
                                dtype = np.float64 )
            