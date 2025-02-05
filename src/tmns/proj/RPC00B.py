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

#  Numpy
import numpy as np

#  Project Libraries
from tmns.proj.transformer import BaseTransformer

class Term(Enum):
    SAMP_OFF           =  1
    SAMP_SCALE         =  2
    LINE_OFF           =  3
    LINE_SCALE         =  4
    LON_OFF            =  5
    LON_SCALE          =  6
    LAT_OFF            =  7
    LAT_SCALE          =  8
    HEIGHT_OFF         =  9
    HEIGHT_SCALE       = 10
    LINE_NUM_COEFF_1   = 11
    LINE_NUM_COEFF_2   = 12
    LINE_NUM_COEFF_3   = 13
    LINE_NUM_COEFF_4   = 14
    LINE_NUM_COEFF_5   = 15
    LINE_NUM_COEFF_6   = 16
    LINE_NUM_COEFF_7   = 17
    LINE_NUM_COEFF_8   = 18
    LINE_NUM_COEFF_9   = 19
    LINE_NUM_COEFF_10  = 20
    LINE_NUM_COEFF_11  = 21
    LINE_NUM_COEFF_12  = 22
    LINE_NUM_COEFF_13  = 23
    LINE_NUM_COEFF_14  = 24
    LINE_NUM_COEFF_15  = 25
    LINE_NUM_COEFF_16  = 26
    LINE_NUM_COEFF_17  = 27
    LINE_NUM_COEFF_18  = 28
    LINE_NUM_COEFF_19  = 29
    LINE_NUM_COEFF_20  = 30
    LINE_DEN_COEFF_1   = 31
    LINE_DEN_COEFF_2   = 32
    LINE_DEN_COEFF_3   = 33
    LINE_DEN_COEFF_4   = 34
    LINE_DEN_COEFF_5   = 35
    LINE_DEN_COEFF_6   = 36
    LINE_DEN_COEFF_7   = 37
    LINE_DEN_COEFF_8   = 38
    LINE_DEN_COEFF_9   = 39
    LINE_DEN_COEFF_10  = 40
    LINE_DEN_COEFF_11  = 41
    LINE_DEN_COEFF_12  = 42
    LINE_DEN_COEFF_13  = 43
    LINE_DEN_COEFF_14  = 44
    LINE_DEN_COEFF_15  = 45
    LINE_DEN_COEFF_16  = 46
    LINE_DEN_COEFF_17  = 47
    LINE_DEN_COEFF_18  = 48
    LINE_DEN_COEFF_19  = 49
    LINE_DEN_COEFF_20  = 50
    SAMP_NUM_COEFF_1   = 51
    SAMP_NUM_COEFF_2   = 52
    SAMP_NUM_COEFF_3   = 53
    SAMP_NUM_COEFF_4   = 54
    SAMP_NUM_COEFF_5   = 55
    SAMP_NUM_COEFF_6   = 56
    SAMP_NUM_COEFF_7   = 57
    SAMP_NUM_COEFF_8   = 58
    SAMP_NUM_COEFF_9   = 59
    SAMP_NUM_COEFF_10  = 60
    SAMP_NUM_COEFF_11  = 61
    SAMP_NUM_COEFF_12  = 62
    SAMP_NUM_COEFF_13  = 63
    SAMP_NUM_COEFF_14  = 64
    SAMP_NUM_COEFF_15  = 65
    SAMP_NUM_COEFF_16  = 66
    SAMP_NUM_COEFF_17  = 67
    SAMP_NUM_COEFF_18  = 68
    SAMP_NUM_COEFF_19  = 69
    SAMP_NUM_COEFF_20  = 70
    SAMP_DEN_COEFF_1   = 71
    SAMP_DEN_COEFF_2   = 72
    SAMP_DEN_COEFF_3   = 73
    SAMP_DEN_COEFF_4   = 74
    SAMP_DEN_COEFF_5   = 75
    SAMP_DEN_COEFF_6   = 76
    SAMP_DEN_COEFF_7   = 77
    SAMP_DEN_COEFF_8   = 78
    SAMP_DEN_COEFF_9   = 79
    SAMP_DEN_COEFF_10  = 80
    SAMP_DEN_COEFF_11  = 81
    SAMP_DEN_COEFF_12  = 82
    SAMP_DEN_COEFF_13  = 83
    SAMP_DEN_COEFF_14  = 84
    SAMP_DEN_COEFF_15  = 85
    SAMP_DEN_COEFF_16  = 86
    SAMP_DEN_COEFF_17  = 87
    SAMP_DEN_COEFF_18  = 88
    SAMP_DEN_COEFF_19  = 89
    SAMP_DEN_COEFF_20  = 90

    @staticmethod
    def from_str( key ):
        return Term[key]
        
    
    

class RPC00B(BaseTransformer):

    def __init__(self, data = None ):
        '''
        Constructor for RPC object
        '''
        self.data = RPC00B.defaults()
        if not data is None:

            for k in data.keys():
                self.data[k] = data[k]

    
    def get( self, key ):
        return self.data[key]

    def set( self, key, value ):
        self.data[key] = value
    
    def center_pixel(self):
        return np.array( [ self.get(Term.SAMP_OFF), self.get(Term.LINE_OFF )], dtype = np.float64 )

    
    def center_coord(self):
        return np.array( [ self.get(Term.LON_OFF), self.get(Term.LAT_OFF), self.get(Term.HEIGHT_OFF)], dtype = np.float64 )

    
    def image_size_pixels(self):
        return np.array( [ self.get(Term.SAMP_SCALE)*2,
                           self.get(Term.LINE_SCALE)*2],
                         dtype = np.float64 )

    
    def world_to_pixel( self, coord, skip_elevation = False, logger = None, method = 'B' ):

        if logger == None:
            logger = logging.getLogger( 'RPC00B.world_to_pixel' )

        #  Convert with offsets
        L = (coord[0] - self.get( Term.LON_OFF ) )    / self.get( Term.LON_SCALE )
        P = (coord[1] - self.get( Term.LAT_OFF ) )    / self.get( Term.LAT_SCALE )

        H = ( - self.get( Term.HEIGHT_OFF ) ) / self.get( Term.HEIGHT_SCALE )
        if skip_elevation == False:
            H = (coord[2] - self.get( Term.HEIGHT_OFF ) ) / self.get( Term.HEIGHT_SCALE )
        print( f'L: {L}, P: {P}, H: {H}' )

        plh_vec = self.get_plh_vector( P = P, L = L, H = H, method = method )

        r_n_n = np.dot( self.get_line_numerator_coefficients(),     plh_vec )
        r_n_d = np.dot( self.get_line_denominator_coefficients(),   plh_vec )
        c_n_n = np.dot( self.get_sample_numerator_coefficients(),   plh_vec )
        c_n_d = np.dot( self.get_sample_denominator_coefficients(), plh_vec )

        return np.array( [ (c_n_n/c_n_d) * self.get(Term.SAMP_SCALE) + self.get(Term.SAMP_OFF),
                           (r_n_n/r_n_d) * self.get(Term.LINE_SCALE) + self.get(Term.LINE_OFF) ],
                         dtype = np.float64 )

    
    def pixel_to_world( self,
                        pixel,
                        dem_model = None,
                        max_iterations = 10,
                        convergence_epsilon = 0.1,
                        method = 'B',
                        logger = None ):

        if logger == None:
            logger = logging.getLogger( 'RPC00B.pixel_to_world' )

        #  The image point must be adjusted by the adjustable parameters as well
        # as the scale and offsets given as part of the RPC param normalization.
        # NOTE: U = line, V = sample
        U = ( pixel[0] - self.get(Term.SAMP_OFF) ) / self.get(Term.SAMP_SCALE)
        V = ( pixel[1] - self.get(Term.LINE_OFF) ) / self.get(Term.LINE_SCALE)

        # Normalized height
        hval = self.get(Term.HEIGHT_SCALE)
        if not dem_model is None:
            hval = dem_model.stats.mean
        
        nhgt = ( hval - self.get(Term.HEIGHT_OFF) ) / self.get(Term.HEIGHT_SCALE)

        #  Initialize values for iterative cycle
        nlat = 0.0
        nlon = 0.0
        
        epsilonV = convergence_epsilon / self.get(Term.SAMP_SCALE)
        epsilonU = convergence_epsilon / self.get(Term.LINE_SCALE)
        
        iteration = 0

        # Iterate until the computed Uc, Vc is within epsilon of the desired
        # image point U, V:
        while iteration <= max_iterations:

            #  Calculate the normalized line and sample Uc, Vc as ratio of
            #  polynomials Pu, Qu and Pv, Qv:
            plh_vec = self.get_plh_vector( P = nlat,
                                           L = nlon,
                                           H = nhgt,
                                           method = method )

            Pv = np.dot( plh_vec, self.get_sample_numerator_coefficients() )
            Qv = np.dot( plh_vec, self.get_sample_denominator_coefficients() )

            Pu = np.dot( plh_vec, self.get_line_numerator_coefficients() )
            Qu = np.dot( plh_vec, self.get_line_denominator_coefficients() )
            
            if np.isnan( Pv ) or np.isnan( Pu ) or np.isnan( Qv ) or np.isnan( Qu ):
                return None
            
            #  Compute result
            xNew = Pv / Qv
            yNew = Pu / Qu
            
            #  Compute residuals between desired and computed line, sample:
            delta_U = U - yNew
            delta_V = V - xNew
            
            #  Check for convergence and skip re-linearization if converged:
            if abs(delta_U) > epsilonU or abs(delta_V) > epsilonV:

                #  Analytically compute the partials of each polynomial wrt lat, lon:
                dPu_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_line_numerator_coefficients() )
                dQu_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_line_denominator_coefficients() )
                dPv_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_sample_numerator_coefficients() )
                dQv_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_sample_denominator_coefficients() )
                dPu_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_line_numerator_coefficients() )
                dQu_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_line_denominator_coefficients() )
                dPv_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_sample_numerator_coefficients() )
                dQv_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_sample_denominator_coefficients() )
         
                # Analytically compute partials of quotients U and V wrt lat, lon:
                dU_dLat = ( Qu * dPu_dLat - Pu * dQu_dLat ) / ( Qu * Qu )
                dU_dLon = ( Qu * dPu_dLon - Pu * dQu_dLon ) / ( Qu * Qu )
                dV_dLat = ( Qv * dPv_dLat - Pv * dQv_dLat ) / ( Qv * Qv )
                dV_dLon = ( Qv * dPv_dLon - Pv * dQv_dLon ) / ( Qv * Qv )
         
                W = dU_dLon * dV_dLat - dU_dLat * dV_dLon
         
                # Now compute the corrections to normalized lat, lon:
                deltaLat = ( dU_dLon * delta_V - dV_dLon * delta_U ) / W
                deltaLon = ( dV_dLat * delta_U - dU_dLat * delta_V ) / W
                nlat += deltaLat
                nlon += deltaLon

            #  Check if we've triggered the exit condition
            if abs(delta_U) > epsilonU or abs(delta_V) > epsilonV:
                break
            
            iteration += 1

        #  Test for exceeding allowed number of iterations. Flag error if so:
        if iteration == max_iterations:
            logger.warning( f'Failed to converge after {max_iterations} iterations.' )
        
        #  Now un-normalize the ground point lat, lon and establish return quantity
        gnd_lon = nlon * self.get(Term.LON_SCALE) + self.get(Term.LON_OFF)
        gnd_lat = nlat * self.get(Term.LAT_SCALE) + self.get(Term.LAT_OFF)

        ground_point = np.array( [ gnd_lon,
                                   gnd_lat,
                                   dem_model.elevation_meters( np.array( [gnd_lon, gnd_lat] ) ) ],
                                 dtype = np.float64 )
        
        return ground_point

    
    def __str__(self):
        output  =  'RPC00B:\n'
        for k in self.data.keys():
            output += f'   - {k.name}:  {self.data[k]}\n'
        return output

    
    def get_plh_vector( self, P, L, H, method = 'B' ):
        
        if method == 'A':
            PLH_vec = np.array( [        1.0,           L,           P,          H,
                                       L * P,       L * H,       P * H,  L * P * H,
                                      L ** 2,      P ** 2,      H ** 2,     L ** 3,
                                  P * L ** 2,  H * L ** 2,  P ** 2 * L,     P ** 3,
                                  H * P ** 2,  H ** 2 * L,  H ** 2 * P,     H ** 3 ],
                                dtype = np.float64 )
            return PLH_vec
        
        else:
            PLH_vec = np.array( [        1.0,           L,           P,           H,
                                       L * P,       L * H,       P * H,      L ** 2,
                                      P ** 2,      H ** 2,   P * L * H,      L ** 3,
                                  L * P ** 2,  L * H ** 2,  L ** 2 * P,      P ** 3,
                                  P * H ** 2,  L ** 2 * H,  P ** 2 * H,      H ** 3 ],
                                dtype = np.float64 )
            
            return PLH_vec

    
    def get_sample_numerator_coefficients(self):

        return np.array( [ self.data[Term.SAMP_NUM_COEFF_1],  self.data[Term.SAMP_NUM_COEFF_2],
                           self.data[Term.SAMP_NUM_COEFF_3],  self.data[Term.SAMP_NUM_COEFF_4],
                           self.data[Term.SAMP_NUM_COEFF_5],  self.data[Term.SAMP_NUM_COEFF_6],
                           self.data[Term.SAMP_NUM_COEFF_7],  self.data[Term.SAMP_NUM_COEFF_8],
                           self.data[Term.SAMP_NUM_COEFF_9],  self.data[Term.SAMP_NUM_COEFF_10],
                           self.data[Term.SAMP_NUM_COEFF_11], self.data[Term.SAMP_NUM_COEFF_12],
                           self.data[Term.SAMP_NUM_COEFF_13], self.data[Term.SAMP_NUM_COEFF_14],
                           self.data[Term.SAMP_NUM_COEFF_15], self.data[Term.SAMP_NUM_COEFF_16],
                           self.data[Term.SAMP_NUM_COEFF_17], self.data[Term.SAMP_NUM_COEFF_18],
                           self.data[Term.SAMP_NUM_COEFF_19], self.data[Term.SAMP_NUM_COEFF_20] ],
                          dtype = np.float64 )

    
    def get_sample_denominator_coefficients(self):

        return np.array( [ self.data[Term.SAMP_DEN_COEFF_1],  self.data[Term.SAMP_DEN_COEFF_2],
                           self.data[Term.SAMP_DEN_COEFF_3],  self.data[Term.SAMP_DEN_COEFF_4],
                           self.data[Term.SAMP_DEN_COEFF_5],  self.data[Term.SAMP_DEN_COEFF_6],
                           self.data[Term.SAMP_DEN_COEFF_7],  self.data[Term.SAMP_DEN_COEFF_8],
                           self.data[Term.SAMP_DEN_COEFF_9],  self.data[Term.SAMP_DEN_COEFF_10],
                           self.data[Term.SAMP_DEN_COEFF_11], self.data[Term.SAMP_DEN_COEFF_12],
                           self.data[Term.SAMP_DEN_COEFF_13], self.data[Term.SAMP_DEN_COEFF_14],
                           self.data[Term.SAMP_DEN_COEFF_15], self.data[Term.SAMP_DEN_COEFF_16],
                           self.data[Term.SAMP_DEN_COEFF_17], self.data[Term.SAMP_DEN_COEFF_18],
                           self.data[Term.SAMP_DEN_COEFF_19], self.data[Term.SAMP_DEN_COEFF_20] ],
                          dtype = np.float64 )

    
    def get_line_numerator_coefficients(self):

        return np.array( [ self.data[Term.LINE_NUM_COEFF_1],  self.data[Term.LINE_NUM_COEFF_2],
                           self.data[Term.LINE_NUM_COEFF_3],  self.data[Term.LINE_NUM_COEFF_4],
                           self.data[Term.LINE_NUM_COEFF_5],  self.data[Term.LINE_NUM_COEFF_6],
                           self.data[Term.LINE_NUM_COEFF_7],  self.data[Term.LINE_NUM_COEFF_8],
                           self.data[Term.LINE_NUM_COEFF_9],  self.data[Term.LINE_NUM_COEFF_10],
                           self.data[Term.LINE_NUM_COEFF_11], self.data[Term.LINE_NUM_COEFF_12],
                           self.data[Term.LINE_NUM_COEFF_13], self.data[Term.LINE_NUM_COEFF_14],
                           self.data[Term.LINE_NUM_COEFF_15], self.data[Term.LINE_NUM_COEFF_16],
                           self.data[Term.LINE_NUM_COEFF_17], self.data[Term.LINE_NUM_COEFF_18],
                           self.data[Term.LINE_NUM_COEFF_19], self.data[Term.LINE_NUM_COEFF_20] ],
                          dtype = np.float64 )

    
    def get_line_denominator_coefficients(self):

        return np.array( [ self.data[Term.LINE_DEN_COEFF_1],  self.data[Term.LINE_DEN_COEFF_2],
                           self.data[Term.LINE_DEN_COEFF_3],  self.data[Term.LINE_DEN_COEFF_4],
                           self.data[Term.LINE_DEN_COEFF_5],  self.data[Term.LINE_DEN_COEFF_6],
                           self.data[Term.LINE_DEN_COEFF_7],  self.data[Term.LINE_DEN_COEFF_8],
                           self.data[Term.LINE_DEN_COEFF_9],  self.data[Term.LINE_DEN_COEFF_10],
                           self.data[Term.LINE_DEN_COEFF_11], self.data[Term.LINE_DEN_COEFF_12],
                           self.data[Term.LINE_DEN_COEFF_13], self.data[Term.LINE_DEN_COEFF_14],
                           self.data[Term.LINE_DEN_COEFF_15], self.data[Term.LINE_DEN_COEFF_16],
                           self.data[Term.LINE_DEN_COEFF_17], self.data[Term.LINE_DEN_COEFF_18],
                           self.data[Term.LINE_DEN_COEFF_19], self.data[Term.LINE_DEN_COEFF_20] ],
                          dtype = np.float64 )

    
    def dPoly_dLat( self, P, L, H, poly):

        parts = np.array( [poly[2], poly[4], poly[6], poly[8], poly[10],  poly[12], poly[14],  poly[15], poly[16],  poly[18] ], dtype = np.float64 )
        terms = np.array( [      1,       L,       H,   2 * P,    L * H, 2 * L * P,    L * L, 3 * P * P,    H * H, 2 * P * H ], dtype = np.float64 )
        return np.dot( parts, terms )

    
    def dPoly_dLon( self, P, L, H, poly):

        parts = np.array( [poly[1], poly[4], poly[5], poly[7], poly[10],  poly[11], poly[12],  poly[13], poly[14],  poly[17] ], dtype = np.float64 )
        terms = np.array( [      1,       P,       H,   2 * L,    P * H, 3 * L * L,    P * P,     H * H, 2 * P * L, 2 * L * H ], dtype = np.float64 )
        return np.dot( parts, terms )

    
    @staticmethod
    def from_dict( data ):
        model = RPC00B( data = data )
        return model

    
    @staticmethod
    def defaults():
        data = { Term.SAMP_OFF:           0.0, Term.SAMP_SCALE:         0.0,
                 Term.LINE_OFF:           0.0, Term.LINE_SCALE:         0.0,
                 Term.LON_OFF:            0.0, Term.LON_SCALE:          0.0,
                 Term.LAT_OFF:            0.0, Term.LAT_SCALE:          0.0,
                 Term.HEIGHT_OFF:         0.0, Term.HEIGHT_SCALE:       0.0,
                 Term.LINE_NUM_COEFF_1:   0.0, Term.LINE_NUM_COEFF_2:   0.0,
                 Term.LINE_NUM_COEFF_3:   0.0, Term.LINE_NUM_COEFF_4:   0.0,
                 Term.LINE_NUM_COEFF_5:   0.0, Term.LINE_NUM_COEFF_6:   0.0,
                 Term.LINE_NUM_COEFF_7:   0.0, Term.LINE_NUM_COEFF_8:   0.0,
                 Term.LINE_NUM_COEFF_9:   0.0, Term.LINE_NUM_COEFF_10:  0.0,
                 Term.LINE_NUM_COEFF_11:  0.0, Term.LINE_NUM_COEFF_12:  0.0,
                 Term.LINE_NUM_COEFF_13:  0.0, Term.LINE_NUM_COEFF_14:  0.0,
                 Term.LINE_NUM_COEFF_15:  0.0, Term.LINE_NUM_COEFF_16:  0.0,
                 Term.LINE_NUM_COEFF_17:  0.0, Term.LINE_NUM_COEFF_18:  0.0,
                 Term.LINE_NUM_COEFF_19:  0.0, Term.LINE_NUM_COEFF_20:  0.0,
                 Term.LINE_DEN_COEFF_1:   0.0, Term.LINE_DEN_COEFF_2:   0.0,
                 Term.LINE_DEN_COEFF_3:   0.0, Term.LINE_DEN_COEFF_4:   0.0,
                 Term.LINE_DEN_COEFF_5:   0.0, Term.LINE_DEN_COEFF_6:   0.0,
                 Term.LINE_DEN_COEFF_7:   0.0, Term.LINE_DEN_COEFF_8:   0.0,
                 Term.LINE_DEN_COEFF_9:   0.0, Term.LINE_DEN_COEFF_10:  0.0,
                 Term.LINE_DEN_COEFF_11:  0.0, Term.LINE_DEN_COEFF_12:  0.0,
                 Term.LINE_DEN_COEFF_13:  0.0, Term.LINE_DEN_COEFF_14:  0.0,
                 Term.LINE_DEN_COEFF_15:  0.0, Term.LINE_DEN_COEFF_16:  0.0,
                 Term.LINE_DEN_COEFF_17:  0.0, Term.LINE_DEN_COEFF_18:  0.0,
                 Term.LINE_DEN_COEFF_19:  0.0, Term.LINE_DEN_COEFF_20:  0.0,
                 Term.SAMP_NUM_COEFF_1:   0.0, Term.SAMP_NUM_COEFF_2:   0.0,
                 Term.SAMP_NUM_COEFF_3:   0.0, Term.SAMP_NUM_COEFF_4:   0.0,
                 Term.SAMP_NUM_COEFF_5:   0.0, Term.SAMP_NUM_COEFF_6:   0.0,
                 Term.SAMP_NUM_COEFF_7:   0.0, Term.SAMP_NUM_COEFF_8:   0.0,
                 Term.SAMP_NUM_COEFF_9:   0.0, Term.SAMP_NUM_COEFF_10:  0.0,
                 Term.SAMP_NUM_COEFF_11:  0.0, Term.SAMP_NUM_COEFF_12:  0.0,
                 Term.SAMP_NUM_COEFF_13:  0.0, Term.SAMP_NUM_COEFF_14:  0.0,
                 Term.SAMP_NUM_COEFF_15:  0.0, Term.SAMP_NUM_COEFF_16:  0.0,
                 Term.SAMP_NUM_COEFF_17:  0.0, Term.SAMP_NUM_COEFF_18:  0.0,
                 Term.SAMP_NUM_COEFF_19:  0.0, Term.SAMP_NUM_COEFF_20:  0.0,
                 Term.SAMP_DEN_COEFF_1:   0.0, Term.SAMP_DEN_COEFF_2:   0.0,
                 Term.SAMP_DEN_COEFF_3:   0.0, Term.SAMP_DEN_COEFF_4:   0.0,
                 Term.SAMP_DEN_COEFF_5:   0.0, Term.SAMP_DEN_COEFF_6:   0.0,
                 Term.SAMP_DEN_COEFF_7:   0.0, Term.SAMP_DEN_COEFF_8:   0.0,
                 Term.SAMP_DEN_COEFF_9:   0.0, Term.SAMP_DEN_COEFF_10:  0.0,
                 Term.SAMP_DEN_COEFF_11:  0.0, Term.SAMP_DEN_COEFF_12:  0.0,
                 Term.SAMP_DEN_COEFF_13:  0.0, Term.SAMP_DEN_COEFF_14:  0.0,
                 Term.SAMP_DEN_COEFF_15:  0.0, Term.SAMP_DEN_COEFF_16:  0.0,
                 Term.SAMP_DEN_COEFF_17:  0.0, Term.SAMP_DEN_COEFF_18:  0.0,
                 Term.SAMP_DEN_COEFF_19:  0.0, Term.SAMP_DEN_COEFF_20:  0.0 }

        return data   

    
    @staticmethod
    def from_components( center_pixel,
                         center_lla,
                         image_width,
                         image_height,
                         max_delta_lla ):

        model = RPC00B()
        
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
    def solve( model, gcps ):

        if logger == None:
            logger = logging.getLogger( 'RPC00B.solve' )

        #  Create values

        
        pass

        