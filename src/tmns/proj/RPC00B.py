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

#  Numpy
import numpy as np
np.set_printoptions(precision=6, floatmode='fixed')

import scipy.optimize

#  Project Libraries
from tmns.core.types       import GCP
from tmns.math.geometry    import Rectangle
from tmns.math.solver      import pseudoinverse
from tmns.proj.rpc_utils   import ( LM_Solver,
                                    WLSQ_Solver )
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
    RMS_ERROR          = 91
    MAX_ERROR          = 92

    @staticmethod
    def from_str( key ):
        return Term[key]
    
    @staticmethod
    def get_sample_num( off ):
        id = Term.SAMP_NUM_COEFF_1.value + off - 1
        return Term(id)

    @staticmethod
    def get_sample_den( off ):
        id = Term.SAMP_DEN_COEFF_1.value + off - 1
        return Term(id)
    
    @staticmethod
    def get_line_num( off ):
        id = Term.LINE_NUM_COEFF_1.value + off - 1
        return Term(id)

    @staticmethod
    def get_line_den( off ):
        id = Term.LINE_DEN_COEFF_1.value + off - 1
        return Term(id)


class RPC_Type(Enum):
    A = 'A'
    B = 'B'

class Solve_Method(Enum):
    WEIGHTED_LEAST_SQUARES = 1
    LEVENBURG_MARQUARDT    = 2

class RPC00B(BaseTransformer):

    def __init__(self, data = None ):
        '''
        Constructor for RPC object
        '''
        self.data = RPC00B.defaults()
        if not data is None:

            for k in data.keys():
                self.data[k] = data[k]

        self.corners = None

    
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

    def world_bounds(self, dem = None ):
        
        if self.corners == None:
            self.estimate_bounds( dem = dem )
        
        bbox = Rectangle( self.corners['tl'] )
        bbox.add_point( self.corners['tr'])
        bbox.add_point( self.corners['bl'])
        bbox.add_point( self.corners['br'])

        return bbox
    
    def world_to_pixel( self,
                        coord, 
                        skip_elevation: bool = False,
                        logger = None,
                        rpc_type: RPC_Type = RPC_Type.B ):

        if logger == None:
            logger = logging.getLogger( 'RPC00B.world_to_pixel' )

        #  Convert with offsets
        L = (coord[0] - self.get( Term.LON_OFF ) )    / self.get( Term.LON_SCALE )
        P = (coord[1] - self.get( Term.LAT_OFF ) )    / self.get( Term.LAT_SCALE )

        H = ( - self.get( Term.HEIGHT_OFF ) ) / self.get( Term.HEIGHT_SCALE )
        if skip_elevation == False:
            H = (coord[2] - self.get( Term.HEIGHT_OFF ) ) / self.get( Term.HEIGHT_SCALE )
        #logger.debug( f'L: {L}, P: {P}, H: {H}' )

        plh_vec = self.get_plh_vector( P = P, L = L, H = H, rpc_type = rpc_type )

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
                        max_iterations: int = 50,
                        convergence_epsilon: float = 0.1,
                        rpc_type: RPC_Type = RPC_Type.B,
                        logger = None ):
        '''
        Convert a Pixel coordinate into World coordinates.
        '''

        if logger == None:
            logger = logging.getLogger( 'RPC00B.pixel_to_world' )


        #  The image point must be adjusted by the adjustable parameters as well
        # as the scale and offsets given as part of the RPC param normalization.
        # NOTE: U = line, V = sample
        V = ( pixel[0] - self.get(Term.SAMP_OFF) ) / self.get(Term.SAMP_SCALE)
        U = ( pixel[1] - self.get(Term.LINE_OFF) ) / self.get(Term.LINE_SCALE)

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
            plh_vec = self.get_plh_vector( P        = nlat,
                                           L        = nlon,
                                           H        = nhgt,
                                           rpc_type = rpc_type )
            
            Pv = np.dot( plh_vec, self.get_sample_numerator_coefficients() )
            Qv = np.dot( plh_vec, self.get_sample_denominator_coefficients() )

            Pu = np.dot( plh_vec, self.get_line_numerator_coefficients() )
            Qu = np.dot( plh_vec, self.get_line_denominator_coefficients() )
            
            if np.isnan( Pv ) or np.isnan( Pu ) or np.isnan( Qv ) or np.isnan( Qu ):
                logger.warning( f'Iteration: {iteration}' )
                logger.warning( f'Pixel: {pixel}' )
                logger.warning( f'NLon: {nlon}, NLat: {nlat}, Nhgt: {nhgt}' )
                logger.warning( f'PLH: {plh_vec}' )
                logger.warning( f'SNum: {self.get_sample_numerator_coefficients()}')
                logger.warning( f'Resulting projection yielded a nan: Pv: {Pv}, Pu: {Pu}, Qv: {Qv}, Qu: {Qu}' )
                return None
            
            #  Compute result
            Vc = Pv / Qv
            Uc = Pu / Qu
            
            #  Compute residuals between desired and computed line, sample:
            delta_U = U - Uc
            delta_V = V - Vc
            
            #logger.debug( f' - Check 1: DeltaU: {delta_U:0.6f}, DeltaV: {delta_V:0.6f}, eU: {epsilonU:0.6f}, eV: {epsilonV:0.6f}' )
            #  Check for convergence and skip re-linearization if converged:
            if abs(delta_U) > epsilonU or abs(delta_V) > epsilonV:

                #  Analytically compute the partials of each polynomial wrt lat, lon:
                dPu_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_line_numerator_coefficients(),     rpc_type = rpc_type )
                dQu_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_line_denominator_coefficients(),   rpc_type = rpc_type )
                dPv_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_sample_numerator_coefficients(),   rpc_type = rpc_type )
                dQv_dLat = self.dPoly_dLat( nlat, nlon, nhgt, self.get_sample_denominator_coefficients(), rpc_type = rpc_type )
                dPu_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_line_numerator_coefficients(),     rpc_type = rpc_type )
                dQu_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_line_denominator_coefficients(),   rpc_type = rpc_type )
                dPv_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_sample_numerator_coefficients(),   rpc_type = rpc_type )
                dQv_dLon = self.dPoly_dLon( nlat, nlon, nhgt, self.get_sample_denominator_coefficients(), rpc_type = rpc_type )
         
                if abs(Qu) < 0.00001:
                    logger.warning( f'Cutting early at iteration {iteration}. Qu ({Qu}) too low. NLAT: {nlat}, NLON: {nlon}' )
                    break
                if abs(Qv) < 0.00001:
                    logger.warning( f'Cutting early at iteration {iteration}. Qu ({Qv}) too low. NLAT: {nlat}, NLON: {nlon}' )
                    break

                # Analytically compute partials of quotients U and V wrt lat, lon:
                dU_dLat = ( Qu * dPu_dLat - Pu * dQu_dLat ) / ( Qu * Qu )
                dU_dLon = ( Qu * dPu_dLon - Pu * dQu_dLon ) / ( Qu * Qu )
                dV_dLat = ( Qv * dPv_dLat - Pv * dQv_dLat ) / ( Qv * Qv )
                dV_dLon = ( Qv * dPv_dLon - Pv * dQv_dLon ) / ( Qv * Qv )
         
                W = dU_dLon * dV_dLat - dU_dLat * dV_dLon
                if abs(W) < 0.00001:
                    logger.warning( f'Cutting early at iteration {iteration}. W ({W}) too low. NLAT: {nlat}, NLON: {nlon}' )
                    break
         
                # Now compute the corrections to normalized lat, lon:
                deltaLat = ( dU_dLon * delta_V - dV_dLon * delta_U ) / W
                deltaLon = ( dV_dLat * delta_U - dU_dLat * delta_V ) / W
                if np.isnan( deltaLat ) or np.isnan( deltaLon ):
                    logger.warning( f'Cutting early at iteration {iteration}. Nan found: {deltaLat}, {deltaLon}' )
                    break
                if np.isinf( deltaLat ) or np.isinf( deltaLon ):
                    logger.warning( f'Cutting early at iteration {iteration}. Inf found: {deltaLat}, {deltaLon}' )
                    break

                nlat += deltaLat
                nlon += deltaLon

            #  Check if we've triggered the exit condition
            if abs(delta_U) < epsilonU and abs(delta_V) < epsilonV:
                break
            
            iteration += 1

        #  Test for exceeding allowed number of iterations. Flag error if so:
        if iteration == max_iterations:
            logger.debug( f'Pixel {pixel} failed to converge after {max_iterations} iterations.' )
            logger.debug( f'     deltaU: {abs(delta_U)}, epsU: {epsilonU}, deltaV: {abs(delta_V)}, epsV: {epsilonV}' )

        #  Now un-normalize the ground point lat, lon and establish return quantity
        gnd_lon = nlon * self.get(Term.LON_SCALE) + self.get(Term.LON_OFF)
        gnd_lat = nlat * self.get(Term.LAT_SCALE) + self.get(Term.LAT_OFF)

        ground_point = np.array( [ gnd_lon,
                                   gnd_lat,
                                   dem_model.elevation_meters( np.array( [gnd_lon, gnd_lat] ) ) ],
                                 dtype = np.float64 )
        
        return ground_point


    @staticmethod
    def adjust_pixel( model, dem, pix_act, pix_exp, world_act ):
        
        pix_adj = np.zeros( pix_act.shape )

        pix_delta = pix_exp - pix_act
        for idx in range( len( pix_delta ) ):
            if abs( pix_delta[idx] ) > 10:
                pix_adj[idx] = pix_delta[idx] / 2.0
        
        return model.pixel_to_world( pix_act + pix_adj,
                                     dem_model = dem ), pix_delta
        
        


    def estimate_bounds(self, dem, num_adjustments = 10 ):

        #  Get the image dims
        image_size = self.image_size_pixels()

        #  Get the 4 corners
        self.corners = { 'tl': self.pixel_to_world( np.array( [ 0, 0 ], dtype = np.float64 ),
                                                    dem_model = dem ),
                         'tr': self.pixel_to_world( np.array( [ image_size[0]-1, 0 ],dtype = np.float64 ),
                                                    dem_model = dem ),
                         'bl': self.pixel_to_world( np.array( [ 0, image_size[1]-1 ],dtype = np.float64 ),
                                                    dem_model = dem ),
                         'br': self.pixel_to_world( np.array( [ image_size[0]-1, image_size[1]-1 ], dtype = np.float64 ),
                                                    dem_model = dem ) }
        
        #for iter in range( num_adjustments ):
        #
        #    #  Offer adjustment for tl
        #    tl_pix_act = self.world_to_pixel( self.corners['tl'] )
        #    self.corners['tl'], delta = RPC00B.adjust_pixel( model    = self,
        #                                                     dem = dem, 
        #                                                     pix_act   = tl_pix_act,
        #                                                     pix_exp   = np.array( [ 0, 0 ], dtype = np.float64 ),
        #                                                     world_act = self.corners['tl'] )
        #    
        #    print( f'Iteration: {iter}, TL Adjustment: {delta}' )
    
    def __str__(self):
        output  =  'RPC00B:\n'
        for k in self.data.keys():
            output += f'   - {k.name}:  {self.data[k]}\n'
        return output

    
    def get_plh_vector( self, P, L, H, rpc_type = RPC_Type.B ):
        
        if rpc_type == RPC_Type.A:
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

    def set_sample_numerator_coefficients( self, coeffs ):

        for idx in range( 20 ):
            self.set( Term.get_sample_num( idx + 1 ), coeffs[idx] )

    
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

    def set_sample_denominator_coefficients( self, coeffs ):

        for idx in range( 20 ):
            self.set( Term.get_sample_den( idx + 1 ), coeffs[idx] )
    
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

    def set_line_numerator_coefficients( self, coeffs ):

        for idx in range( 20 ):
            self.set( Term.get_line_num( idx + 1 ), coeffs[idx] )
    
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

    def set_line_denominator_coefficients( self, coeffs ):

        for idx in range( 20 ):
            self.set( Term.get_line_den( idx + 1 ), coeffs[idx] )
    
    def dPoly_dLat( self, P, L, H, poly, rpc_type: RPC_Type = RPC_Type.B ):

        parts = None
        terms = None

        if rpc_type == RPC_Type.A:
            parts = np.array( [poly[2], poly[4], poly[6], poly[7], poly[9],  poly[12],   poly[14],  poly[15],   poly[16],  poly[18] ], dtype = np.float64 )
            terms = np.array( [      1,       L,       H,   L * H,   2 * P,     L * L,  2 * L * P, 3 * P * P,  2 * P * H,     H * H ], dtype = np.float64 )

        elif rpc_type == RPC_Type.B:
            parts = np.array( [poly[2], poly[4], poly[6], poly[8], poly[10],  poly[12], poly[14],  poly[15], poly[16],  poly[18] ], dtype = np.float64 )
            terms = np.array( [      1,       L,       H,   2 * P,    L * H, 2 * L * P,    L * L, 3 * P * P,    H * H, 2 * P * H ], dtype = np.float64 )
        
        return np.dot( parts, terms )

    
    def dPoly_dLon( self, P, L, H, poly, rpc_type: RPC_Type = RPC_Type.B ):

        parts = None
        terms = None

        if rpc_type == RPC_Type.A:
            parts = np.array( [poly[1], poly[4], poly[5], poly[7], poly[8],  poly[11], poly[12],  poly[13], poly[14],  poly[17] ], dtype = np.float64 )
            terms = np.array( [      1,       P,       H,   P * H,   2 * L, 3 * L * L, 2 * L * P, 2 * L * H,   P * P,     H * H ], dtype = np.float64 )

        elif rpc_type == RPC_Type.B:
            parts = np.array( [poly[1], poly[4], poly[5], poly[7], poly[10],  poly[11], poly[12],  poly[13], poly[14],  poly[17] ], dtype = np.float64 )
            terms = np.array( [      1,       P,       H,   2 * L,    P * H, 3 * L * L,    P * P,     H * H, 2 * P * L, 2 * L * H ], dtype = np.float64 )
        
        return np.dot( parts, terms )

    def dPoly_dHgt( self, P, L, H, poly, rpc_type: RPC_Type = RPC_Type.B ):

        if rpc_type == RPC_Type.A:
            parts = np.array( [poly[3], poly[5], poly[6], poly[7], poly[10],  poly[13],  poly[16],  poly[17],  poly[18],  poly[19] ], dtype = np.float64 )
            terms = np.array( [      1,       L,       P,   L * P,    2 * H,     L * L,     P * P, 2 * L * H, 2 * P * H,  3 * H * H ], dtype = np.float64 )
        
        elif rpc_type == RPC_Type.B:
            parts = np.array( [poly[3], poly[5], poly[6], poly[9], poly[10],  poly[13],   poly[16],  poly[17],  poly[18],  poly[19] ], dtype = np.float64 )
            terms = np.array( [      1,       L,       P,   2 * H,    L * P, 2 * L * H,  2 * P * H,     L * L,     P * P, 3 * H * H ], dtype = np.float64 )
      
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
                 Term.SAMP_DEN_COEFF_19:  0.0, Term.SAMP_DEN_COEFF_20:  0.0,
                 Term.RMS_ERROR:          0.0, Term.MAX_ERROR:          0.0 }

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

    def write_txt( self, pathname ):

        with open( pathname, 'w' ) as fout:
            for k in self.data.keys():
                fout.write( f'{k.name}: {self.data[k]}\n' )

    @staticmethod
    def solve( gcps: list[GCP], 
               image_size           = None,
               dem                  = None, 
               rpc_type: RPC_Type   = RPC_Type.B, 
               skip_elevation       = False,
               solver: Solve_Method = Solve_Method.WEIGHTED_LEAST_SQUARES,
               logger               = None ):

        res = None
        if solver == Solve_Method.WEIGHTED_LEAST_SQUARES:
            res = RPC00B.solve_wlsq( gcps,
                                     rpc_type = rpc_type,
                                     logger   = logger )
        
        elif solver == Solve_Method.LEVENBURG_MARQUARDT:
            res = RPC00B.solve_lm( gcps,
                                   dem      = dem,
                                   rpc_type = rpc_type,
                                   logger   = logger )

        #  Solve the bounding rectangle
        res.estimate_bounds( dem = dem )

        return res

    @staticmethod
    def solve_lm( gcps: list[GCP], 
                   image_size           = None,
                   dem                  = None,
                   skip_elevation       = False, 
                   rpc_type: RPC_Type   = RPC_Type.B,
                   logger               = None ):
        
        if logger == None:
            logger = logging.getLogger( 'RPC00B.solve_lm' )

        #  Create model
        model = RPC00B.solve_wlsq( gcps,
                                   image_size,
                                   rpc_type,
                                   logger )
        
        logger.debug( 'Optimizing using SciPy Least Squares' )
        parts = { 'model': model,
                  'gcps':  gcps,
                  'skip_elevation': skip_elevation,
                  'logger': logger,
                  'rpc_type':  rpc_type }
        
        init_coeffs = np.concatenate( [ model.get_sample_numerator_coefficients(),
                                        model.get_sample_denominator_coefficients(),
                                        model.get_line_numerator_coefficients(),
                                        model.get_line_denominator_coefficients() ] )

        res = scipy.optimize.least_squares( LM_Solver.fitness,
                                            x0 = init_coeffs,
                                            method = 'lm',
                                            verbose = 2,
                                            ftol = 1e-4,
                                            loss = 'linear',
                                            kwargs=parts )

        #  Update teh model
        return model
    
    @staticmethod
    def solve_wlsq( gcps: list[GCP], 
                    image_size           = None,
                    rpc_type: RPC_Type   = RPC_Type.B,
                    logger               = None ):
        
        if logger == None:
            logger = logging.getLogger( 'RPC00B.solve_wlsq' )
        logger.debug( 'Solving for base model with Weighted Least Squares' )


        model = RPC00B.create_base_model( gcps,
                                          image_size,
                                          rpc_type,
                                          logger )
        rpc_model = model['rpc_model']

        #  Perform Least-Squares Fit
        x_coeff = WLSQ_Solver.solve_coefficients( model['fx'], model['x'], model['y'], model['z'] )
        y_coeff = WLSQ_Solver.solve_coefficients( model['fy'], model['x'], model['y'], model['z'] )

        rpc_model.set( Term.LINE_NUM_COEFF_1, y_coeff[0] )
        rpc_model.set( Term.LINE_DEN_COEFF_1, 1.0 )
        rpc_model.set( Term.SAMP_NUM_COEFF_1, x_coeff[0] )
        rpc_model.set( Term.SAMP_DEN_COEFF_1, 1.0 )

        for idx in range( 2, 20 ):
            rpc_model.set( Term.get_line_num(idx), y_coeff[idx] )
            rpc_model.set( Term.get_line_den(idx), y_coeff[idx+19] )
            rpc_model.set( Term.get_sample_num(idx), x_coeff[idx] )
            rpc_model.set( Term.get_sample_den(idx), x_coeff[idx+19] )


        #  Compute RMSE for errors
        sumSquareError = 0
        maxResidual = 0
        for gcp in gcps:

            npix = rpc_model.world_to_pixel( gcp.coordinate,
                                             skip_elevation = False,
                                             logger         = logger,
                                             rpc_type       = rpc_type )
            
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
    def create_base_model( gcps: list[GCP],
                           image_size        = None,
                           rpc_type: RPC_Type = RPC_Type.B,
                           logger             = None ):

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
        maxZ = 0

        #------------------------------------------#
        #-        Normalize all coordinates       -#
        #------------------------------------------#
        for idx in range( len( gcps ) ):

            #  Normalize geographic coordinates
            d_lla = gcps[idx].coordinate - center_lla
            
            x[idx] = d_lla[0]
            y[idx] = d_lla[1]
            z[idx] = d_lla[2]

            maxZ = abs(gcps[idx].coordinate[2])

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
            
        #  Solve delta        
        if abs(maxDeltaLLA[2]) < 1:
            logger.info( f'Height model is too small ({maxDeltaLLA[2]}).  Using delta of {maxZ}' )
            maxDeltaLLA[2] = maxZ

        for idx in range( len( gcps ) ):

            x[idx] /= maxDeltaLLA[0]
            y[idx] /= maxDeltaLLA[1]
            z[idx] /= maxDeltaLLA[2]

        #  Create a new model
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
    
        return { 'rpc_model': rpc_model,
                 'x': x,
                 'y': y,
                 'z': z,
                 'fx': fx,
                 'fy': fy }
    
    
