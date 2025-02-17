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
import math

#  Numpy
import numpy as np

#  Terminus Libraries
from tmns.math.solver import pseudoinverse

class WLSQ_Solver:

    @staticmethod
    def system_of_equations( r,
                             lon_vals,
                             lat_vals,
                             hgt_vals,
                             logger = None ):

        if logger == None:
            logger = logging.getLogger( 'RPC00B.system_of_equations' )

        eq = np.zeros( [ len( r ), 39 ], dtype = np.float64 )
        for idx in range( len( r ) ):

            eq[idx][0]  = 1
            eq[idx][1]  = lon_vals[idx]
            eq[idx][2]  = lat_vals[idx]
            eq[idx][3]  = hgt_vals[idx]
            eq[idx][4]  = lon_vals[idx] * lat_vals[idx]
            eq[idx][5]  = lon_vals[idx] * hgt_vals[idx]
            eq[idx][6]  = lat_vals[idx] * hgt_vals[idx]
            eq[idx][7]  = lon_vals[idx] * lon_vals[idx]
            eq[idx][8]  = lat_vals[idx] * lat_vals[idx]
            eq[idx][9]  = hgt_vals[idx] * hgt_vals[idx]
            eq[idx][10] = lon_vals[idx] * lat_vals[idx] * hgt_vals[idx]
            eq[idx][11] = lon_vals[idx] * lon_vals[idx] * lon_vals[idx]
            eq[idx][12] = lon_vals[idx] * lat_vals[idx] * lat_vals[idx]
            eq[idx][13] = lon_vals[idx] * hgt_vals[idx] * hgt_vals[idx]
            eq[idx][14] = lon_vals[idx] * lon_vals[idx] * lat_vals[idx]
            eq[idx][15] = lat_vals[idx] * lat_vals[idx] * lat_vals[idx]
            eq[idx][16] = lat_vals[idx] * hgt_vals[idx] * hgt_vals[idx]
            eq[idx][17] = lon_vals[idx] * lon_vals[idx] * hgt_vals[idx]
            eq[idx][18] = lat_vals[idx] * lat_vals[idx] * hgt_vals[idx]
            eq[idx][19] = hgt_vals[idx] * hgt_vals[idx] * hgt_vals[idx]
            eq[idx][20] = -r[idx] * lon_vals[idx]
            eq[idx][21] = -r[idx] * lat_vals[idx]
            eq[idx][22] = -r[idx] * hgt_vals[idx]
            eq[idx][23] = -r[idx] * lon_vals[idx] * lat_vals[idx]
            eq[idx][24] = -r[idx] * lon_vals[idx] * hgt_vals[idx]
            eq[idx][25] = -r[idx] * lat_vals[idx] * hgt_vals[idx]
            eq[idx][26] = -r[idx] * lon_vals[idx] * lon_vals[idx]
            eq[idx][27] = -r[idx] * lat_vals[idx] * lat_vals[idx]
            eq[idx][28] = -r[idx] * hgt_vals[idx] * hgt_vals[idx]
            eq[idx][29] = -r[idx] * lon_vals[idx] * lat_vals[idx] * hgt_vals[idx]
            eq[idx][30] = -r[idx] * lon_vals[idx] * lon_vals[idx] * lon_vals[idx]
            eq[idx][31] = -r[idx] * lon_vals[idx] * lat_vals[idx] * lat_vals[idx]
            eq[idx][32] = -r[idx] * lon_vals[idx] * hgt_vals[idx] * hgt_vals[idx]
            eq[idx][33] = -r[idx] * lon_vals[idx] * lon_vals[idx] * lat_vals[idx]
            eq[idx][34] = -r[idx] * lat_vals[idx] * lat_vals[idx] * lat_vals[idx]
            eq[idx][35] = -r[idx] * lat_vals[idx] * hgt_vals[idx] * hgt_vals[idx]
            eq[idx][36] = -r[idx] * lon_vals[idx] * lon_vals[idx] * hgt_vals[idx]
            eq[idx][37] = -r[idx] * lat_vals[idx] * lat_vals[idx] * hgt_vals[idx]
            eq[idx][38] = -r[idx] * hgt_vals[idx] * hgt_vals[idx] * hgt_vals[idx]


        return eq
    
    @staticmethod
    def solve_coefficients( pix_terms,
                            lon_vals, 
                            lat_vals, 
                            hgt_vals, 
                            logger = None ):

        if logger == None:
            logger = logging.getLogger( 'WLSQ_Solver.solve_coefficients' )

        idx = 0

        r = np.copy( pix_terms )
        w = np.ones( pix_terms.shape )
        w = np.diag( w )

        M = WLSQ_Solver.system_of_equations( r,
                                             lon_vals,
                                             lat_vals,
                                             hgt_vals,
                                             logger = logger )

        iteration = 0
        coefficients = None
        temp_coeff = None
        residual_value = 1e6
        for index in range( 10 ):

            if residual_value < 0.00001:
                break

            W2 = np.dot( w, w )

            temp_coeff = pseudoinverse( M.T @ W2 @ M ) @ M.T @ W2 @ r

            #  Set denominator matrix
            denominator = np.ones( 20 )
            for idx in range( 19 ):
                denominator[idx+1] = temp_coeff[20 + idx]
            
            #  Setup weight matrix
            weights = WLSQ_Solver.setup_weight_matrix( denominator,
                                                       r,
                                                       lon_vals,
                                                       lat_vals,
                                                       hgt_vals )
            
            #  Compute Residuals
            residual = M.T @ W2 @ ( M @ temp_coeff - r )

            #  Compute inner product
            temp_res = np.dot( residual, residual )
            residual_value = math.sqrt( float( temp_res ) )
            
            iteration += 1
        
        coefficients = temp_coeff

        return coefficients
    
    @staticmethod
    def setup_weight_matrix( coeffs,
                             f, x, y, z ):

        result = np.zeros( f.shape[0] )
        row = np.zeros( len( coeffs ) )

        for idx in range( f.shape[0] ):
            row[0]  = 1
            row[1]  = x[idx]
            row[2]  = y[idx]
            row[3]  = z[idx]
            row[4]  = x[idx] * y[idx]
            row[5]  = x[idx] * z[idx]
            row[6]  = y[idx] * z[idx]
            row[7]  = x[idx] * x[idx]
            row[8]  = y[idx] * y[idx]
            row[9]  = z[idx] * z[idx]
            row[10] = x[idx] * y[idx] * z[idx]
            row[11] = x[idx] * x[idx] * x[idx]
            row[12] = x[idx] * y[idx] * y[idx]
            row[13] = x[idx] * z[idx] * z[idx]
            row[14] = x[idx] * x[idx] * y[idx]
            row[15] = y[idx] * y[idx] * y[idx]
            row[16] = y[idx] * z[idx] * z[idx]
            row[17] = x[idx] * x[idx] * z[idx]
            row[18] = y[idx] * y[idx] * z[idx]
            row[19] = z[idx] * z[idx] * z[idx]

            result[idx] = 0.0
            for idx2 in range( len( row ) ):
                result[idx] += row[idx2] * coeffs[idx2]
            
      
            if result[idx] > 0.000000001:
                result[idx] = 1.0/result[idx]

        return result
    

class LM_Solver:

    @staticmethod
    def fitness( coeffs, *args, **kwargs ):

        #  Update the model coefficients
        model = kwargs['model']

        model.set_sample_numerator_coefficients( coeffs[0:20] )
        model.set_sample_denominator_coefficients( coeffs[20:40] )
        model.set_line_numerator_coefficients( coeffs[40:60] )
        model.set_line_denominator_coefficients( coeffs[60:80] )

        #  Test against GCPs
        sum_delta = 0
        gcps = kwargs['gcps']
        skip_elevation = kwargs['skip_elevation']
        logger = kwargs['logger']
        rpc_type = kwargs['rpc_type']

        residuals = np.zeros( len( gcps ) )

        sum_res = 0
        idx = 0
        for gcp in gcps:

            pix = model.world_to_pixel( gcp.coordinate,
                                        skip_elevation = skip_elevation,
                                        logger         = logger,
                                        rpc_type       = rpc_type )

            delta = pix - gcp.pixel
            residuals[idx] = math.sqrt( np.dot( delta, delta ) )

            if np.isinf(residuals[idx]) or np.isnan( residuals[idx] ):
                residuals[idx] = 99999999
            
            sum_res += residuals[idx]
            idx += 1

        mean_res = sum_res / len(gcps)
        #print( f'Mean Residual: {mean_res}' )

        
        return residuals


            

