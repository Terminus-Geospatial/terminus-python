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

#  Numpy 
import numpy as np

def pseudoinverse( A, logger = None, epsilon = 0.00001 ):
    '''
    Use np.linalg.pinv() instead.  This is only shown to help
    people when doing it themselves in another language.
    '''

    U, S, V_t = np.linalg.svd( A, full_matrices = False )

    S_m = np.diag( S )
    for x in range( len( S ) ):
        if S_m[x,x] < epsilon:
            S_m[x,x] = 0
        else:
            S_m[x,x] = 1 / S_m[x,x]

    A_inv = V_t.T @ S_m @ U.T

    return A_inv
