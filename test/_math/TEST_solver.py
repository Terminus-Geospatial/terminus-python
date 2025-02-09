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
import unittest

#  Terminus Libraries
from tmns.math.solver import pseudoinverse

#  Numpy 
import numpy as np


class math_solver( unittest.TestCase ):

    def test_pseudoinverse( self ):

        #  Create test matrix
        A1 = np.array( [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10,11,12]],
                      dtype = np.float64 )
        
        A1_inv = pseudoinverse( A1 )        
        A1_exp = np.linalg.pinv( A1 )
        self.assertLess( np.sum( np.abs( A1_inv - A1_exp ) ) )
        
