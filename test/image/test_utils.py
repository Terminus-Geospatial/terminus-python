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

#  Numpy Libraries
import numpy as np

#  Python Libraries
import unittest

#  Terminus Libraries
from tmns.image.utils import ( interp_index,
                               interp_index_nd )


class image_utils(unittest.TestCase):

    def test_interp_index(self):

        res = interp_index( 0.5 )
        self.assertEqual( len(res), 2 )
        self.assertEqual( res[0], 0 )
        self.assertEqual( res[1], 1 )

        res = interp_index( 5.0000001 )
        self.assertEqual( len(res), 2 )
        self.assertEqual( res[0], 5 )
        self.assertEqual( res[1], 5 )

        res = interp_index( 9.999999999 )
        self.assertEqual( len(res), 2 )
        self.assertEqual( res[0], 10 )
        self.assertEqual( res[1], 10 )

    def test_interp_index_nd( self ):

        res = interp_index_nd( [0.1, -2.2, 3.3] )
        
        self.assertEqual( len(res), 8 )
        self.assertLess( np.sum( np.abs( np.array( res[0] ) - np.array( [0, -3, 3] ) ) ), 0.01 )
        self.assertLess( np.sum( np.abs( np.array( res[1] ) - np.array( [1, -3, 3] ) ) ), 0.01 )
        self.assertLess( np.sum( np.abs( np.array( res[2] ) - np.array( [0, -2, 3] ) ) ), 0.01 )
        self.assertLess( np.sum( np.abs( np.array( res[3] ) - np.array( [1, -2, 3] ) ) ), 0.01 )
        self.assertLess( np.sum( np.abs( np.array( res[4] ) - np.array( [0, -3, 4] ) ) ), 0.01 )
        self.assertLess( np.sum( np.abs( np.array( res[5] ) - np.array( [1, -3, 4] ) ) ), 0.01 )
        self.assertLess( np.sum( np.abs( np.array( res[6] ) - np.array( [0, -2, 4] ) ) ), 0.01 )
        self.assertLess( np.sum( np.abs( np.array( res[7] ) - np.array( [1, -2, 4] ) ) ), 0.01 )