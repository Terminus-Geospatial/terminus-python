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

#  Numpy 
import numpy as np

#  Terminus Libraries
from tmns.nitf.base import TRE_Base

class Tag(Enum):

    IID     =  1
    EDITION =  2
    R0      =  3
    RX      =  4
    RY      =  5
    RZ      =  6
    RXX     =  7
    RXY     =  8
    RXZ     =  9
    RYY     = 10
    RYZ     = 11
    RZZ     = 12
    C0      = 13
    CX      = 14
    CY      = 15
    CZ      = 16
    CXX     = 17
    CXY     = 18
    CXZ     = 19
    CYY     = 20
    CYZ     = 21
    CZZ     = 22
    RNIS    = 23
    CNIS    = 24
    TNIS    = 25
    RSSIZE  = 26
    CSSIZ   = 25


class RSMPIA(TRE_Base):

    def __init__(self):
        pass

    def get( self, key ):
        return self.data[key]

    

    def low_order_polynomial( self, lla_rad ):

        plh_vector = RSMPIA.plh_vector( lla_rad )
        return np.array( [ np.dot( self.column_coeffs(), plh_vector ),
                           np.dot( self.row_coeffs(), plh_vector ) ],
                         dtype = np.float64 )

    @staticmethod
    def plh_vector( lla_rad ):
        x = lla_rad[0]
        y = lla_rad[1]
        z = lla_rad[2]
        return np.array( [   1,   x,   y,   z, x*x, 
                           x*y, x*z, y*y, y*z, z*z ],
                         dtype = np.float64 )


    def column_coeffs(self):
        return np.array( [ self.get( Tag.C0 ),   self.get( Tag.CX ),
                           self.get( Tag.CY ),   self.get( Tag.CZ ),
                           self.get( Tag.CXX ),  self.get( Tag.CXY ),
                           self.get( Tag.CXZ ),  self.get( Tag.CYY ),
                           self.get( Tag.CYZ ),  self.get( Tag.CZZ )],
                        dtype = np.float64 )

    def row_coeffs(self):
        return np.array( [ self.get( Tag.R0 ),
                           self.get( Tag.RX ),
                           self.get( Tag.RY ),
                           self.get( Tag.RZ ),
                           self.get( Tag.RXX ),
                           self.get( Tag.RXY ),
                           self.get( Tag.RXZ ),
                           self.get( Tag.RYY ),
                           self.get( Tag.RYZ ),
                           self.get( Tag.RZZ )],
                        dtype = np.float64 )
