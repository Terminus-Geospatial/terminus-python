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

    IID      =  1
    EDITION  =  2
    RSN      =  3    # Row Section Number
    CSN      =  4    # Column Section Number
    RFEP     =  5    # Row Fitting Error
    CFEP     =  6    # Column Fitting Error
    RNRMO    =  7    # Row Normalization Offset
    CNRMO    =  8    # Column Normalization Offset
    XNRMO    =  9    # X Normalization Offset
    YNRMO    = 10    # Y Normalization Offset
    ZNRMO    = 11    # Z Normalization Offset
    RNRMSF   = 12    # Row Normalization Scale Factor
    CNRMSF   = 13    # Column Normalization Scale Factor
    XNRMSF   = 14    # X Row Normalization Scale Factor
    YNRMSF   = 15    # Y Row Normalization Scale Factor
    ZNRMSF   = 16    # Z Row Normalization Scale Factor
    RNPWRX   = 17    # Row Numerator Polynomial Maximum Power of X
    RNPWRY   = 18    # Row Numerator Polynomial Maximum Power of Y
    RNPWRZ   = 19    # Row Numerator Polynomial Maximum Power of Z
    RNTRMS   = 20    # Row Numerator Polynomial Number of Polynomial Terms
    RNPCF    = 21    # Polynomial Coefficient
    RDPWRX   = 22    # Row Denominator Polynomial Maximum Power of X
    RDPWRY   = 23    # Row Denominator Polynomial Maximum Power of Y
    RDPWRZ   = 24    # Row Denominator Polynomial Maximum Power of Z
    RDTRMS   = 25    # Row Denominator Polynomial Number of Polynomial Terms
    RDPCF    = 26    # Polynomial Coefficient
    CNPWRX   = 27    # Column Numerator Polynomial Maximum Power of X
    CNPWRY   = 28    # Column Numerator Polynomial Maximum Power of Y
    CNPWRZ   = 29    # Column Numerator Polynomial Maximum Power of Z
    CNTRMS   = 30    # Column Numerator Polynomial Number of Polynomial Terms
    CNPCF    = 31    # Polynomial Coefficient
    CDPWRX   = 32    # Column Denominator Polynomial Maximum Power of X
    CDPWRY   = 33    # Column Denominator Polynomial Maximum Power of Y
    CDPWRZ   = 34    # Column Denominator Polynomial Maximum Power of Z
    CDTRMS   = 35    # Column Denominator Polynomial Number of Polynomial Terms
    CDPCF    = 36    # Polynomial Coefficient
    

class RSMPCA(TRE_Base):

    def __init__(self):

        self.data = RSMPCA.defaults()

    def get( self, key ):
        return self.data[key]
    
    def normalize_lla( self, x, y, z, skip_elevation ):

        xnrmo = self.get( Tag.XNRMO )
        ynrmo = self.get( Tag.YNRMO )
        znrmo = self.get( Tag.ZNRMO )

        xnrmsf = self.get( Tag.XNRMSF )
        ynrmsf = self.get( Tag.YNRMSF )
        znrmsf = self.get( Tag.ZNRMSF )

        res = np.array( [ ( x - xnrmo ) / xnrmsf,
                          ( y - ynrmo ) / ynrmsf,
                          ( z - znrmo ) / znrmsf ] )
        if skip_elevation:
            res[2] = -znrmo / znrmsf


    @staticmethod
    def defaults():
        pass
