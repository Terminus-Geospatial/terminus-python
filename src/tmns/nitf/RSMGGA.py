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

#  Terminus Libraries
from tmns.nitf.base import TRE_Base

class RSMGGA_TAG(Enum):

    IID      =  1
    EDITION  =  2
    GGRSN    =  3
    GGCSN    =  4
    GGRFEP   =  5
    GGCFEP   =  6
    INTORD   =  7
    NPLN     =  8
    DELTAZ   =  9
    DELTAX   = 10
    DELTAY   = 11
    ZPLN1    = 12
    XIPLN1   = 13
    YIPLN1   = 14
    REFROW   = 15
    REFCOL   = 16
    TNUMRD   = 17
    TNUMCD   = 18
    FNUMRD   = 19
    FNUMCD   = 20
    IXO      = 21
    IYO      = 22
    NXPTS    = 23
    NYPTS    = 24
    RCOORD   = 25
    CCOORD   = 26



class RSMGGA(TRE_Base):

    def __init__(self):
        pass
