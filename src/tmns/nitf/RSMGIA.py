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

class RSMGIA_TAG(Enum):

    IID      =  1
    EDITION  =  2
    GR0      =  3
    GRX      =  4
    GRY      =  5
    GRZ      =  6
    GRXX     =  7
    GRXY     =  8
    GRXZ     =  9
    GRYY     = 10
    GRYZ     = 11
    GRZZ     = 12
    GC0      = 13
    GCX      = 14
    GCY      = 15
    GCZ      = 16
    GCXX     = 17
    GCXY     = 18
    GCYY     = 19
    GCYZ     = 20
    GCZZ     = 21
    GRNIS    = 22
    GCNIS    = 23
    GTNIS    = 24
    GRSSIZ   = 25
    GCSSIZ   = 26




class RSMGIA(TRE_Base):

    def __init__(self):
        pass
