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

class RSMDCA_TAG(Enum):

    IID      =  1
    EDITION  =  2
    TID      =  3
    NPAR     =  4
    NIMGE    =  5
    NPART    =  6
    IIDI     =  7
    NPARI    =  8
    XUOL     =  9
    YUOL     = 10
    ZUOL     = 11
    XUXL     = 12
    XUYL     = 13
    XUZL     = 14
    YUXL     = 15
    YUYL     = 16
    YUZL     = 17
    ZUXL     = 18
    ZUYL     = 19
    ZUZL     = 20
    IRO      = 21
    IRX      = 22
    IRY      = 23
    IRZ      = 24
    IRXX     = 25
    IRXY     = 26
    IRXZ     = 27
    IRYY     = 28
    IRYZ     = 29
    IRZZ     = 30
    ICO      = 31
    ICX      = 32
    ICY      = 33
    ICZ      = 34
    ICXX     = 35
    ICXY     = 36
    ICXZ     = 37
    ICYY     = 38
    ICYZ     = 39
    ICZZ     = 40
    GXO      = 41 
    GYO      = 42
    GZO      = 43
    GXR      = 44
    GYR      = 45
    GZR      = 46
    GS       = 47
    GXX      = 48
    GXY      = 49
    GXZ      = 50
    GYX      = 51
    GYY      = 52
    GYZ      = 53
    GZX      = 54
    GZY      = 55
    GZZ      = 56
    DERCOV   = 57

class RSMDCA(TRE_Base):

    def __init__(self):
        pass
