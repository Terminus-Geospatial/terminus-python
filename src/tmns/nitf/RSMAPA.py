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

class RSMAPA_TAG(Enum):

    IID      =  1
    EDITION  =  2
    TID      =  3
    NPAR     =  4
    XUOL     =  5
    YUOL     =  6
    ZUOL     =  7
    XUXL     =  8
    XUYL     =  9
    XUZL     = 10
    YUXL     = 11
    YUYL     = 12
    YUZL     = 13
    ZUXL     = 14
    ZUYL     = 15
    ZUZL     = 16
    IRO      = 17
    IRX      = 18
    IRY      = 19
    IRZ      = 20
    IRXX     = 21
    IRXY     = 22
    IRXZ     = 23
    IRYY     = 24
    IRYZ     = 25
    IRZZ     = 26
    ICO      = 27
    ICX      = 28
    ICY      = 29
    ICZ      = 30
    ICXX     = 31
    ICXY     = 32
    ICXZ     = 33
    ICYY     = 34
    ICYZ     = 35
    ICZZ     = 36
    GXO      = 37
    GYO      = 38
    GZO      = 39
    GXR      = 40
    GYR      = 41
    GZR      = 42
    GS       = 43
    GXX      = 44
    GXY      = 45
    GXZ      = 46
    GYX      = 47
    GYY      = 48
    GYZ      = 49
    GZX      = 50
    GZY      = 51
    GZZ      = 52
    PARVAL   = 53


class RSMAPA(TRE_Base):

    def __init__(self):
        pass
