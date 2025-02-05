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

class RSMECA_TAG(Enum):

    IID      =  1
    EDITION  =  2
    TID      =  3
    INCLIC   =  4
    INCLUC   =  5
    NPAR     =  6
    NPARO    =  7
    IGN      =  8
    CVDATE   =  9
    XUOL     = 10
    YUOL     = 11
    ZUOL     = 12
    XUXL     = 13
    XUYL     = 14
    XUZL     = 15
    YUXL     = 16
    YUYL     = 17
    YUZL     = 18
    ZUXL     = 19
    ZUYL     = 20
    ZUZL     = 21
    IRO      = 22
    IRX      = 23
    IRY      = 24
    IRZ      = 25
    IRXX     = 26
    IRXY     = 27
    IRXZ     = 28
    IRYY     = 29
    IRYZ     = 30
    IRZZ     = 31
    ICO      = 32
    ICX      = 33
    ICY      = 34
    ICZ      = 35
    ICXX     = 36
    ICXY     = 37
    ICXZ     = 38
    ICYY     = 39
    ICYZ     = 40
    ICZZ     = 41
    GXO      = 42 
    GYO      = 43
    GZO      = 44
    GXR      = 45
    GYR      = 46
    GZR      = 47
    GS       = 48
    GXX      = 49
    GXY      = 50
    GXZ      = 51
    GYX      = 52
    GYY      = 53
    GYZ      = 54
    GZX      = 55
    GZY      = 56
    GZZ      = 57
    NUMOPG   = 58
    ERRCVG   = 59
    TCDF     = 60
    NCSEG    = 61
    CORSEG   = 62
    TAUSEG   = 63
    MAP      = 64
    URR      = 65
    URC      = 66
    UCC      = 67
    UNCSR    = 68
    UCORSR   = 69
    UTAUSR   = 70
    UNCSC    = 71
    UCORSC   = 72
    UTAUSC   = 73



class RSMECA(TRE_Base):

    def __init__(self):
        pass
