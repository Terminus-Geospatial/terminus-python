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

class RSMAPB_TAG(Enum):

    IID      =  1
    EDITION  =  2
    TID      =  3
    NPAR     =  4
    APTYP    =  5
    LOCTYP   =  6
    NSFX     =  7
    NSFY     =  8
    NSFZ     =  9
    NOFFX    = 10
    NOFFY    = 11
    NOFFZ    = 12
    XUOL     = 13
    YUOL     = 14
    ZUOL     = 15
    XUXL     = 16
    XUYL     = 17
    XUZL     = 18
    YUXL     = 19
    YUYL     = 20
    YUZL     = 21
    ZUXL     = 22
    ZUYL     = 23
    ZUZL     = 24
    APBASE   = 25
    NISAP    = 26
    NISAPR   = 27
    XPWRR    = 28
    YPWRR    = 29
    ZPWRR    = 30
    NISAPC   = 31
    NISAPC   = 32
    XPWRC    = 33
    YPWRC    = 34
    ZPWRC    = 35
    NGSAP    = 36
    GSAPID   = 37
    NBASIS   = 38
    AEL      = 39
    PARVAL   = 40


class RSMAPB(TRE_Base):

    def __init__(self):
        pass
