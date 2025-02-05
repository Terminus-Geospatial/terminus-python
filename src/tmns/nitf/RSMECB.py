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
    NPARO    =  6
    IGN      =  7
    CVDATE   =  8
    NPAR     =  9
    APTYP    = 10
    LOCTYP   = 11
    NSFX     = 12
    NSFY     = 13
    NSFZ     = 14
    NOFFX    = 15
    NOFFY    = 16
    NOFFZ    = 17
    XUOL     = 18
    YUOL     = 19
    ZUOL     = 20
    XUXL     = 21
    XUYL     = 22
    XUZL     = 23
    YUXL     = 24
    YUYL     = 25
    YUZL     = 26
    ZUXL     = 27
    ZUYL     = 28
    ZUZL     = 29
    APBASE   = 30
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
    NUMOPG
    ERRCVG
    TCDF
    ACSMC
    NCSEG
    CORSEG
    TAUSEG
    AC
    ALPC
    BETC
    TC
    MAP
    URR
    URC
    UCC
    UACSMC
    UNCSR
    UCORSR
    UTAUSR
    UNCSC
    UCORSC
    UTAUSC
    UACR
    UALPCR
    UBETCR
    UTCR
    UACC
    UALPCC
    UBETCC
    UTCC



class RSMECB(TRE_Base):

    def __init__(self):
        pass
