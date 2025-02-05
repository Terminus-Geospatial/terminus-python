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
import datetime
from enum import Enum
import math

#  Terminus Libraries
from tmns.nitf.base import TRE_Base

class Tag(Enum):

    IID      =  1
    EDITION  =  2
    ISID     =  3
    SID      =  4
    STID     =  5
    YEAR     =  6
    MONTH    =  7
    DAY      =  8
    HOUR     =  9
    MINUTE   = 10
    SECOND   = 11
    NRG      = 12
    NCG      = 13
    TRG      = 14
    TCG      = 15
    GRNDD    = 16
    XUOR     = 17
    YUOR     = 18
    ZUOR     = 19
    XUXR     = 20
    XUYR     = 21
    XUZR     = 22
    YUXR     = 23
    YUYR     = 24
    YUZR     = 25
    ZUXR     = 26
    ZUYR     = 27
    ZUZR     = 28
    V1X      = 29
    V1Y      = 30
    V1Z      = 31
    V2X      = 32
    V2Y      = 33
    V2Z      = 34
    V3X      = 35
    V3Y      = 36
    V3Z      = 37
    V4X      = 38
    V4Y      = 39
    V4Z      = 40
    V5X      = 41
    V5Y      = 42
    V5Z      = 43
    V6X      = 44
    V6Y      = 45
    V6Z      = 46
    V7X      = 47
    V7Y      = 48
    V7Z      = 49
    V8X      = 50
    V8Y      = 51
    V8Z      = 52
    GRPX     = 53
    GRPY     = 54
    GRPZ     = 55
    FULLR    = 56
    FULLC    = 57
    MINR     = 58
    MAXR     = 59
    MINC     = 60
    MAXC     = 61
    IE0      = 62
    IER      = 63
    IEC      = 64
    IERR     = 65
    IERC     = 66
    IECC     = 67
    IA0      = 68
    IAR      = 69
    IAC      = 70
    IARR     = 71
    IARC     = 72
    IACC     = 73
    SPX      = 74
    SVX      = 75
    SAX      = 76
    SPY      = 77  # Sensor Position Y
    SVY      = 78  # Sensor Velocity Y
    SAY      = 79  # Sensor Accelleration Y
    SPZ      = 80  # Sensor Position Z
    SVZ      = 81  # Sensor Velocity Z
    SAZ      = 82  # Sensor Accelleration Z


class RSMIDA(TRE_Base):

    def __init__(self):
        
        self.data = RSMIDA.defaults()


    @staticmethod
    def defaults():

        dt = datetime.datetime.now()

        return { Tag.IID:     '',         Tag.EDITION: '',
                 Tag.ISID:    '',         Tag.SID:     '',
                 Tag.STID:    '',         Tag.YEAR:    dt.year,
                 Tag.MONTH:   dt.month,   Tag.DAY:     dt.day,
                 Tag.HOUR:    dt.hour,    Tag.MINUTE:  dt.minute,
                 Tag.SECOND:  dt.second,  Tag.NRG:     None,
                 Tag.NCG:     None,       Tag.TRG:     None,
                 Tag.TCG:     None,       Tag.GRNDD:   None,
                 Tag.XUOR:    None,       Tag.YUOR:    None,
                 Tag.ZUOR:    None,       Tag.XUXR:    None,
                 Tag.XUYR:    None,       Tag.XYZR:    None,
                 Tag.YUXR:    None,       Tag.YUYR:    None,
                 Tag.YYZR:    None,       Tag.ZUXR:    None,
                 Tag.ZUYR:    None,       Tag.ZYZR:    None,
                 Tag.V1X:     None,       Tag.V1Y:     None,
                 Tag.V1Z:     None,       Tag.V2X:     None,
                 Tag.V2Y:     None,       Tag.V2Z:     None,
                 Tag.V3X:     None,       Tag.V3Y:     None,
                 Tag.V3Z:     None,
                 Tag.V4X:     None,
                 Tag.V4Y:     None,
                 Tag.V4Z:     None,
                 Tag.V5X:     None,
                 Tag.V5Y:     None,
                 Tag.V5Z:     None,
                 Tag.V6X:     None,
                 Tag.V6Y:     None,
                 Tag.V6Z:     None,
                 Tag.V7X:     None,
                 Tag.V7Y:     None,
                 Tag.V7Z:     None,
                 Tag.V8X:     None,
                 Tag.V8Y:     None,
                 Tag.V8Z:     None,
                 Tag.GRPX:    None,
                 Tag.GRPY:    None,
                 Tag.GRPZ:    None,
                 Tag.FULLR:   0,
                 Tag.FULLC:   0,
                 Tag.MINR:    0,
                 Tag.MAXR:    0,
                 Tag.MINC:    0,
                 Tag.MAXC:    0,
                 Tag.IE0:     None,
                 Tag.IER:     None,
                 Tag.IEC:     None,
                 Tag.IERR:    None,
                 Tag.IERC:    None,
                 Tag.IECC:    None,
                 Tag.IA0:     None,
                 Tag.IAR:     None,
                 Tag.IAC:     None,
                 Tag.IARR:    None,
                 Tag.IARC:    None,
                 Tag.IACC:    None,
                 Tag.SPX:     None,
                 Tag.SVX:     None,
                 Tag.SAX:     None,
                 Tag.SPY:     None,
                 Tag.SVY:     None,
                 Tag.SAY:     None,
                 Tag.SPZ:     None,
                 Tag.SVZ:     None,
                 Tag.SAZ:     None }