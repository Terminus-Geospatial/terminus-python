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

import numpy as np


class RPC00B:

    def __init__(self):

        self.LINE_OFF: int        = int()
        self.SAMP_OFF: int        = int()

        self.LAT_OFF: float       = float()
        self.LON_OFF: float       = float()
        self.HEIGHT_OFF: float    = float()

        self.LINE_SCALE: float    = float()
        self.SAMP_SCALE: float    = float()
        self.LAT_SCALE: float     = float()
        self.LON_SCALE: float     = float()
        self.HEIGHT_SCALE: float  = float()

        self.LINE_NUM_COEFFs = np.zeros( 20 )
        self.LINE_DEM_COEFFs = np.zeros( 20 )
        self.SAMP_NUM_COEFFs = np.zeros( 20 )
        self.SAMP_DEN_COEFFs = np.zeros( 20 )

    def world_to_pixel( self, lat, lon, height ):

        #  Convert with offsets
        P = (lat    - self.LAT_OFF)    / self.LAT_SCALE
        L = (lon    - self.LON_OFF)    / self.LON_SCALE
        H = (height - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        PLH_vec = np.array( [ 1.0,
                              L,
                              P,
                              H,
                              L * P,
                              L * H,
                              P * H,
                              L ** 2,
                              P ** 2,
                              H ** 2,
                              P * L * H,
                              L ** 3,
                              L * P ** 2,
                              L * H ** 2,
                              L ** 2 * P,
                              P ** 3,
                              P * H ** 2,
                              L ** 2 * H,
                              P ** 2 * H,
                              H ** 3 ], dtype = np.float64 )

        r_n_n = np.dot( self.LINE_NUM_COEFFs, PLH_vec )
        r_n_d = np.dot( self.LINE_DEN_COEFFs, PLH_vec )
        c_n_n = np.dot( self.SAMP_NUM_COEFFs, PLH_vec )
        c_n_d = np.dot( self.SAMP_DEN_COEFFs, PLH_vec )

    def __str__(self):
        
        output  =  'RPC00B:\n'
        output += f'  - Line Offset:      {self.LINE_OFF}\n'
        output += f'  - Sample Offset:    {self.SAMP_OFF}\n'
        output += f'  - Latitude Offset:  {self.LAT_OFF}\n'
        output += f'  - Longitude Offset: {self.LON_OFF}\n'
        output += f'  - Height Offset:    {self.HEIGHT_OFF}\n'
        output += f'  - Line Scale:       {self.LINE_SCALE}\n'
        output += f'  - Sample Scale:     {self.SAMP_SCALE}\n'
        output += f'  - Latitude Scale:   {self.LAT_SCALE}\n'
        output += f'  - Longitude Scale:  {self.LON_SCALE}\n'
        output += f'  - Height Scale:     {self.HEIGHT_SCALE}\n'
        return output
        
        
    @staticmethod
    def from_components( center_pixel,
                         center_lla,
                         image_width,
                         image_height,
                         max_delta_lla ):

        model = RPC00B()
        
        model.SAMP_OFF   = center_pixel[0]
        model.SAMP_SCALE = image_width / 2.0

        model.LINE_OFF   = center_pixel[1]
        model.LINE_SCALE = image_height / 2.0

        model.LON_OFF      = center_lla[0]
        model.LON_SCALE    = max_delta_lla[0]

        model.LON_OFF      = center_lla[1]
        model.LAT_SCALE    = max_delta_lla[1]

        model.HEIGHT_OFF   = center_lla[2]
        model.HEIGHT_SCALE = max_delta_lla[2]
        
        return model

        