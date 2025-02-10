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

#  Numpy Libraries
import numpy as np
import numpy.typing as npt

class GCP:

    def __init__( self, id: int, pixel: np.ndarray, coordinate: np.ndarray ):

        self.id         = id
        self.pixel      = np.copy( pixel )
        self.coordinate = np.copy( coordinate )


    def __str__(self):
        return f'GCP: id: {self.id}, pixel: {self.pixel}, coordinate: {self.coordinate}'
    