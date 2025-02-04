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

class Rectangle:
    '''
    Stores a rectangle which can be expanded by adding points. 
    Stores the mean of points added to it.
    '''

    def __init__( self, point, size = None ):

        self.min_pnt = point

        if size is None:
            self.size = np.zeros( len(point), np.float64 )
        else:
            self.size = size

        self.sum_pnt = np.copy(point)
        self.cnt_pnt = 1
        
    def min_point(self):
        return self.min_pnt

    def max_point(self):
        return self.min_pnt + self.size

    def mean_point(self):
        return self.sum_pnt / self.cnt_pnt
    
    def add_point(self, new_point ):

        min_delta = new_point - self.min_point()
        max_delta = new_point - self.max_point()

        self.sum_pnt += new_point
        self.cnt_pnt += 1

        for x in range( 0, len( new_point ) ):
            if min_delta[x] < 0:
                self.min_pnt[x] = new_point[x]
                self.size[x]   += abs(min_delta[x])
            elif max_delta[x] > 0:
                self.size[x] += abs(max_delta[x])

    def inside( self, pt ):
        '''
        Check if a point is inside the rectangle.
        '''
        if len(pt) != len(self.min_pnt):
            raise Exception( f'Rectangle has dimensions of {len(self.min_pnt)} but input has dimensions of {len(pt)}')

        for idx in range( 0, len(self.min_pnt)):
            if pt[idx] < self.min_point()[idx]:
                return False
            if pt[idx] > self.max_point()[idx]:
                return False
        return True
            

    def __str__(self):
        output  =  'Rectangle:\n'
        output += f' - min_point: {self.min_point()}\n'
        output += f' - max_point: {self.max_point()}\n'
        output += f' - sum_point: {self.sum_pnt}'
        return output
        
    @staticmethod
    def from_minmax( min_x, min_y, max_x, max_y ):
        
        minp = np.array( [ min_x, min_y ], dtype = np.float64 )
        size = np.array( [ max_x - min_x, max_y - min_y ], dtype = np.float64 )

        return Rectangle( minp, size )