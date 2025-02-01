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


def interp_index( val, eps = 0.0001 ):

    #  Check if close to int value    
    int_val = int(round( val ))
    if abs( int_val - val ) < eps:
        return [int_val]
    
    if int_val < val:
        return [ int_val, int_val + 1]
    else:
        return [ int_val - 1, int_val ]
    

def interp_index_nd( vals, eps = 0.0001 ):

    indices = []

    #  Create the indecies we need to iterate over
    for x in range( len( vals ) ):
        indices.append( interp_index( vals[x], eps ) )

    #  Prime the first index
    pixels = [ [x] for x in indices[0] ]

    #  Construct combinations of indices
    #  - First iterate over each dimension
    for idx in range( 1, len( indices ) ):
        temp_pixels = []

        #  - Second, iterate over each value in this dimension
        for p_idx in range( len( indices[idx] ) ):
            
            dim_value = indices[idx][p_idx]

            for f_idx in range( len( pixels ) ):
                new_pix = list(pixels[f_idx])
                new_pix.append( dim_value )
                temp_pixels.append( new_pix )
                
        pixels = temp_pixels
    
    return pixels

def interp_image( image, dx, dy, p00, p10, p01, p11 ):
    
    p00 = image[p00[1],p00[0]]
    p10 = image[p10[1],p10[0]]
    p01 = image[p01[1],p01[0]]
    p11 = image[p11[1],p11[0]]

    dx1 = p00 * (1-dx) + p01 * dx
    dx2 = p10 * (1-dx) + p11 * dx
    
    dy = dx1 * (1-dy) + dx2 * (dy)

    return dy

