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
import logging
import os
import unittest

#  Terminus Libraries
import tmns.proj.RPC00B as RPC00B
import tmns.proj.RSM 

def load_rpc_file( rpc_path ):

    data = {}
    with open( rpc_path, 'r' ) as fin:
        for line in fin.readlines():
            parts = line.replace(' ','').strip().split(':')

            term = RPC00B.Term.from_str(parts[0])
            value = float(parts[1])
            data[term] = value

    #  Create RPC object from data
    return RPC00B.RPC00B.from_dict( data )

class proj_RSM( unittest.TestCase ):

    def test_planet1(self):

        logger = logging.getLogger( 'test_RSM.planet1' )

        #  Make sure the RPC example file exists
        rpc_path = os.path.realpath( os.path.join( os.path.dirname( __file__ ),
                                                   '../data/20240717_171045_18_24af_1B_AnalyticMS_RPC.TXT' ) )
        
        self.assertTrue( os.path.exists( rpc_path ) )

        #  Load RPC Model
        model = load_rpc_file( rpc_path )

        #  Create GCPs
        


    