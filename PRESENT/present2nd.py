import numpy as np 
import itertools
from tools import *

Sbox = [ 0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2 ]

P = [ 0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,12,28,44,60,13,29,45,61,14,30,46,62,15,31,47, 63 ]
INVP = [ 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 ]

def passInvP ( X ):
    Y = np.zeros( 64, dtype = np.int8 )
    for i in range(64):
        Y[ INVP[i] ] = X[i]
    return Y

def passP ( X ):
    Y = np.zeros( 64, dtype = np.int8 )
    for i in range(64):
        Y[ P[i] ] = X[i]
    return Y

def passSbox2nd( LAT, x ):
    y = np.zeros( 16 * 16 * 16 * 16 )
    for t in range( 16 ):
        for v1 in range( 16 ):
            for u3 in range( 16 ):
                for u2 in range( 16 ):
                    for u1 in range(16 ):
                        #print( t, v1, u3, u2, u1 )
                        y[ (t << 12) + (v1 << 8) + (u3 << 4) + u2 ] += LAT[v1, u1]* LAT[t, u1 ^ u2 ^ u3 ] * x[ (u1 << 8) + (u2 << 4) + u3 ]
    
    z = np.zeros( 16 * 16 * 16 * 16 )
    for t in range( 16 ):
        for v1 in range( 16 ):
            for v2 in range( 16 ):
                for u3 in range( 16 ):
                    for u2 in range(16 ):
                        #print( t, v1, v2, u3, u2 )
                        z[ (t << 12) + (v1 << 8) + (v2 << 4) + u3 ] += LAT[v2, u2] * y[ (t << 12) + (v1 << 8) + (u3 << 4) + u2 ]

    w = np.zeros( 16 * 16 * 16 )
    for t in range( 16 ):
        for v1 in range( 16 ):
            for v2 in range( 16 ):
                v3 = t ^ v1 ^ v2
                for u3 in range(16 ):
                    #print( t, v1, v2, u3 )
                    w[ (v1 << 8) + (v2 << 4) + v3 ] += LAT[v3, u3] * z[ (t << 12) + (v1 << 8) + (v2 << 4) + u3 ]

    return w

def getBias( ROUND, LAT, Diff1, Diff2 ):
    '''
    Diffi = (x, d), x is the position of Sbox, d is the difference
    '''
    # prepare the input coordinates
    X = np.zeros( (16, 16*16*16), dtype = np.float64 )
    for i in range( 16 ):
        if i == Diff1[0]:
            X[ Diff1[0], ( Diff1[1] << 8 ) + Diff1[1] ] = 1
        elif i == Diff2[0]:
            X[ Diff2[0], ( Diff2[1] << 4 ) + Diff2[1] ] = 1
        else:
            X[i, 0] = 1

        fwt( X[i] )

    XX = []
    

    for r in range(ROUND):
        # pass the sbox layer
        for i in range( 16 ):
            X[i] = passSbox2nd( LAT, X[i] )

        XX.append( X )

        if r < ROUND - 1:
            Y = np.copy( X )
            # pass the linear layer
            for i in range( 16 ):
                for v1 in range( 16 ):
                    for v2 in range(16):
                        for v3 in range(16):
                            V1 = np.zeros( 64, dtype = np.int8 )
                            V2 = np.zeros( 64, dtype = np.int8 )
                            V3 = np.zeros( 64, dtype = np.int8 )
                            for j in range(4):
                                if v1 >> (3 - j) & 0x1:
                                    V1[ 4 * i + j ] = 1
                                if v2 >> (3 - j) & 0x1:
                                    V2[ 4 * i + j ] = 1
                                if v3 >> (3 - j) & 0x1:
                                    V3[ 4 * i + j ] = 1
                            U1 = passInvP( V1 )
                            U2 = passInvP( V2 )
                            U3 = passInvP( V3 )

                            cor = 1
                            for j in range(16):
                                u1 = 8 * U1[4 * j] + 4 * U1[4 * j + 1] + 2 * U1[4 * j + 2] + U1[4 * j + 3]
                                u2 = 8 * U2[4 * j] + 4 * U2[4 * j + 1] + 2 * U2[4 * j + 2] + U2[4 * j + 3]
                                u3 = 8 * U3[4 * j] + 4 * U3[4 * j + 1] + 2 * U3[4 * j + 2] + U3[4 * j + 3]
                                cor *= Y[j, (u1 << 8) + (u2 << 4) + u3]

                            X[i, (v1 << 8) + (v2 << 4) + v3 ] = cor
    return XX

if __name__ == '__main__':
    ROUND = 10
    LAT = genLAT(Sbox, 4) 
    Diff1 = (0,0xd)
    Diff2 = (5,0xd)

    bias = getBias( ROUND, LAT, Diff1, Diff2 )

    for r in range( ROUND ):
        maxv = 0
        for i in range(16):
            for v in range(1, 16):
                if abs( bias[r][i, ( v << 8 ) + ( v << 4 ) + v] ) > maxv:
                    maxv = bias[r][i, ( v << 8 ) +( v << 4 ) + v ]
                print( r, i, v, bias[r][ i, (v << 8) + (v << 4) + v ] )
    
        print( r, maxv, np.log2( abs( maxv) ) )
        

