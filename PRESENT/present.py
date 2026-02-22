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


def getBias( ROUND, LAT, Diff, Mask ):
    '''
    Diff = (x, d), x is the position of Sbox, d is the difference
    '''
    # prepare the input coordinates
    X = np.zeros( (16, 16), dtype = np.float64 )
    for i in range( 16 ):
        for j in range( len( Diff ) ):
            if i == Diff[j][0]:
                X[i, Diff[j][1]] = 1 
                break
        else:
            X[i, 0] = 1

        #print( i, X[i] )
        
        #print( i, X[i] )

        #input()

        fwt( X[i] )

    
    for r in range(ROUND):
        # pass the sbox layer
        for i in range( 16 ):
            X[i] = LAT**2 @ X[i]

        if r < ROUND - 1:
            Y = np.copy( X )
            # pass the linear layer
            for i in range( 16 ):
                for v in range( 16 ):
                    V = np.zeros( 64, dtype = np.int8 )
                    for j in range(4):
                        if v >> (3 - j) & 0x1:
                            V[ 4 * i + j ] = 1

                    U = passInvP( V )

                    cor = 1
                    for j in range(16):
                        u = 8 * U[4 * j] + 4 * U[4 * j + 1] + 2 * U[4 * j + 2] + U[4 * j + 3]
                        cor *= Y[j,u]
                    X[i,v] = cor

    COR = 1
    for j in range( len(Mask) ):
        #print( Mask[j][0], Mask[j][1] )
        COR *= X[ Mask[j][0], Mask[j][1] ]

    return COR

def getBias_Opt2( ROUND, LAT, Diff, Mask ):
    '''
    Diff = (x, d), x is the position of Sbox, d is the difference
    '''
    # prepare the input coordinates
    X = np.zeros( (16, 16), dtype = np.float64 )
    for i in range( 16 ):
        for j in range( len( Diff ) ):
            if i == Diff[j][0]:
                X[i, Diff[j][1]] = 1 
                break
        else:
            X[i, 0] = 1

        #print( i, X[i] )
        
        #print( i, X[i] )

        #input()

        fwt( X[i] )

    res = 0
    for r in range(ROUND):
        # pass the sbox layer
        for i in range( 16 ):
            X[i] = LAT**2 @ X[i]

        if r < ROUND - 2:
            Y = np.copy( X )
            # pass the linear layer
            for i in range( 16 ):
                for v in range( 16 ):
                    V = np.zeros( 64, dtype = np.int8 )
                    for j in range(4):
                        if v >> (3 - j) & 0x1:
                            V[ 4 * i + j ] = 1

                    U = passInvP( V )

                    cor = 1
                    for j in range(16):
                        u = 8 * U[4 * j] + 4 * U[4 * j + 1] + 2 * U[4 * j + 2] + U[4 * j + 3]
                        cor *= Y[j,u]
                    X[i,v] = cor
            
        if r == ROUND - 2:
            XX = np.zeros( 1 << 12, dtype = np.float64 )
            Y = np.copy( X )
            # pass the linear layer
            for v in range( 1 << (4 * len(Mask)) ):
                V = np.zeros( 64, dtype = np.int8 )
                for j in range( len( Mask ) ):
                    v0 = v >> (4 * j) & 0xf
                    for k in range( 4 ):
                        V[ 4 * Mask[j][0] + k ] = v0 >> ( 3 - k ) & 0x1
                U = passInvP( V )
                cor = 1
                for j in range(16):
                    u = 8 * U[4 * j] + 4 * U[4 * j + 1] + 2 * U[4 * j + 2] + U[4 * j + 3]
                    cor *= Y[j,u]
                XX[v] = cor
        
            res = np.kron( np.kron( LAT**2, LAT**2 ), LAT**2 ) @ XX 
            break

    return res[ ( Mask[0][1] << 8 ) ^ ( Mask[1][1] << 4 ) ^ ( Mask[2][1] << 0 )  ]

def getBias_Opt3( ROUND, LAT, Diff, Mask ):
    '''
    Diff = (x, d), x is the position of Sbox, d is the difference
    '''
    DDT = genDDT( Sbox, 4 )

    COR = 0

    OD = [ list( range(16 )) for i in range( len(Diff) ) ]

    print( OD )

    for combination_Out in itertools.product(*OD):
        for i in range( len( Diff ) ):
            if DDT[ combination_Out[i], Diff[i][1] ] == 0:
                break
        else:
            #print( 'comb', combination_Out )
            #input()

            # prepare the input coordinates
            X = np.zeros( (16, 16), dtype = np.float64 )

            for i in range( len(Diff) ):
                X[Diff[i][0], combination_Out[i]] = 1
            
            for i in range(16):
                if sum( X[i] ) == 0:
                    X[i, 0] = 1
            
            for i in range(16):
                #print( X[i] )
                fwt( X[i] )
                #print( X[i] )
                #input()
                #print()

            for r in range(ROUND):
                # pass the sbox layer
                if r == 0:
                    pass
                else:
                    for i in range( 16 ):
                        X[i] = LAT**2 @ X[i]

                if r < ROUND - 2:
                    Y = np.copy( X )
                    # pass the linear layer
                    for i in range( 16 ):
                        for v in range( 16 ):
                            V = np.zeros( 64, dtype = np.int8 )
                            for j in range(4):
                                if v >> (3 - j) & 0x1:
                                    V[ 4 * i + j ] = 1

                            U = passInvP( V )

                            cor = 1
                            for j in range(16):
                                u = 8 * U[4 * j] + 4 * U[4 * j + 1] + 2 * U[4 * j + 2] + U[4 * j + 3]
                                cor *= Y[j,u]
                            X[i,v] = cor

            ccc = 1 
            for i in range( len(Diff) ):
                ccc *= DDT[ combination_Out[i], Diff[i][1] ]

            for j in range( len(Mask) ):
                ccc *= X[ Mask[j][0], Mask[j][1] ]
                #input()

            COR += ccc

    return COR
    
def getBias_Opt4( ROUND, LAT, Diff, Mask ):
    '''
    Diff = (x, d), x is the position of Sbox, d is the difference
    '''
    DDT = genDDT( Sbox, 4 )

    COR = 0

    OD = [ list( range(16 )) for i in range( len(Diff) ) ]

    #print( OD )

    for combination_Out in itertools.product(*OD):
        for i in range( len( Diff ) ):
            if DDT[ combination_Out[i], Diff[i][1] ] == 0:
                break
        else:
            #print( 'comb', combination_Out )
            #input()

            # prepare the input coordinates
            X = np.zeros( (16, 16), dtype = np.float64 )

            for i in range( len(Diff) ):
                X[Diff[i][0], combination_Out[i]] = 1
            
            for i in range(16):
                if sum( X[i] ) == 0:
                    X[i, 0] = 1
            
            for i in range(16):
                #print( X[i] )
                fwt( X[i] )
                #print( X[i] )
                #input()
                #print()

            XX = np.zeros( 1 << (4 * len(Mask)), dtype = np.float64 )
            res = 0
            for r in range(ROUND):
                # pass the sbox layer
                if r == 0:
                    pass
                else:
                    for i in range( 16 ):
                        X[i] = LAT**2 @ X[i]

                if r < ROUND - 2:
                    Y = np.copy( X )
                    # pass the linear layer
                    for i in range( 16 ):
                        for v in range( 16 ):
                            V = np.zeros( 64, dtype = np.int8 )
                            for j in range(4):
                                if v >> (3 - j) & 0x1:
                                    V[ 4 * i + j ] = 1

                            U = passInvP( V )

                            cor = 1
                            for j in range(16):
                                u = 8 * U[4 * j] + 4 * U[4 * j + 1] + 2 * U[4 * j + 2] + U[4 * j + 3]
                                cor *= Y[j,u]
                            X[i,v] = cor

                if r == ROUND - 2:
                    Y = np.copy( X )
                    # pass the linear layer
                    for v in range( 1 << (4 * len(Mask)) ):
                        V = np.zeros( 64, dtype = np.int8 )
                        for j in range( len( Mask ) ):
                            v0 = v >> (4 * j) & 0xf
                            for k in range( 4 ):
                                V[ 4 * Mask[j][0] + k ] = v0 >> ( 3 - k ) & 0x1
                        U = passInvP( V )
                        cor = 1
                        for j in range(16):
                            u = 8 * U[4 * j] + 4 * U[4 * j + 1] + 2 * U[4 * j + 2] + U[4 * j + 3]
                            cor *= Y[j,u]
                        XX[v] = cor

                    Mat = LAT**2
                    for ii in range( len(Mask) - 1 ):
                        Mat = np.kron( Mat, LAT**2 )
                    #res = np.kron( np.kron( LAT**2, LAT**2 ), LAT**2 ) @ XX 
                    res = Mat @ XX
                    break

            ccc = 1 
            for i in range( len(Diff) ):
                ccc *= DDT[ combination_Out[i], Diff[i][1] ]
            COR += ccc * res

            #index = 0
            #for j in range( len(Mask) ):
            #    index += Mask[j][1] << (4 * (len(Mask)-1-j))
            #ccc *= res[ index  ]

            #COR += ccc

    return COR

if __name__ == '__main__':
    #for i in range(64):
    #    print( P[ INVP[i] ] )

    LAT = genLAT(Sbox, 4) 
    LAT2 = LAT**2

    '''
    for v in range( 16 ):
        fwt( LAT2[v,:] )
        #fwt( LAT2[v,:] )
    
    print( LAT )

    exit(-1)
    '''


    #Diff = [ [8,7], [11,7] ]
    #Diff = [ [7,2] ]

    #Mask = [ [4, 0xb] ]

    #Diff = [ [1, 9], [3, 0x9]]
    #Diff = [ [7,2] ]

    #Mask = [ [4, 0xb] ]
    #Mask = [ [1, 0xd] ]


    #Diff = [ [12, 0xf], [15, 0xf]]
    #Diff = [ [7,2] ]
    #Mask = [ [4, 0xb] ]
    #Mask = [ [10, 0xd] ]

    #Diff = [ [8,7], [11, 0xf]]
    #Mask = [ [1, 0xf], [9, 15], [13, 15] ]
    #Mask = [ [4, 0xb] ]

    #Diff = [ [0, 9], [3, 9] ]
    #Mask = [ [2, 0xd] ]
    #Mask = [ [1, 0xf], [9, 15], [13, 15] ]

    #bias = getBias_Opt4( 17, LAT, Diff, Mask )


    #for i in range( 1 << 12 ):
    #    print( i, np.log2( bias[i] ) )
    #print( 'Opt 4 ', Diff, Mask, np.log2( np.max( bias[1:] ) ) )

    # 10 round
    #Diff = [ [0, 9], [3, 9] ]
    #Mask = [ [1, 0xd] ]
    Diff = [ [12, 0x7], [15,0x7]]
    Mask = [ [10, 0xe] ]
    bias = getBias_Opt4( 10, LAT, Diff, Mask )
    print( bias[0xe], np.log2( bias[0xe] ) )

    # 11 round
    #Diff = [ [0, 9], [3, 9] ]
    #Mask = [ [1, 0xd] ]
    Diff = [ [8, 0x9], [11,0x9]]
    Mask = [ [4, 0xb] ]
    bias = getBias_Opt4( 11, LAT, Diff, Mask )
    print( bias[0xb], np.log2( bias[0xb] ) )

    # 12 round
    #Diff = [ [0, 9], [3, 9] ]
    #Mask = [ [1, 0xd] ]
    Diff = [ [8, 0x7], [11,0xf]]
    Mask = [ [1, 0xf], [1, 0xf], [1, 0xf] ]
    bias = getBias_Opt4( 12, LAT, Diff, Mask )
    print( bias[0xfff], np.log2( bias[0xfff] ) )

    # 13 round
    Diff = [ [0, 0x9], [3, 0x9]]
    Mask = [ [1, 0xd] ]
    bias = getBias_Opt4( 13, LAT, Diff, Mask )
    print( bias[0xd], np.log2( bias[0xd] ) )

    # 14 round
    print( "Round 14 ")
    Diff = [ [5, 0xd] ] 
    Mask = [ [12, 0xb] ]
    bias = getBias_Opt4( 14, LAT, Diff, Mask )
    #print( 'Opt 4 ', Diff, Mask, bias[0xa], np.log2( abs( bias[1:] ) )  )
    print( bias[0xb], np.log2( bias[0xb] ) )

    print( "Round 15 ")
    Diff = [ [9, 0xd] ] 
    Mask = [ [12, 0xb] ]
    bias = getBias_Opt4( 15, LAT, Diff, Mask )
    print( 'Opt 4 ', Diff, Mask, bias[1:], np.log2( abs( bias[1:] ) )  )
    print( bias[0xb], np.log2( bias[0xb] ) )

    print( "Round 16 ")
    Diff = [ [5, 0xd] ] 
    Mask = [ [12, 0xb] ]
    bias = getBias_Opt4( 16, LAT, Diff, Mask )
    print( 'Opt 4 ', Diff, Mask, bias[1:], np.log2( abs( bias[1:] ) )  )
    print( bias[0xb], np.log2( bias[0xb] ) )

    print( "Round 17 ")
    Diff = [ [5, 0xd] ] 
    Mask = [ [12, 0xb] ]
    bias = getBias_Opt4( 17, LAT, Diff, Mask )
    print( 'Opt 4 ', Diff, Mask, bias[1:], np.log2( abs( bias[1:] ) )  )
    print( bias[0xb], np.log2( bias[0xb] ) )

    print( "Round 18 ")
    Diff = [ [5, 0xd] ] 
    Mask = [ [12, 0xb] ]
    bias = getBias_Opt4( 18, LAT, Diff, Mask )
    print( 'Opt 4 ', Diff, Mask, bias[1:], np.log2( abs( bias[1:] ) )  )
    print( bias[0xb], np.log2( bias[0xb] ) )

    exit(0)


    # 13 round
    #Diff = [ [0, 9], [3, 9] ]
    #Mask = [ [1, 0xd] ]
    Diff = [ [0, 9], [3,9]]
    Mask = [ [1, 0xd] ]
    bias = getBias_Opt4( 13, LAT, Diff, Mask )
    print( 'Opt 4 ', Diff, Mask,  bias[0xd], np.log2( abs( bias[1:] ) )  )


    exit(0)

    maxv = 0
    for diff_pos in range( 0, 16 ):
        for diff_value in range( 1, 16 ):
            for mask_pos in range( 0, 16 ):
                for mask_value in range( 0, 16 ):
                    Diff = [ [ diff_pos, diff_value ] ]
                    Mask = [ [mask_pos, mask_value] ]
                    #Diff = [ diff_pos, diff_value ]
                    bias = getBias_Opt4( 18, LAT, Diff, Mask )
                    #print( bias )
                    print( 'Opt 4 ', Diff,  Mask, np.log2( np.max( bias[1:] ) )  )

                    if np.max( bias[1:] ) > maxv:
                        print( '************************************' )
                        maxv = np.max( bias[1] )
                        print( Diff, Mask )




 
