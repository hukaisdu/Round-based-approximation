import numpy as np
N = 257

np.set_printoptions(
    threshold=np.inf,
    linewidth=np.inf)

def Rot( x, t ):
    return ( (x << t) ^ (x >> ( N - t )) ) & (2 ** N - 1)

def rotr( x, t ):
    return ( ( x >> t ) | ( x << (N - t) ) ) & ( 2**N -1 )

def Chi( x ):
    return x ^ Rot((x ^ (2**N-1)), 1 ) & Rot( x, 2 )

M00 = 0.5 * np.array( [[1, 1],[1,1]], dtype = np.float64 )
M01 = 0.5 * np.array( [[1, -1],[1, -1]], dtype = np.float64 )
M10 = 0.5 * np.array( [[1, 1],[-1,1]], dtype = np.float64 )
M11 = 0.5 * np.array( [[1, -1],[-1,-1]], dtype = np.float64 )

MT = [ M00, M01, M10, M11 ]

def dot( a, x, N ):
    z = a & x
    res = 0
    for i in range(N):
        res ^= z >> i & 1
    return res

def fwt(v):
    n = len(v)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                x = v[j]
                y = v[j + h]
                v[j] = x + y
                v[j + h] = x - y
        h <<= 1  

def Mj(v0, v1, u0, u1, j):
    U0 = ( rotr( v0 ^ v1, 1 ) ) >> ( N - 1 - j ) & 0x1
    U1 = ( u0 ^ u1 ^ v0 ^ v1 ) >>  ( N - 1 - j ) & 0x1
    M0 = MT[ U0 * 2 + U1 ]
    V0 = rotr( v1, 1 ) >> ( N - 1 - j ) & 0x1
    V1 = (u1 ^ v1) >> ( N - 1 - j) & 0x1
    M1 = MT[ V0 * 2 + V1 ]
    MM = np.kron( M0, M1 )
    return MM

def getUV( v0, v1, u0, u1 ):
    vec0 = np.array( [1,0], dtype = np.float64 )
    vec1 = np.array( [0,1], dtype = np.float64 )
    IM = np.identity( 4, dtype = np.float64 )
    for j in range( 0, N ):
        IM = Mj(v0, v1, u0, u1, j )@IM

    IM1 = np.kron( vec0, vec0 ) @IM@ np.kron( vec0, vec0 )
    IM2 = np.kron( vec0, vec1 ) @IM@ np.kron( vec0, vec1 )
    IM3 = np.kron( vec1, vec0 ) @IM@ np.kron( vec1, vec0 )
    IM4 = np.kron( vec1, vec1 ) @IM@ np.kron( vec1, vec1 )

    return IM1 + IM2 + IM3 + IM4

def getLinearMatrix():
    M = np.zeros( (N, N), dtype = np.int64 )
    for unit in range( N ):
        x = np.zeros( N, dtype = np.int8 )
        x[unit] = 1
        # pi function
        y = np.zeros( N, dtype = np.int8 )
        for i in range( N ):
            y[i] = x[ (121 * i) % N ]
        # theta
        for i in range( N ):
            x[i] = y[i] ^ y[(i+3) % N] ^ y[ (i + 10) % N]
        M[:, unit] = x[:]
    return M

def passLinear( gamma ):
    M = getLinearMatrix()
    sigma = np.zeros( (N, 4), dtype = np.float64 )
    for ii in range( N ):
        for v in range( 4 ):
            v0 = v >> 1 & 0x1
            v1 = v & 0x1
            vec0 = np.zeros( N, dtype = np.int8 )
            vec0[ii] = v0
            vec1 = np.zeros( N, dtype = np.int8 )
            vec1[ii] = v1
            u0 = M.T @ vec0
            u1 = M.T @ vec1
            cor = 1
            for jj in range( N ):
                cor *= gamma[jj,  u0[jj] * 2 + u1[jj] ]
            sigma[ii, v] = cor
    return sigma

def passChi( gamma ): # gamma
    vec0 = np.array( [1,0], dtype = np.float64 )
    vec1 = np.array( [0,1], dtype = np.float64 )

    sigma = np.zeros( (N, 4), dtype = np.float64 )

    for ii in range( N ):
        for v0 in range(2):
            for v1 in range(2):
                #V0 = np.array( [ 0 for i in range( N ) ], dtype = np.int8 )
                #V1 = np.array( [ 0 for i in range( N ) ], dtype = np.int8 )
                #V0[ii] = v0
                #V1[ii] = v1
                V0 = v0 << ( N - 1 - ii )
                V1 = v1 << ( N - 1 - ii )

                # [ (2 * v0 + v1), 0, 0, 0, ... ]

                T = np.identity( 4, dtype = np.float64 )

                for jj in range(N):
                    temp = np.zeros( (4, 4), dtype = np.float64 )
                    for u0 in range(2):
                        for u1 in range(2):
                            U0 = u0 << ( N - 1 - jj )
                            U1 = u1 << ( N - 1 - jj)
                            temp += gamma[jj][ u0 *2 + u1 ] * Mj(V0, V1, U0, U1, jj)
                    T = temp @ T
                #print( T )
                #input()

                sigma[ii, v0 * 2 + v1] = ( np.kron( vec0, vec0 ) @ T @ np.kron( vec0, vec0 ) ) + \
                                         ( np.kron( vec0, vec1 ) @ T @ np.kron( vec0, vec1 ) ) + \
                                         ( np.kron( vec1, vec0 ) @ T @ np.kron( vec1, vec0 ) ) + \
                                         ( np.kron( vec1, vec1 ) @ T @ np.kron( vec1, vec1 ) )
    return sigma

def getBias( ROUND, Diff ):

    GAMMA = np.zeros( (N, 4), dtype = np.float64 )

    for opt in range( 8 ):
        opt0 = opt>>2 & 0x1
        opt1 = opt>>1 & 0x1
        opt2 = opt>>0 & 0x1

        gamma = np.zeros( (N, 4), dtype = np.float64 )
        for i in range(N):
            if i == 0: 
                gamma[i, 2 * opt0 + Diff[i] ] = 1
                fwt( gamma[i] )
            elif i == 1:
                gamma[i, 2 * opt1 + Diff[i] ] = 1
                fwt( gamma[i] )
            elif i == 256:
                gamma[i, 2 * opt2 + Diff[i] ] = 1
                fwt( gamma[i] )
            else:
                for j in range(2):
                    gamma[i, 2 * j + Diff[i] ] = 1
                fwt( gamma[i] )
                gamma[i] = gamma[i] / 2

        for r in range( ROUND ):
            gamma = passChi( gamma )

            maxv = 0
            for i in range(N):
                #print( i, gamma[i, 1], np.log2( abs( gamma[i, 1] ) )   )
                if abs( gamma[i, 1 ] ) > maxv:
                    maxv = abs( gamma[i, 1 ] )
            print( "Round %d" % r )
            print( maxv )

            if r < ROUND - 1:
                gamma = passLinear( gamma )

        GAMMA += gamma

    return GAMMA/8

if __name__ == '__main__':
    Diff = np.zeros( N, dtype = np.int8 )
    Diff[0] = 1
    gamma = getBias( 5, Diff )

    for i in range(N):
        print( i, gamma[i] )
