import numpy as np
import multiprocessing

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
    """执行快速沃尔什变换（FWT），原地修改向量v"""
    n = len(v)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                # 获取当前分块中的元素对
                x = v[j]
                y = v[j + h]
                # 原位更新值：前一半为和，后一半为差
                v[j] = x + y
                v[j + h] = x - y
        h <<= 1  # 倍增分块大小

def Mj2nd(v0, v1, v2, v3,  u0, u1, u2, u3, j):
    U0 = ( rotr( v0 ^ v1 ^ v2 ^ v3, 1 ) ) >> ( N - 1 - j ) & 0x1
    U1 = ( u0 ^ u1 ^ u2 ^ u3 ^ v0 ^ v1 ^ v2 ^ v3 ) >>  ( N - 1 - j ) & 0x1
    M0 = MT[ U0 * 2 + U1 ]

    V0 = rotr( v1, 1 ) >> ( N - 1 - j ) & 0x1
    V1 = (u1 ^ v1) >> ( N - 1 - j) & 0x1
    M1 = MT[ V0 * 2 + V1 ]

    W0 = rotr( v2, 1 ) >> ( N - 1 - j ) & 0x1
    W1 = (u2 ^ v2) >> ( N - 1 - j) & 0x1
    M2 = MT[ W0 * 2 + W1 ]

    X0 = rotr( v3, 1 ) >> ( N - 1 - j ) & 0x1
    X1 = (u3 ^ v3) >> ( N - 1 - j) & 0x1
    M3 = MT[ X0 * 2 + X1 ]

    MM = np.kron( np.kron( np.kron( M0, M1 ), M2 ), M3 )

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
    sigma = np.zeros( (N, 16), dtype = np.float64 )
    for ii in range( N ):
        #print( ii )
        for v in range( 16 ):
            v0 = v >> 3 & 0x1
            v1 = v >> 2 & 0x1
            v2 = v >> 1 & 0x1
            v3 = v >> 0 & 0x1

            vec0 = np.zeros( N, dtype = np.int8 )
            vec0[ii] = v0

            vec1 = np.zeros( N, dtype = np.int8 )
            vec1[ii] = v1

            vec2 = np.zeros( N, dtype = np.int8 )
            vec2[ii] = v2

            vec3 = np.zeros( N, dtype = np.int8 )
            vec3[ii] = v3

            u0 = M.T @ vec0
            u1 = M.T @ vec1
            u2 = M.T @ vec2
            u3 = M.T @ vec3

            cor = 1
            for jj in range( N ):
                cor *= gamma[ jj,  u0[jj] * 8 + u1[jj] * 4 + u2[jj] * 2 + u3[jj] ]

            sigma[ii, v] = cor

    return sigma

def passChi( gamma ): # gamma
    vec0 = np.array( [1,0], dtype = np.float64 )
    vec1 = np.array( [0,1], dtype = np.float64 )

    VEC = [ vec0, vec1 ]

    sigma = np.zeros( (N, 16), dtype = np.float64 )

    for ii in range( N ):
        #presults = pool.starmap(worker_function, tasks)int( ii )
        for v0 in range(2):
            for v1 in range(2):
                for v2 in range(2):
                    for v3 in range(2):
                        V0 = v0 << ( N - 1 - ii )
                        V1 = v1 << ( N - 1 - ii )
                        V2 = v2 << ( N - 1 - ii )
                        V3 = v3 << ( N - 1 - ii )

                        T = np.identity( 16, dtype = np.float64 )

                        for jj in range(N):
                            temp = np.zeros( (16, 16), dtype = np.float64 )
                            for u0 in range(2):
                                for u1 in range(2):
                                    for u2 in range(2):
                                        for u3 in range(2):
                                            U0 = u0 << ( N - 1 - jj )
                                            U1 = u1 << ( N - 1 - jj )
                                            U2 = u2 << ( N - 1 - jj )
                                            U3 = u3 << ( N - 1 - jj )
                                            temp += gamma[jj][ u0 *8 + u1 * 4 + u2 * 2 + u3 ] * Mj2nd(V0, V1, V2, V3, U0, U1, U2, U3, jj)
                            T = temp @ T

                        sigma[ii, v0 * 8 + v1 * 4 + v2 * 2 + v3] = 0

                        for i0 in range(2):
                            for i1 in range(2):
                                for i2 in range(2):
                                    for i3 in range(2):
                                        E = np.kron( np.kron( np.kron( VEC[i0], VEC[i1] ), VEC[i2] ), VEC[i3] )
                                        sigma[ ii, v0 * 8 + v1 * 4 + v2 * 2 + v3 ] += E @ T @ E
    return sigma

def getBias( ROUND, Diff1, Diff2 ):
    gamma = np.zeros( (N, 16), dtype = np.float64 )
    for i in range(N):
        if i == Diff1[0]:
            for j in range(2):
                gamma[i, 8 * j + Diff1[1] * 4 + Diff1[1] ] = 1
        elif i == Diff2[0]:
            for j in range(2):
                gamma[i, 8 * j + Diff2[1] * 2 + Diff2[1] ] = 1
        else:
            for j in range(2):
                gamma[i, 8 * j + 0 ] = 1

        fwt( gamma[i] )
        gamma[i] = gamma[i] / 2

        #print( gamma[i] )

    for r in range( ROUND ):
        gamma = passChi( gamma )

        #----------------------
        print( "Round %d" % r )
        maxv = 0
        realmaxv = 0
        I = -1

        for i in range(N):
            print( i, gamma[i, 1 * 4 + 1 * 2 + 1 ], np.log2( abs( gamma[i, 1 * 4 + 1 * 2 + 1 ] ) )   )
            if abs( gamma[i, 1 * 4 + 1 * 2 + 1 ] ) > maxv:
                maxv = abs( gamma[i, 1 * 4 + 1 * 2 + 1 ] )
                realmaxv = gamma[i, 1 * 4 + 1 * 2 + 1 ]
                I = i
        print( realmaxv, np.log2( abs( maxv) ), I )


        if r < ROUND - 1:
            gamma = passLinear( gamma )

    return gamma


if __name__ == '__main__':
    #Diff = np.zeros( N, dtype = np.int8 )
    Diff1 = (0, 1)
    Diff2 = (61, 1)

    #task = [ ( Diff1, (i, 1) ) for i in range(1, N) ]
    #print( task )

    getBias( 8, Diff1, Diff2 )

    #with multiprocessing.Pool(processes=128) as pool:
    #    results = pool.starmap( getBias, task )

    #for i in range( N ):
    #    for v in range( 1, 2 ):
    #        print( gamma[i, v * 4 + v * 2 + v] )
