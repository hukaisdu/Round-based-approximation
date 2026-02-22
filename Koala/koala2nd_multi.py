import numpy as np
import multiprocessing as mp
from functools import partial
import itertools

N = 257
THREAD = 16

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


def process_index(args, gamma, M, N):
    ii, gamma, M, N = args
    local_sigma = np.zeros(16, dtype=np.float64)
    for v in range(16):
        v0 = v >> 3 & 0x1
        v1 = v >> 2 & 0x1
        v2 = v >> 1 & 0x1
        v3 = v >> 0 & 0x1

        vec0 = np.zeros(N, dtype=np.int8)
        vec0[ii] = v0

        vec1 = np.zeros(N, dtype=np.int8)
        vec1[ii] = v1

        vec2 = np.zeros(N, dtype=np.int8)
        vec2[ii] = v2

        vec3 = np.zeros(N, dtype=np.int8)
        vec3[ii] = v3

        u0 = M.T @ vec0
        u1 = M.T @ vec1
        u2 = M.T @ vec2
        u3 = M.T @ vec3

        cor = 1
        for jj in range(N):
            cor *= gamma[jj, u0[jj] * 8 + u1[jj] * 4 + u2[jj] * 2 + u3[jj]]

        local_sigma[v] = cor
    return ii, local_sigma

def passLinear_multiprocess(gamma, num_processes=THREAD):
    M = getLinearMatrix()
    N = gamma.shape[0]  
    sigma = np.zeros((N, 16), dtype=np.float64)

    with mp.Pool(processes=num_processes) as pool:
        args_list = [(ii, gamma, M, N) for ii in range(N)]

        process_func = partial(process_index, gamma=gamma, M=M, N=N)

        results = pool.map_async(process_func, [(ii, gamma, M, N) for ii in range(N)])

        for ii, result in results.get():
            sigma[ii, :] = result

    return sigma

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

def process_single_ii(args, gamma, N, Mj2nd_func):
    ii, gamma, N, Mj2nd_func = args
    vec0 = np.array([1, 0], dtype=np.float64)
    vec1 = np.array([0, 1], dtype=np.float64)
    VEC = [vec0, vec1]

    sigma_ii = np.zeros(16, dtype=np.float64)

    v_combinations = list(itertools.product(range(2), repeat=4))

    for v0, v1, v2, v3 in v_combinations:
        V0 = v0 << (N - 1 - ii)
        V1 = v1 << (N - 1 - ii)
        V2 = v2 << (N - 1 - ii)
        V3 = v3 << (N - 1 - ii)

        T = np.identity(16, dtype=np.float64)

        for jj in range(N):
            temp = np.zeros((16, 16), dtype=np.float64)

            u_combinations = list(itertools.product(range(2), repeat=4))

            for u0, u1, u2, u3 in u_combinations:
                U0 = u0 << (N - 1 - jj)
                U1 = u1 << (N - 1 - jj)
                U2 = u2 << (N - 1 - jj)
                U3 = u3 << (N - 1 - jj)

                temp += gamma[jj][u0 * 8 + u1 * 4 + u2 * 2 + u3] * Mj2nd_func(V0, V1, V2, V3, U0, U1, U2, U3, jj)

            T = temp @ T

        sigma_ii_val = 0

        i_combinations = list(itertools.product(range(2), repeat=4))

        for i0, i1, i2, i3 in i_combinations:
            E = np.kron(np.kron(np.kron(VEC[i0], VEC[i1]), VEC[i2]), VEC[i3])
            sigma_ii_val += E @ T @ E

        sigma_ii[v0 * 8 + v1 * 4 + v2 * 2 + v3] = sigma_ii_val

    return ii, sigma_ii

def passChi_multiprocess(gamma, num_processes=THREAD):
    N = gamma.shape[0]  
    sigma = np.zeros((N, 16), dtype=np.float64)

    if num_processes is None:
        num_processes = mp.cpu_count()

    args_list = [(ii, gamma, N, Mj2nd) for ii in range(N)]

    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_single_ii, gamma=gamma, N=N, Mj2nd_func=Mj2nd)

        results = pool.map(process_func, args_list)

        for ii, result in results:
            sigma[ii, :] = result

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

    for r in range( ROUND ):
        gamma = passChi_multiprocess( gamma )
        if r < ROUND - 1:
            gamma = passLinear_multiprocess( gamma )

    return gamma

def getBias_Opt( ROUND, Diff1, Diff2 ):

    GAMMA = np.zeros( (ROUND, N, 16), dtype = np.float64 )

    for value in range( 64 ):
        print ( value )
        v0 = value >> 5 & 0x1
        v1 = value >> 4 & 0x1
        v2 = value >> 3 & 0x1
        v3 = value >> 2 & 0x1
        v4 = value >> 1 & 0x1
        v5 = value >> 0 & 0x1

        gamma = np.zeros( (N, 16), dtype = np.float64 )

        for i in range(N):
            if i == Diff1[0]:
                gamma[i, v0 * 8 + Diff1[1] * 4 + Diff1[1] ] = 1
                fwt( gamma[i] )
            elif i == Diff1[0] + 1:
                gamma[i, v1 * 8 ] = 1
                fwt( gamma[i] )
            elif i == ( Diff1[0] + N - 1 ) % N:
                gamma[i, v2 * 8 ] = 1
                fwt( gamma[i] )
            elif i == Diff2[0]:
                gamma[i, v3 * 8 + Diff2[1] * 2 + Diff2[1] ] = 1
                fwt( gamma[i] )
            elif i == Diff2[0] + 1:
                gamma[i, v4 * 8 ] = 1
                fwt( gamma[i] )
            elif i == ( Diff2[0] + N - 1 ) % N:
                gamma[i, v5 * 8 ] = 1
                fwt( gamma[i] )
            else:
                for j in range(2):
                    gamma[i, 8 * j + 0 ] = 1
                fwt( gamma[i] )
                gamma[i] = gamma[i] / 2

        for r in range( ROUND ):
            gamma = passChi_multiprocess( gamma )
            if r == 5:
                print( value, gamma[8, 1 * 4 + 1 * 2 + 1 ] )

            GAMMA[r] += gamma
            if r < ROUND - 1:
                gamma = passLinear_multiprocess( gamma )

    GAMMA /= 64

    return GAMMA

if __name__ == '__main__':
    Diff1 = (0, 1)
    Diff2 = (11, 1)

    ROUND = 5

    print( Diff1, Diff2 )

    gamma = getBias( ROUND, Diff1, Diff2 )

    maxv = 0
    real = 0
    I = -1
    for i in range(N):
        print( f'( {i}, {100 * gamma[i, 1 * 4 + 1 * 2 + 1 ] }'  ) 
        if abs( gamma[i, 1 * 4 + 1 * 2 + 1 ] ) > maxv:
            maxv = abs( gamma[i, 1 * 4 + 1 * 2 + 1 ] )
            real = gamma[i, 1 * 4 + 1 * 2 + 1 ]
            I = i
    print( f'Index: {I}, Max: {real}'  )


    #task = [ ( Diff1, (i, 1) ) for i in range(1, N) ]
    #print( task )

    #getBias( 4, Diff1, Diff2 )

    #with multiprocessing.Pool(processes=128) as pool:
    #    results = pool.starmap( getBias, task )

    #for i in range( N ):
    #    for v in range( 1, 2 ):
    #        print( gamma[i, v * 4 + v * 2 + v] )
