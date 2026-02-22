import numpy as np

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

def genDDT( Sbox, N ):
    size = 1 << N  
    DDT = np.zeros( (size, size), dtype = np.float64 )  
    for u in range( size ):
        for x in range( size ):
            DDT[ Sbox[x] ^ Sbox[x ^ u], u ] += 1
    return DDT / 16

def genLAT(Sbox, N):
    size = 1 << N 
    LAT = np.zeros( (size, size), dtype = np.float64 )  
    
    for beta in range(size):
        v = np.zeros( size, dtype = np.float64 )
        for x in range(size):
            d = dot(beta, Sbox[x], N)
            v[x] = 1 if d == 0 else -1
        
        fwt(v)
        
        LAT[beta,:] = v
    return LAT / size

