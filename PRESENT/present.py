import numpy as np 
import itertools
from functools import lru_cache
from tools import *

Sbox = [ 0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2 ]

P = [ 0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,12,28,44,60,13,29,45,61,14,30,46,62,15,31,47, 63 ]
INVP = [ 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 ]

NIBBLES = 16
NIBBLE_BITS = 4
SBOX_VALUES = 16

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


def _nibble_from_bits(bits, pos):
    return 8 * bits[4 * pos] + 4 * bits[4 * pos + 1] + 2 * bits[4 * pos + 2] + bits[4 * pos + 3]


def _mask_to_bits(mask_pos, mask_value):
    bits = np.zeros(64, dtype=np.int8)
    for bit in range(NIBBLE_BITS):
        bits[NIBBLE_BITS * mask_pos + bit] = (mask_value >> (3 - bit)) & 1
    return bits


@lru_cache(maxsize=1)
def _single_nibble_linear_table():
    table = np.zeros((NIBBLES, SBOX_VALUES, NIBBLES), dtype=np.uint8)
    for mask_pos in range(NIBBLES):
        for mask_value in range(SBOX_VALUES):
            u = passInvP(_mask_to_bits(mask_pos, mask_value))
            for src_pos in range(NIBBLES):
                table[mask_pos, mask_value, src_pos] = _nibble_from_bits(u, src_pos)
    return table


@lru_cache(maxsize=None)
def _selected_nibble_linear_table(mask_positions):
    table = np.zeros((SBOX_VALUES ** len(mask_positions), NIBBLES), dtype=np.uint8)
    for value in range(len(table)):
        bits = np.zeros(64, dtype=np.int8)
        for offset, mask_pos in enumerate(mask_positions):
            mask_value = (value >> (NIBBLE_BITS * offset)) & 0xf
            bits |= _mask_to_bits(mask_pos, mask_value)

        u = passInvP(bits)
        for src_pos in range(NIBBLES):
            table[value, src_pos] = _nibble_from_bits(u, src_pos)
    return table


@lru_cache(maxsize=1)
def _fwt_basis():
    basis = np.eye(SBOX_VALUES, dtype=np.float64)
    for value in range(SBOX_VALUES):
        fwt(basis[value])
    return basis


@lru_cache(maxsize=None)
def _kron_lat2(mask_count, lat2_key):
    lat2 = np.array(lat2_key, dtype=np.float64).reshape(SBOX_VALUES, SBOX_VALUES)
    mat = lat2
    for _ in range(mask_count - 1):
        mat = np.kron(mat, lat2)
    return mat


def _lat2_cache_key(lat2):
    return tuple(np.asarray(lat2, dtype=np.float64).ravel())


@lru_cache(maxsize=1)
def _ddt():
    return genDDT(Sbox, 4)


def _apply_linear_layer(x):
    table = _single_nibble_linear_table()
    src_pos = np.arange(NIBBLES)[None, None, :]
    return np.prod(x[src_pos, table], axis=2)


def _apply_selected_linear_layer(x, mask_positions):
    table = _selected_nibble_linear_table(tuple(mask_positions))
    src_pos = np.arange(NIBBLES)[None, :]
    return np.prod(x[src_pos, table], axis=1)


def _valid_sbox_outputs(ddt, diff):
    choices = []
    for _, diff_value in diff:
        values = [(out_value, ddt[out_value, diff_value]) for out_value in range(SBOX_VALUES) if ddt[out_value, diff_value] != 0]
        choices.append(values)
    return choices


def _initial_x_from_outputs(diff, output_values):
    x = np.empty((NIBBLES, SBOX_VALUES), dtype=np.float64)
    x[:] = _fwt_basis()[0]
    for (pos, _), output_value in zip(diff, output_values):
        x[pos] = _fwt_basis()[output_value]
    return x


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
    ddt = _ddt()
    lat2 = LAT**2
    mask_positions = tuple(mask_pos for mask_pos, _ in Mask)
    result = np.zeros(SBOX_VALUES ** len(Mask), dtype=np.float64)
    final_sbox_layer = _kron_lat2(len(Mask), _lat2_cache_key(lat2))

    output_choices = _valid_sbox_outputs(ddt, Diff)
    for output_terms in itertools.product(*output_choices):
        output_values = [term[0] for term in output_terms]
        ddt_weight = np.prod([term[1] for term in output_terms])

        x = _initial_x_from_outputs(Diff, output_values)
        for r in range(ROUND - 1):
            if r > 0:
                x = lat2 @ x.T
                x = x.T

            if r < ROUND - 2:
                x = _apply_linear_layer(x)
            else:
                linear_values = _apply_selected_linear_layer(x, mask_positions)
                result += ddt_weight * (final_sbox_layer @ linear_values)

    return result


def search_best_differential_linear(rounds, lat, diff_values=None, mask_values=None, verbose=True):
    if diff_values is None:
        diff_values = range(1, SBOX_VALUES)
    if mask_values is None:
        mask_values = range(1, SBOX_VALUES)

    best = {
        "bias": 0.0,
        "log2_abs_bias": -np.inf,
        "diff": None,
        "mask": None,
        "mask_value": None,
    }

    for diff_pos in range(NIBBLES):
        for diff_value in diff_values:
            diff = [[diff_pos, diff_value]]
            for mask_pos in range(NIBBLES):
                mask = [[mask_pos, 0]]
                bias = getBias_Opt4(rounds, lat, diff, mask)
                for mask_value in mask_values:
                    current = abs(bias[mask_value])
                    if current > best["bias"]:
                        log2_current = np.log2(current) if current > 0 else -np.inf
                        best = {
                            "bias": float(current),
                            "log2_abs_bias": float(log2_current),
                            "diff": diff,
                            "mask": [[mask_pos, mask_value]],
                            "mask_value": mask_value,
                        }
                        if verbose:
                            print(
                                "new best",
                                "round", rounds,
                                "Diff", best["diff"],
                                "Mask", best["mask"],
                                "bias", best["bias"],
                                "log2", best["log2_abs_bias"],
                            )

    return best


def _format_pairs(pairs):
    return "[" + ", ".join(f"({pos:2d}, 0x{value:x})" for pos, value in pairs) + "]"


def _result_index(mask):
    index = 0
    for offset, (_, mask_value) in enumerate(mask):
        index |= mask_value << (NIBBLE_BITS * offset)
    return index


def _known_result(rounds, lat, diff, mask):
    bias = getBias_Opt4(rounds, lat, diff, mask)
    value = bias[_result_index(mask)]
    abs_value = abs(value)
    log2_value = np.log2(abs_value) if abs_value > 0 else -np.inf
    return {
        "rounds": rounds,
        "diff": diff,
        "mask": mask,
        "bias": value,
        "log2_abs_bias": log2_value,
    }


def _print_known_results(results):
    print("\nKnown differential-linear results")
    print("-" * 104)
    print(f"{'Round':>5}  {'Diff':<24}  {'Mask':<34}  {'bias':>22}  {'log2(abs)':>12}")
    print("-" * 104)
    for result in results:
        print(
            f"{result['rounds']:>5}  "
            f"{_format_pairs(result['diff']):<24}  "
            f"{_format_pairs(result['mask']):<34}  "
            f"{result['bias']:>22.15e}  "
            f"{result['log2_abs_bias']:>12.6f}"
        )
    print("-" * 104)


def run_known_examples(lat):
    cases = [
        (10, [[12, 0x7], [15, 0x7]], [[10, 0xe]]),
        (11, [[8, 0x9], [11, 0x9]], [[4, 0xb]]),
        (12, [[8, 0x7], [11, 0xf]], [[1, 0xf], [1, 0xf], [1, 0xf]]),
        (13, [[0, 0x9], [3, 0x9]], [[1, 0xd]]),
        (14, [[5, 0xd]], [[10, 0xb]]),
        (15, [[5, 0xd]], [[10, 0xb]]),
        (16, [[5, 0xd]], [[10, 0xb]]),
        (17, [[5, 0xd]], [[10, 0xb]]),
        (18, [[5, 0xd]], [[10, 0xb]]),
    ]
    _print_known_results([_known_result(rounds, lat, diff, mask) for rounds, diff, mask in cases])

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


    RUN_KNOWN_EXAMPLES = True

    # change the rounds here to find the best 
    # DL distinguishers for different rounds
    SEARCH_ROUNDS = 17
    DIFF_VALUES = range(1, SBOX_VALUES)
    MASK_VALUES = range(1, SBOX_VALUES)

    if RUN_KNOWN_EXAMPLES:
        run_known_examples(LAT)

    best = search_best_differential_linear(
        SEARCH_ROUNDS,
        LAT,
        diff_values=DIFF_VALUES,
        mask_values=MASK_VALUES,
    )
    print("best", best)




 
