#include<iostream>
#include<vector>
#include<cassert>
#include<cmath>
#include <vector>
#include <omp.h>
#include <cstddef> // for size_t
#include"asconinfo.h"

using namespace std;

#define THREAD 16

vector< vector<double> > updateMatrix_2nd( const vector< vector< vector<word32> > > & Mask, 
                                           const vector< vector< double > > & x )
{
    vector< vector< double > > y (64, vector<double> ( 32 * 32 * 32 *32, 1 ) );

    
    for ( int s = 0; s < 64; s++ )
    {
        #pragma omp parallel for num_threads(THREAD) schedule(static)
        for ( int v0 = 0; v0 < 32; v0++ )
        {
            auto U0 = Mask[s][v0]; 
            for ( int v1 = 0; v1 < 32; v1++ )
            {
                auto U1 = Mask[s][v1]; 
                for ( int v2 = 0; v2 < 32; v2++ )
                {
                    auto U2 = Mask[s][v2]; 
                    for ( int v3 = 0; v3 < 32; v3++ )
                    {
                        auto U3 = Mask[s][v3]; 

                        for ( int j = 0; j < 64; j++ )
                        {
                            word32 index1 = bit_2_int( {U0[j], U0[j + 64], U0[j + 128], U0[j + 192], U0[j+256]}, 5 );
                            word32 index2 = bit_2_int( {U1[j], U1[j + 64], U1[j + 128], U1[j + 192], U1[j+256]}, 5 );
                            word32 index3 = bit_2_int( {U2[j], U2[j + 64], U2[j + 128], U2[j + 192], U2[j+256]}, 5 );
                            word32 index4 = bit_2_int( {U3[j], U3[j + 64], U3[j + 128], U3[j + 192], U3[j+256]}, 5 );

                            y[s][ (v0 << 15) + (v1 << 10) + (v2 << 5) + v3 ] *= 
                                x[j][ index1 * (1<<15) + index2 * (1 << 10) + index3 * (1 << 5) + index4 ];
                        }
                    }
                }
            }
        }
    }
    return y;
}

vector< vector<double> > updateMatrix_Zero_2nd( const vector< vector< vector<word32> > > & Mask, 
                                           const vector< vector< double > > & x )
{
    vector< vector< double > > y (64, vector<double> ( 32 * 32 * 32, 1 ) );

    for ( int s = 0; s < 64; s++ )
    {
        #pragma omp parallel for num_threads(THREAD) schedule(static)
        for ( int v1 = 0; v1 < 32; v1++ )
        {
            auto U1 = Mask[s][v1]; 
            for ( int v2 = 0; v2 < 32; v2++ )
            {
                auto U2 = Mask[s][v2]; 
                for ( int v3 = 0; v3 < 32; v3++ )
                {
                    auto U3 = Mask[s][v3]; 

                    for ( int j = 0; j < 64; j++ )
                    {
                        word32 index1 = bit_2_int( {U1[j], U1[j + 64], U1[j + 128], U1[j + 192], U1[j+256]}, 5 );
                        word32 index2 = bit_2_int( {U2[j], U2[j + 64], U2[j + 128], U2[j + 192], U2[j+256]}, 5 );
                        word32 index3 = bit_2_int( {U3[j], U3[j + 64], U3[j + 128], U3[j + 192], U3[j+256]}, 5 );

                        y[s][ (v1 << 10) + (v2 << 5) + v3 ] *= 
                                x[j][ index1 * (1<<10) + index2 * (1 << 5) + index3 ];
                    }
                }
            }
        }
    }
    return y;
}

vector<double> updateSingleSbox_2nd(const vector<vector<double>>& LAT, const vector<double>& x) 
{
    const size_t N = 32; // 假设 N = 32

    auto idx = [N](size_t i, size_t j, size_t k, size_t l) 
    {
        return i * (N*N*N) + j * (N*N) + k * N + l;
    };

    std::vector<double> D_vec(N*N*N*N, 0.0);
    for (size_t t = 0; t < N; t++) 
    {
        for (size_t u1 = 0; u1 < N; u1++) 
        {
            for (size_t u2 = 0; u2 < N; u2++) 
            {
                for (size_t u3 = 0; u3 < N; u3++) 
                {
                    for (size_t u0 = 0; u0 < N; u0++) 
                    {
                        if ( x[u0*(1<<15) + u1*(1<<10) + u2*(1<<5) + u3] != 0 )
                        {
                            size_t index = idx(u1, u2, u3, t);
                            D_vec[index] += LAT[t][u0 ^ u1 ^ u2 ^ u3] * x[u0*(1<<15) + u1*(1<<10) + u2*(1<<5) + u3];
                        }
                    }
                }
            }
        }
    }

    std::vector<double> DD_vec(N*N*N*N, 0.0);
    for (size_t u2 = 0; u2 < N; u2++) 
    {
        for (size_t u3 = 0; u3 < N; u3++) 
        {
            for (size_t t = 0; t < N; t++) 
            {
                for (size_t v1 = 0; v1 < N; v1++) 
                {
                    for (size_t u1 = 0; u1 < N; u1++) 
                    {
                        size_t src_idx = idx(u1, u2, u3, t);
                        size_t dst_idx = idx(u2, u3, t, v1);
                        if ( D_vec[ src_idx ] != 0 )
                            DD_vec[dst_idx] += LAT[v1][u1] * D_vec[src_idx];
                    }
                }
            }
        }
    }
    std::vector<double>().swap(D_vec);

    std::vector<double> DDD_vec(N*N*N*N, 0.0);
    for (size_t u3 = 0; u3 < N; u3++) 
    {
        for (size_t t = 0; t < N; t++)
        {
            for (size_t v1 = 0; v1 < N; v1++) 
            {
                for (size_t v2 = 0; v2 < N; v2++) 
                {
                    for (size_t u2 = 0; u2 < N; u2++) 
                    {
                        size_t src_idx = idx(u2, u3, t, v1);
                        size_t dst_idx = idx(u3, t, v1, v2);
                        if ( DD_vec[src_idx] != 0 )
                        DDD_vec[dst_idx] += LAT[v2][u2] * DD_vec[src_idx];
                    }
                }
            }
        }
    }
    std::vector<double>().swap(DD_vec);

    std::vector<double> DDDD_vec(N*N*N*N, 0.0);
    for (size_t t = 0; t < N; t++) 
    {
        for (size_t v1 = 0; v1 < N; v1++) 
        {
            for (size_t v2 = 0; v2 < N; v2++) 
            {
                for (size_t v3 = 0; v3 < N; v3++) 
                {
                    for (size_t u3 = 0; u3 < N; u3++) 
                    {
                        size_t src_idx = idx(u3, t, v1, v2);
                        size_t dst_idx = idx(t, v1, v2, v3);
                        if ( DDD_vec[src_idx] != 0 )
                        DDDD_vec[dst_idx] += LAT[v3][u3] * DDD_vec[src_idx];
                    }
                }
            }
        }
    }
    std::vector<double>().swap(DDD_vec);

    std::vector<double> y(N*N*N*N, 0.0);
    for (size_t v0 = 0; v0 < N; v0++) 
    {
        for (size_t v1 = 0; v1 < N; v1++) 
        {
            for (size_t v2 = 0; v2 < N; v2++) 
            {
                for (size_t v3 = 0; v3 < N; v3++) 
                {
                    size_t t = v0 ^ v1 ^ v2 ^ v3;
                    size_t y_idx = v0*(1<<15) + v1*(1<<10) + v2*(1<<5) + v3;
                    size_t dddd_idx = idx(t, v1, v2, v3);
                    if ( DDDD_vec[ dddd_idx ] != 0 )
                    y[y_idx] = DDDD_vec[dddd_idx];
                }
            }
        }
    }

    return y;
}

vector<double> updateSingleSbox_Zero_2nd(const vector<vector<double>>& LAT, const vector<double>& x) 
{
    const size_t N = 32; // 假设 N = 32
                         //
    auto idx4 = [N](size_t i, size_t j, size_t k, size_t l ) 
    {
        return i * (N*N*N) + j * (N*N) + k*N + l;
    };

    auto idx = [N](size_t i, size_t j, size_t k ) 
    {
        return i * (N*N) + j * (N) + k;
    };

    std::vector<double> D_vec(N*N*N*N, 0.0);

    for (size_t t = 0; t < N; t++ ) 
    {
        for (size_t v1 = 0; v1 < N; v1++ )
        {
            for (size_t u2 = 0; u2 < N; u2++ ) 
            {
                for (size_t u3 = 0; u3 < N; u3++ ) 
                {
                    for (size_t u1 = 0; u1 < N; u1++ ) 
                    {
                        size_t index = idx4( u2, u3, t, v1 );
                        D_vec[index] += LAT[v1][u1] * LAT[t][u1 ^ u2 ^ u3] * x[u1*(1<<10) + u2*(1<<5) + u3];
                    }
                }
            }
        }
    }

    std::vector<double> DD_vec(N*N*N*N, 0.0);

    for (size_t u3 = 0; u3 < N; u3++) 
    {
        for (size_t t = 0; t < N; t++) 
        {
            for ( size_t v2 = 0; v2 < N; v2++ )
            {
                for ( size_t v1 = 0; v1 < N; v1++ )
                {
                    for (size_t u2 = 0; u2 < N; u2++) 
                    {
                        size_t src_idx = idx4(u2, u3, t, v1);
                        size_t dst_idx = idx4(u3, t, v1, v2);

                        DD_vec[dst_idx] += LAT[v2][u2] * D_vec[src_idx];
                    }
                }
            }
        }
    }

    std::vector<double>().swap(D_vec);

    std::vector<double> DDD_vec(N*N*N*N, 0.0);
    for (size_t t = 0; t < N; t++ )
    {
        for (size_t v1 = 0; v1 < N; v1++ ) 
        {
            for (size_t v2 = 0; v2 < N; v2++ ) 
            {
                auto v3 = t ^ v1 ^ v2;

                for (size_t u3 = 0; u3 < N; u3++ ) 
                {
                    size_t src_idx = idx4(u3, t, v1, v2);
                    size_t dst_idx = idx4(t, v1, v2, v3);

                    DDD_vec[dst_idx] += LAT[v3][u3] * DD_vec[src_idx];
                }
            }
        }
    }
    std::vector<double>().swap(DD_vec);

    std::vector<double> y(N*N*N, 0.0);

    for (size_t v1 = 0; v1 < N; v1++) 
    {
        for (size_t v2 = 0; v2 < N; v2++) 
        {
            for (size_t v3 = 0; v3 < N; v3++) 
            {
                size_t t = v1 ^ v2 ^ v3;

                size_t y_idx = idx( v1, v2, v3 );

                size_t dddd_idx = idx4(t, v1, v2, v3);

                y[y_idx] = DDD_vec[dddd_idx];
            }
        }
    }

    return y;
}

vector< vector<double> > updateFullSbox_2nd( const vector< vector<double> > & LAT, 
                                 const vector< vector<double> > & x ) 
{
    vector< vector< double > > y ( 64 );

    #pragma omp parallel for num_threads(THREAD) schedule(static)
    for ( int s = 0; s < 64; s++ )
    {
        y[s] = updateSingleSbox_2nd( LAT, x[s] ); 
    }

    return y;
}

vector< vector<double> > updateFullSbox_Zero_2nd( const vector< vector<double> > & LAT, 
                                 const vector< vector<double> > & x ) 
{
    vector< vector< double > > y ( 64 );

    #pragma omp parallel for num_threads(THREAD) schedule(static)
    for ( int s = 0; s < 64; s++ )
        y[s] = updateSingleSbox_Zero_2nd( LAT, x[s] ); 

    return y;
}


vector< vector<double> > getBias_Permutation_2nd( int ROUND, 
    const int pos0, word32 diff0, const int pos1, word32 diff1 )
{
    auto LAT = genLAT( Sbox, 32 ); 
    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    vector< vector<double> > X ( 64, vector< double > ( 32 * 32 * 32 * 32, 0 ) );

    for ( int i = 0; i < 64; i++ )
    {
        if ( i == pos0 )
        {
            for ( int j = 0; j < 32; j++ )
                X[i][j * (32 * 32 * 32 ) + diff0 * 32 * 32 + diff0  ] = 1;
        }
        else if ( i == pos1 )
        {
            for ( int j = 0; j < 32; j++ )
                X[i][j * (32 * 32 * 32 ) + diff1 * 32 + diff1 ] = 1;
        }
        else
        {
            for ( int j = 0; j < 32; j++ )
                X[i][j * (32 * 32 * 32 ) ] = 1;
        }

        fwt( X[i] );

        for ( int k = 0; k < 32 * 32 * 32 * 32; k++ )
            X[i][k] /= 32.0;
    }

    for ( int r = 0; r < ROUND; r++ )
    {
        cout << r << endl;

        // constants
        for ( int i = 0; i < 64; i++ )
        {
            if ( ( Const[r] >> ( 63 - i ) & 0x1 ) == 0 )
                continue;
            else
            {
                for ( int j = 0; j < 32; j++ )
                {
                    if ( j >> 2 & 0x1 )
                    {
                        for ( int k = 0; k < 32 * 32 * 32; k++ )
                            X[i][ j * 32 * 32 * 32 + k ] *= -1;
                    }
                }
            }
        }
        // Sbox layer
        X = updateFullSbox_2nd( LAT, X ); 

        // Linear Layer
        if ( r < ROUND - 1 )
            X = updateMatrix_2nd( preparedMask, X );
    }

    return X;
}

vector< vector<double> > getBias_Init_2nd( int ROUND, 
    const int pos0, word32 diff0, const int pos1, word32 diff1 )
{
    auto LAT = genLAT( Sbox, 32 ); 
    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    word64 iv = 0x80400c0600000000ull;

    vector< vector<double> > X ( 64, vector< double > ( 32 * 32 * 32 * 32, 0 ) );

    for ( int i = 0; i < 64; i++ )
    {
        if ( i == pos0 )
        {
            for ( int j = 0; j < 32; j++ )
            {
                if ( ( iv >> (63 - i) & 1 ) == ( j >> 4 & 0x1 ) && ( ( j >> 1 & 0x1 ) == ( j & 0x1 ) )  )
                    X[i][j * (32 * 32 * 32 ) + diff0 * 32 * 32 + diff0  ] = 1;
            }
        }
        else if ( i == pos1 )
        {
            for ( int j = 0; j < 32; j++ )
                if ( ( iv >> (63 - i) & 1 ) == ( j >> 4 & 0x1 ) && ( ( j >> 1 & 0x1 ) == ( j & 0x1 ) )  )
                    X[i][j * (32 * 32 * 32 ) + diff1 * 32 + diff1 ] = 1;
        }
        else
        {
            for ( int j = 0; j < 32; j++ )
                if ( ( iv >> (63 - i) & 1 ) == ( j >> 4 & 0x1 ) ) 
                    X[i][j * (32 * 32 * 32 ) ] = 1;
        }

        int w = 0;
        for ( int k = 0; k < 32 * 32 * 32 * 32; k++ )
            w += X[i][k];

        fwt( X[i] );

        for ( int k = 0; k < 32 * 32 * 32 * 32; k++ )
            X[i][k] /= (1.0 * w);

            /*
        for ( int k0 = 0; k0 < 32; k0++ )
        {
            for ( int k1 = 0; k1 < 32 * 32 * 32; k1++ )
                cout << X[i][(k0 << 15) + k1] << " ";

            cout << "i " << i << " k0 " << k0 << endl;
            getchar();
        }
            */
    }

    //getchar();

    for ( int r = 0; r < ROUND; r++ )
    {
        cout << r << endl;

        // constants
        for ( int i = 0; i < 64; i++ )
        {
            if ( ( Const[r] >> ( 63 - i ) & 0x1 ) == 0 )
                continue;
            else
            {
                for ( int j = 0; j < 32; j++ )
                {
                    if ( j >> 2 & 0x1 )
                    {
                        for ( int k = 0; k < 32 * 32 * 32; k++ )
                            X[i][ j * 32 * 32 * 32 + k ] *= -1;
                    }
                }
            }
        }
        // Sbox layer
        X = updateFullSbox_2nd( LAT, X ); 

        // Linear Layer
        if ( r < ROUND - 1 )
            X = updateMatrix_2nd( preparedMask, X );
    }

    vector< vector<double> > COR (64, vector<double> (32, 0 ) );

    for ( int s = 0; s < 64; s++ )
        for ( int i = 0; i < 32; i++ )
            COR[s][ i ] = X[s][ ( i << 10 ) + ( i << 5 ) + i ];
    return COR;
}

vector< vector<double> > getBias_Init_Zero_2nd( int ROUND, 
    const int pos0, word32 diff0, const int pos1, word32 diff1 )
{
    auto LAT = genLAT( Sbox, 32 ); 
    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    word64 IV = 0x80400c0600000000ull;
    // IV = 1
    vector<word32> IN1;
    for ( int i = 0; i < 32; i++ )
        if ( ( ( i >> 4 & 0x1 ) == 1 ) && ( ( ( i >> 1 ) & 0x1 ) == ( i & 0x1 ) ) ) 
            IN1.push_back( i );
    
    vector<double>  Diff0_DDT1( 32, 0 );  
    for ( auto i : IN1 )
        Diff0_DDT1[ Sbox[i] ^ Sbox[i ^ diff0 ] ] += 1 / ( 1.0 * IN1.size() );

    vector<double>  Diff1_DDT1( 32, 0 );  
    for ( auto i : IN1 )
        Diff1_DDT1[ Sbox[i] ^ Sbox[i ^ diff1 ] ] += 1 / ( 1.0 * IN1.size() );

    // IV = 0
    vector<word32> IN0;
    for ( int i = 0; i < 32; i++ )
        if ( ( ( i >> 4 & 0x1 ) == 0 ) && ( ( ( i >> 1 ) & 0x1 ) == ( i & 0x1 ) ) ) 
            IN0.push_back( i );
    
    vector<double>  Diff0_DDT0( 32, 0 );  
    for ( auto i : IN0 )
        Diff0_DDT0[ Sbox[i] ^ Sbox[i ^ diff0 ] ] += 1 / ( 1.0 * IN0.size() );

    vector<double>  Diff1_DDT0( 32, 0 );  
    for ( auto i : IN0 )
        Diff1_DDT0[ Sbox[i] ^ Sbox[i ^ diff1 ] ] += 1 / ( 1.0 * IN0.size() );

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );
    for ( word32 o0 = 0; o0 < 32; o0++ )
    {
        bool flag0 = false;

        //cout << "Flag " << flag0 << endl;

        if ( IV >> pos0 & 1 )
        {
            if ( Diff0_DDT1[ o0 ] == 0 )
                flag0 = true; 
        }
        else
        {
            if ( Diff0_DDT0[ o0 ] == 0 )
                flag0 = true;
        }

        //cout << "Flag " << Diff0_DDT0[o0] << " " << Diff0_DDT1[o0] << " " << flag0 << endl;

        //getchar();

        if ( flag0 ) continue;


        for ( word32 o1 = 0; o1 < 32; o1++ )
        {
            //cout << Diff1_DDT0[o1] << " " << Diff1_DDT1[o1] << endl;
            //getchar();

            bool flag1 = false;
            if ( IV >> pos1 & 1 )
            {
                if ( Diff1_DDT1[ o1 ] == 0 )
                    flag1 = true; 
            }
            else
            {
                if ( Diff1_DDT0[ o1 ] == 0 )
                    flag1 = true;
            }
            if ( flag1 ) continue;

            vector< vector<double> > X ( 64, vector< double > ( 32 * 32 * 32, 0 ) );
            for ( int i = 0; i < 64; i++ )
            {
                if ( i == pos0 )
                {
                    X[i][ (o0 << 10) + o0 ] = 1;
                }
                else if ( i == pos1 )
                {
                    X[i][ (o1 << 5) + o1 ] = 1;
                }
                else
                {
                    X[i][0] = 1;
                }

                fwt( X[i] );
            }

            for ( int r = 0; r < ROUND; r++ )
            {
                //cout << "Round-" << r << endl;

                // Sbox layer
                if ( r >= 1 )
                X = updateFullSbox_Zero_2nd( LAT, X ); 

                // Linear Layer
                if ( r < ROUND - 1 )
                    X = updateMatrix_Zero_2nd( preparedMask, X );
            }

            for ( int s = 0; s < 64; s++ )
                for ( int j = 0; j < 32; j++ )
                {
                    int ind0 = IV >> (63 - pos0) & 0x1;
                    int ind1 = IV >> (63 - pos0) & 0x1;
                    int ind = (ind0 << 1) + ind1;

                    if ( ind == 0 )
                        COR[s][j] += Diff0_DDT0[o0] * Diff1_DDT0[o1] * X[s][ (j << 10) + (j << 5) + j ];  
                    else if ( ind == 1 )
                        COR[s][j] += Diff0_DDT0[o0] * Diff1_DDT1[o1] * X[s][ (j << 10) + (j << 5) + j ];  
                    else if ( ind == 2 )
                        COR[s][j] += Diff0_DDT1[o0] * Diff1_DDT0[o1] * X[s][ (j << 10) + (j << 5) + j ];  
                    else if ( ind == 3 )
                        COR[s][j] += Diff0_DDT1[o0] * Diff1_DDT1[o1] * X[s][ (j << 10) + (j << 5) + j ];  

                }
        }
    }

    return COR;
}

vector< vector<double> > getBias_Init_Zero_2nd_Opt3( int ROUND, 
    const int pos0, word32 diff0, const int pos1, word32 diff1 )
{
    auto LAT = genLAT( Sbox, 32 ); 
    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    word64 IV = 0x80400c0600000000ull;
    // IV = 1
    vector<word32> IN1;
    for ( int i = 0; i < 32; i++ )
        if ( ( ( i >> 4 & 0x1 ) == 1 ) && ( ( ( i >> 1 ) & 0x1 ) == ( i & 0x1 ) ) ) 
            IN1.push_back( i );
    
    vector<double>  Diff0_DDT1( 32, 0 );  
    for ( auto i : IN1 )
        Diff0_DDT1[ Sbox[i] ^ Sbox[i ^ diff0 ] ] += 1 / ( 1.0 * IN1.size() );

    vector<double>  Diff1_DDT1( 32, 0 );  
    for ( auto i : IN1 )
        Diff1_DDT1[ Sbox[i] ^ Sbox[i ^ diff1 ] ] += 1 / ( 1.0 * IN1.size() );

    // IV = 0
    vector<word32> IN0;
    for ( int i = 0; i < 32; i++ )
        if ( ( ( i >> 4 & 0x1 ) == 0 ) && ( ( ( i >> 1 ) & 0x1 ) == ( i & 0x1 ) ) ) 
            IN0.push_back( i );
    
    vector<double>  Diff0_DDT0( 32, 0 );  
    for ( auto i : IN0 )
        Diff0_DDT0[ Sbox[i] ^ Sbox[i ^ diff0 ] ] += 1 / ( 1.0 * IN0.size() );

    vector<double>  Diff1_DDT0( 32, 0 );  
    for ( auto i : IN0 )
        Diff1_DDT0[ Sbox[i] ^ Sbox[i ^ diff1 ] ] += 1 / ( 1.0 * IN0.size() );

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );
    for ( word32 o0 = 0; o0 < 32; o0++ )
    {
        bool flag0 = false;

        //cout << "Flag " << flag0 << endl;

        if ( IV >> pos0 & 1 )
        {
            if ( Diff0_DDT1[ o0 ] == 0 )
                flag0 = true; 
        }
        else
        {
            if ( Diff0_DDT0[ o0 ] == 0 )
                flag0 = true;
        }

        //cout << "Flag " << Diff0_DDT0[o0] << " " << Diff0_DDT1[o0] << " " << flag0 << endl;

        //getchar();

        if ( flag0 ) continue;


        for ( word32 o1 = 0; o1 < 32; o1++ )
        {
            //cout << Diff1_DDT0[o1] << " " << Diff1_DDT1[o1] << endl;
            //getchar();

            bool flag1 = false;
            if ( IV >> pos1 & 1 )
            {
                if ( Diff1_DDT1[ o1 ] == 0 )
                    flag1 = true; 
            }
            else
            {
                if ( Diff1_DDT0[ o1 ] == 0 )
                    flag1 = true;
            }
            if ( flag1 ) continue;

            vector< vector<double> > X ( 64, vector< double > ( 32 * 32 * 32, 0 ) );
            for ( int i = 0; i < 64; i++ )
            {
                if ( i == pos0 )
                {
                    X[i][ (o0 << 10) + o0 ] = 1;
                }
                else if ( i == pos1 )
                {
                    X[i][ (o1 << 5) + o1 ] = 1;
                }
                else
                {
                    X[i][0] = 1;
                }

                fwt( X[i] );
            }

            for ( int r = 0; r < ROUND; r++ )
            {
                //cout << "Round-" << r << endl;

                // Sbox layer
                if ( r >= 1 )
                X = updateFullSbox_Zero_2nd( LAT, X ); 

                // Linear Layer
                if ( r < ROUND - 1 )
                    X = updateMatrix_Zero_2nd( preparedMask, X );
            }

            for ( int s = 0; s < 64; s++ )
                for ( int j = 0; j < 32; j++ )
                {
                    int ind0 = IV >> (63 - pos0) & 0x1;
                    int ind1 = IV >> (63 - pos0) & 0x1;
                    int ind = (ind0 << 1) + ind1;

                    if ( ind == 0 )
                        COR[s][j] += Diff0_DDT0[o0] * Diff1_DDT0[o1] * X[s][ (j << 10) + (j << 5) + j ];  
                    else if ( ind == 1 )
                        COR[s][j] += Diff0_DDT0[o0] * Diff1_DDT1[o1] * X[s][ (j << 10) + (j << 5) + j ];  
                    else if ( ind == 2 )
                        COR[s][j] += Diff0_DDT1[o0] * Diff1_DDT0[o1] * X[s][ (j << 10) + (j << 5) + j ];  
                    else if ( ind == 3 )
                        COR[s][j] += Diff0_DDT1[o0] * Diff1_DDT1[o1] * X[s][ (j << 10) + (j << 5) + j ];  

                }
        }
    }

    return COR;
}

int main()
{

    auto X = getBias_Init_Zero_2nd( 5, 0, 0x3, 61, 0x3 );

    double max = 0;
    int I, J;

    for ( int i = 0; i < 64; i++ )
    {
        for ( int j = 8; j < 32; j += 8 )
        {
            cout << i << " " << X[i][j] << " " << log2( abs( X[i][j] ) ); 

            if ( abs( X[i][j] ) > max )  
            {
                I = i; J = j;
                max = abs( X[i][j] );
            }
        }
        cout << endl;
    }

    cout << " offset " << dec << 61 << " " << dec << I << " " << hex << J << " " << max << " " << log2( max ) << endl;
}


