#include<iostream>
#include<vector>
#include<cassert>
#include<cmath>
#include<vector>
#include<cstddef> // for size_t
#include<algorithm>
#include<tuple>
#include<iomanip>
#include<thread> 
#include<mutex> 
#include<atomic>
#include"asconinfo.h"

using namespace std;

vector< vector<pair<word32, double> > > cartesian_product( 
        const vector<vector<pair<word32, double>>>& input_vectors ) 
{
    if (input_vectors.empty()) 
        return {{}};
    
    vector<vector<pair<word32, double>>> result = {{}};
    
    for (const auto& vec : input_vectors) 
    {
        vector<vector<pair<word32, double>>> new_result;
        
        for (const auto& combination : result) 
        {
            for (const auto& element : vec) 
            {
                vector<pair<word32, double>> new_combination = combination;
                new_combination.push_back(element);
                new_result.push_back( std::move(new_combination));
            }
        }
        
        result = std::move(new_result);
    }
    
    return result;
}

vector< vector< pair<word32, double> > > get2RoundDiff( int pos, word32 diff )
{
    int ROT[5][2] = { { 19, 28 }, { 61, 39 }, { 1, 6 }, { 10, 17 }, { 7, 41 } };

    vector< vector<word32> > state( 5, vector< word32 > ( 64, 0 ) );
    vector< vector<word32> > state1( 5, vector< word32 > ( 64, 0 ) );
    for ( int i = 0; i < 64; i++ )
    {
        if ( i == pos )
        {
            state[0][ i ] = diff >> 4 & 0x1;
            state[1][ i ] = diff >> 3 & 0x1;
            state[2][ i ] = diff >> 2 & 0x1;
            state[3][ i ] = diff >> 1 & 0x1;
            state[4][ i ] = diff >> 0 & 0x1;
        }
    }

    for ( int i = 0; i < 5; i++ )
        for ( int j = 0; j < 64; j++ )
            state1[i][j] = state[i][j] ^ state[i][ ( j + (64 - ROT[i][0] ) ) % 64 ] ^ state[i][ ( j + (64 - ROT[i][1] ) ) % 64 ];

    auto DDT = genDDT( Sbox );

    vector< vector< pair<word32, double> > > V( 64 );

    for ( int s = 0; s < 64; s++ )
    {
        word32 diff = 0;
        diff = ( state1[0][s] << 4 ) + ( state1[1][s] << 3 ) + ( state1[2][s] << 2 ) + ( state1[3][s] << 1 ) + ( state1[4][s] << 0 );

        for ( int oo = 0; oo < 32; oo++ )
        {
            if ( DDT[oo][diff] > 0 )
                V[s].push_back( make_pair( oo, DDT[oo][diff] ) );
        }
    }

    return V;
}

vector< word32 > passLinearLayer( const vector<word32> & diff )
{
    int ROT[5][2] = { { 19, 28 }, { 61, 39 }, { 1, 6 }, { 10, 17 }, { 7, 41 } };

    vector< vector<word32> > state( 5, vector< word32 > ( 64, 0 ) );
    vector< vector<word32> > state1( 5, vector< word32 > ( 64, 0 ) );

    for ( int i = 0; i < 64; i++ )
    {
        state[0][ i ] = diff[i] >> 4 & 0x1;
        state[1][ i ] = diff[i] >> 3 & 0x1;
        state[2][ i ] = diff[i] >> 2 & 0x1;
        state[3][ i ] = diff[i] >> 1 & 0x1;
        state[4][ i ] = diff[i] >> 0 & 0x1;
    }

    for ( int i = 0; i < 5; i++ )
        for ( int j = 0; j < 64; j++ )
            state1[i][j] = state[i][j] ^ state[i][ ( j + (64 - ROT[i][0] ) ) % 64 ] ^ state[i][ ( j + (64 - ROT[i][1] ) ) % 64 ];

    vector< word32 >  odiff( 64 );
    for ( int s = 0; s < 64; s++ )
    {
        word32 diffx = 0;
        diffx = ( state1[0][s] << 4 ) + ( state1[1][s] << 3 ) + ( state1[2][s] << 2 ) + ( state1[3][s] << 1 ) + ( state1[4][s] << 0 );

        odiff[s] = diffx;
    }
    return odiff;
}

vector< vector<double> > updateMatrix( const vector< vector< vector<word32> > > & Mask, 
    const vector< vector< double > > & x )
{
    vector< vector< double > > y (64, vector<double> ( 1024, 1 ) );

    for ( int s = 0; s < 64; s++ )
    {
        for ( int v0 = 0; v0 < 32; v0++ )
        {
            auto U0 = Mask[s][v0]; 

            for ( int v1 = 0; v1 < 32; v1++ )
            {
                auto U1 = Mask[s][v1];

                for ( int j = 0; j < 64; j++ )
                {
                    word32 index1 = bit_2_int( {U0[j], U0[j + 64], U0[j + 128], U0[j + 192], U0[j+256]}, 5 );
                    word32 index2 = bit_2_int( {U1[j], U1[j + 64], U1[j + 128], U1[j + 192], U1[j+256]}, 5 );

                    y[s][ v0 * 32 + v1 ] *= x[j][ index1 * 32 + index2 ];
                }
            }
        }
    }
    return y;
}

vector< vector<double> > updateMatrix_Zero( const vector< vector< vector<word32> > > & Mask, const vector< vector< double > > & x )
{
    vector< vector< double > > y (64, vector<double> ( 32, 1 ) );

    for ( int s = 0; s < 64; s++ )
    {
        for ( int v = 0; v < 32; v++ )
        {
            auto U = Mask[s][v]; 

            for ( int j = 0; j < 64; j++ )
            {
                word32 index1 = bit_2_int( { U[j], U[j + 64], U[j + 128], U[j + 192], U[j+256]}, 5 );
                y[s][ v ] *= x[j][ index1 ];
            }
        }
    }
    return y;
}

vector<double> updateSingleSbox( const vector< vector<double> > & LAT, const vector<double> & x ) 
{
    int N = LAT.size();

    vector< vector<double> > D ( N, vector<double> ( N, 0 ) );   

    for ( int t = 0; t < N; t++ )
        for ( int u1 = 0; u1 < N; u1++ )
            for ( int u0 = 0; u0 < N; u0++)
                D[u1][t] += LAT[ t ] [u1 ^ u0 ] * x[ u0 * N + u1 ];  

    vector< vector<double> > DD ( N, vector<double> ( N, 0 ) );   


    for ( int t = 0; t < N; t++ )
        for ( int v1 = 0; v1 < N; v1++ )
            for ( int u1 = 0; u1 < N; u1++ )
                DD[t][v1] += LAT[v1][u1] * D[u1][t];

    vector<double> y (N*N, 0);

    for ( int v0 = 0; v0 < N; v0++ )
        for ( int v1 = 0; v1 < N; v1++ )
            y[ v0 * 32 + v1 ] = DD[ v0 ^ v1 ] [ v1 ];

    return y;
}

vector<double> updateSingleSbox_Zero( const vector< vector<double> > & LAT, const vector<double> & x ) 
{
    int N = LAT.size();

    vector<double> D ( N, 0 );   

    for ( int t = 0; t < N; t++ )
        for ( int u = 0; u < N; u++ )
            D[t] += LAT[t][u] * LAT[t][u] * x[u];  
    return D;
}

vector< vector<double> > updateFullSbox( const vector< vector<double> > & LAT, 
                                 const vector< vector<double> > & x ) 
{
    vector< vector< double > > y ( 64 );

    for ( int s = 0; s < 64; s++ )
        y[s] = updateSingleSbox( LAT, x[s] ); 

    return y;
}

vector< vector<double> > updateFullSbox_Zero( const vector< vector<double> > & LAT, 
                                 const vector< vector<double> > & x ) 
{
    vector< vector< double > > y ( 64 );

    for ( int s = 0; s < 64; s++ )
        y[s] = updateSingleSbox_Zero( LAT, x[s] ); 

    return y;
}


vector< vector<double> > getBias_Permutation( int ROUND, const vector<word32> & Diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    vector< vector< double > > X ( 64, vector< double > ( 1024, 0 ) );
    for ( int i = 0; i < 64; i++ )
    {
        for ( int j = 0; j < 32; j++ )
        {
            X[i][ j * 32 + Diff[i] ] = 1;
        }
        fwt ( X[i] ); 

        for ( int j = 0; j < 1024; j++ )
            X[i][j] /= 32.0;

        for ( int j = 0; j < 1024; j++ )
            cout << X[i][j] << ",";
        cout << endl;
    }

    getchar();

    for ( int r = 0; r < ROUND; r++ )
    {
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
                        for ( int k = 0; k < 32; k++ )
                            X[i][ j * 32 + k ] *= -1;
                    }

                }
            }
        }

        // Sbox layer
        X = updateFullSbox( LAT, X ); 

        // Linear layer
        if ( r < ROUND - 1 )
            X = updateMatrix( preparedMask, X ); 
    }


    return X;

    // Sbox layer
}


vector< vector<word32> > getExtendedDDT( word32 Sbox[], word32 inDiff )
{
    vector< vector<word32> > D(32); 

    for ( int x = 0; x < 32; x++ )
        D[ Sbox[x] ^ Sbox[ x ^ inDiff ] ].push_back( Sbox[x] );
    return D;
}

vector< vector<double> > getBias_Permutation_Opt2( int ROUND, const int pos, const word32 diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    auto DDT = genDDT( Sbox );
    auto extendedDDT = getExtendedDDT ( Sbox, diff ); 

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );

    for ( int o = 0; o < 32; o++ )
    {
        if ( DDT[ o ][ diff ] == 0 )
            continue;

        cout << "output " << o << endl;

        vector< vector< double > > X ( 64, vector< double > ( 1024, 0 ) );

        for ( int i = 0; i < 64; i++ )
        {
            if ( i == pos )
            {
                for ( int j = 0; j < 32; j++ )
                {
                    if ( find( extendedDDT[o].begin(), extendedDDT[o].end(), j ) != extendedDDT[o].end() )
                        X[i][ j * 32 + o ] = 1;
                }

                fwt ( X[i] ); 

                for ( int j = 0; j < 1024; j++ )
                    X[i][j] /= ( 1.0 * extendedDDT[ o ].size() );
            }
            else
            {
                for ( int j = 0; j < 32; j++ )
                {
                    X[i][ j * 32 ] = 1;
                }

                fwt ( X[i] ); 

                for ( int j = 0; j < 1024; j++ )
                    X[i][j] /= 32.0;
            }
        }

        for ( int r = 0; r < ROUND; r++ )
        {
            //cout << " ROund " << r << endl;
            // constants
            if ( r > 0 )
            {
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
                                for ( int k = 0; k < 32; k++ )
                                    X[i][ j * 32 + k ] *= -1;
                            }

                        }
                    }
                }

                // Sbox layer
                X = updateFullSbox( LAT, X ); 
            }

            // Linear layer
            if ( r < ROUND - 1 )
                X = updateMatrix( preparedMask, X ); 
        }

        cout << X[46][0x18] << endl;

        for ( int s = 0; s < 64; s++ )
            for ( int j = 0; j < 32; j++ )
                COR[s][j] += DDT[o][diff] * X[s][j];
    }


    return COR;
}

vector< vector<double> > getBias_Permutation_Zero_Opt2( int ROUND, const int pos, const word32 diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    auto DDT = genDDT( Sbox );

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );

    for ( int o = 0; o < 32; o++ )
    {
        if ( DDT[ o ][ diff ] == 0 )
            continue;

        cout << o << " " << DDT[o][diff] << endl;

        vector< vector< double > > X ( 64, vector< double > ( 32, 0 ) );

        for ( int i = 0; i < 64; i++ )
        {
            if ( i == pos )
            {
                for ( int j = 0; j < 32; j++ )
                    X[i][ o ] = 1;

                fwt ( X[i] ); 
            }
            else
            {
                X[i][ 0 ] = 1;
                fwt ( X[i] ); 
            }
        }

        for ( int r = 0; r < ROUND; r++ )
        {
            if ( r > 0 )
            {
                // Sbox layer
                X = updateFullSbox_Zero( LAT, X ); 
            }

            // Linear layer
            if ( r < ROUND - 1 )
                X = updateMatrix_Zero( preparedMask, X ); 
        }

        cout << X[54][16] << endl;

        for ( int s = 0; s < 64; s++ )
            for ( int j = 0; j < 32; j++ )
                COR[s][j] += DDT[o][diff] * X[s][j];
    }


    return COR;
}

vector< vector<double> > getBias_Init_Zero( int ROUND, const int pos, const word32 diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    word64 IV = 0x80400c0600000000ULL;


    vector< vector< double > > X ( 64, vector< double > ( 32, 0 ) );

    for ( int i = 0; i < 64; i++ )
    {
        if ( i == pos )
        {
            X[i][ diff ] = 1;
            fwt ( X[i] ); 
        }
        else
        {
            X[i][ 0 ] = 1;
            fwt ( X[i] ); 
        }
    }

    for ( int r = 0; r < ROUND; r++ )
    {
        // Sbox layer
        X = updateFullSbox_Zero( LAT, X ); 

        // Linear layer
        if ( r < ROUND - 1 )
            X = updateMatrix_Zero( preparedMask, X ); 
    }


    return X;
}

vector< vector<double> > getBias_Init_Zero_Opt2_EqualNonce( int ROUND, const int pos, const word32 diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    //word64 IV = 0x80400c0600000000ULL;

    // IV = 1, Diff
    vector<word32> IN1;
    for ( int i = 0; i < 32; i++ )
        if ( ( ( i >> 4 & 0x1 ) == 1 ) && ( ( ( i >> 1 ) & 0x1 ) == ( i & 0x1 ) ) ) 
            IN1.push_back( i );
    
    vector<double>  DDT1( 32, 0 );  
    for ( auto i : IN1 )
        DDT1[ Sbox[i] ^ Sbox[i ^ diff ] ] += 1 / ( 1.0 * IN1.size() );

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );

    for ( word32 o = 0; o < 32; o++ )
    {
        if ( DDT1[o] == 0 )
            continue;

        auto VO = get2RoundDiff( pos, o );

        auto Product = cartesian_product( VO ); 

        for (size_t di = 0; di < Product.size(); ++di) 
        {
            //cout << o << " " << di << endl;

            double pro = DDT1[o];

            vector< vector< double > > X ( 64, vector< double > ( 32, 0 ) );

            //cout << Product[i].size() << endl;

            //getchar();
            vector<word32> diff(64);

            for (size_t dj = 0; dj < Product[di].size(); ++dj) 
            {
                const auto & [word, value] = Product[di][dj];
                pro *= value;

                diff[dj] = word;
            }

            auto odiff = passLinearLayer( diff );

            for ( int s = 0; s < 64; s++ )
            {
                X[s][ odiff[s] ] = 1;
                fwt ( X[s] ); 
            }

            for ( int r = 0; r < ROUND; r++ )
            {
                // Sbox layer
                if ( r >= 2 )
                    X = updateFullSbox_Zero( LAT, X ); 

                // Linear layer
                if ( ( r < ROUND - 1 ) && ( r >= 2 ) )
                    X = updateMatrix_Zero( preparedMask, X ); 
            }

            for ( int s = 0; s < 64; s++ )
                for ( int j = 0; j < 32; j++ )
                    COR[s][j] += pro * X[s][j];
        }
    }

    return COR;
}

vector< vector<double> > getBias_Init_Zero_Opt2_NoneEqualNonce( int ROUND, const int pos, const word32 diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    //word64 IV = 0x80400c0600000000ULL;

    // IV = 1, Diff
    vector<word32> IN1;
    for ( int i = 0; i < 32; i++ )
        if ( ( ( i >> 4 & 0x1 ) == 1 ) ) 
            IN1.push_back( i );
    
    vector<double>  DDT1( 32, 0 );  
    for ( auto i : IN1 )
        DDT1[ Sbox[i] ^ Sbox[i ^ diff ] ] += 1 / ( 1.0 * IN1.size() );

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );

    for ( word32 o = 0; o < 32; o++ )
    {
        if ( DDT1[o] == 0 )
            continue;

        auto VO = get2RoundDiff( pos, o );

        auto Product = cartesian_product( VO ); 

        for (size_t di = 0; di < Product.size(); ++di) 
        {
            //cout << o << " " << di << endl;

            double pro = DDT1[o];

            vector< vector< double > > X ( 64, vector< double > ( 32, 0 ) );

            //cout << Product[i].size() << endl;

            //getchar();

            for (size_t dj = 0; dj < Product[di].size(); ++dj) 
            {
                const auto & [word, value] = Product[di][dj];
                pro *= value;

                X[dj][ word ] = 1;
                fwt ( X[dj] ); 
            }

            for ( int r = 0; r < ROUND; r++ )
            {
                // Sbox layer
                if ( r >= 2 )
                    X = updateFullSbox_Zero( LAT, X ); 

                // Linear layer
                if ( ( r < ROUND - 1 ) && ( r >= 2 ) )
                    X = updateMatrix_Zero( preparedMask, X ); 
            }

            for ( int s = 0; s < 64; s++ )
                for ( int j = 0; j < 32; j++ )
                    COR[s][j] += pro * X[s][j];
        }
    }

    return COR;
}

vector< vector<double> > getBias_Permutation_Zero_Opt3( int ROUND, const int pos, const word32 diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    //word64 IV = 0x80400c0600000000ULL;

    // IV = 1, Diff
    vector<word32> IN1;
    for ( int i = 0; i < 32; i++ )
        if ( ( ( i >> 4 & 0x1 ) == 1 ) && ( ( ( i >> 1 ) & 0x1 ) == ( i & 0x1 ) ) ) 
            IN1.push_back( i );
    //IN1.push_back( 6 );
    
    vector<double>  DDT1( 32, 0 );  
    for ( auto i : IN1 )
        DDT1[ Sbox[i] ^ Sbox[i ^ diff ] ] += 1 / ( 1.0 * IN1.size() );

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );

    for ( word32 o = 0; o < 32; o++ )
    {
        if ( DDT1[o] == 0 )
            continue;

        auto VO = get2RoundDiff( pos, o );

        auto Product = cartesian_product( VO ); 

        //cout << Product.size() << endl;
        
        //continue;

        int count = 0;
        for (size_t di = 0; di < Product.size(); ++di) 
        {
            //cout << o << " " << di << endl;

            double pro = DDT1[o];

            vector< vector< double > > X ( 64, vector< double > ( 32, 0 ) );

            //cout << Product[i].size() << endl;

            //getchar();

            vector<word32> diff(64);

            for (size_t dj = 0; dj < Product[di].size(); ++dj) 
            {
                const auto & [word, value] = Product[di][dj];
                pro *= value;

                diff[dj] = word;
            }

            //cout << count ++ << " " << pro << " " << log2( pro ) << endl;

            auto odiff = passLinearLayer( diff );

            for ( int s = 0; s < 64; s++ )
            {
                X[s][ odiff[s] ] = 1;
                fwt ( X[s] ); 
            }

            for ( int r = 0; r < ROUND; r++ )
            {
                // Sbox layer
                if ( r >= 2 )
                    X = updateFullSbox_Zero( LAT, X ); 

                // Linear layer
                if ( ( r < ROUND - 1 ) && ( r >= 2 ) )
                    X = updateMatrix_Zero( preparedMask, X ); 
            }

            for ( int s = 0; s < 64; s++ )
                for ( int j = 0; j < 32; j++ ) 
                    COR[s][j] += pro * X[s][j];
        }
    }

    return COR;
}


vector< vector<double> > getBias_Permutation_Zero_Opt3_MT( int ROUND, const int pos, const word32 diff )
{
    // genLAT
    auto LAT = genLAT( Sbox, 32 );

    auto Mat = genAsconLinearMatrix();
    auto preparedMask = getPreparedMask( Mat ); 

    //word64 IV = 0x80400c0600000000ULL;

    // IV = 1, Diff
    vector<word32> IN1;
    //for ( int i = 0; i < 32; i++ )
        //if ( ( ( i >> 4 & 0x1 ) == 1 ) && ( ( ( i >> 1 ) & 0x1 ) == ( i & 0x1 ) ) ) 
        //    IN1.push_back( i );
    IN1.push_back( 6 );
    
    vector<double>  DDT1( 32, 0 );  
    for ( auto i : IN1 )
        DDT1[ Sbox[i] ^ Sbox[i ^ diff ] ] += 1 / ( 1.0 * IN1.size() );

    vector < vector< double > > COR ( 64, vector< double > ( 32, 0 ) );

    for ( word32 o = 0; o < 32; o++ )
    {
        if ( DDT1[o] == 0 )
    print( Rot( 2, 1 ) )
            continue;

        auto VO = get2RoundDiff( pos, o );

        auto Product = cartesian_product( VO ); 

        //cout << Product.size() << endl;
        
        //continue;

        int count = 0;
        for (size_t di = 0; di < Product.size(); ++di) 
        {
            //cout << o << " " << di << endl;

            double pro = DDT1[o];

            vector< vector< double > > X ( 64, vector< double > ( 32, 0 ) );

            //cout << Product[i].size() << endl;

            //getchar();

            vector<word32> diff(64);

            for (size_t dj = 0; dj < Product[di].size(); ++dj) 
            {
                const auto & [word, value] = Product[di][dj];
                pro *= value;

                diff[dj] = word;
            }

            cout << count ++ << " " << pro << " " << log2( pro ) << endl;

            auto odiff = passLinearLayer( diff );

            for ( int s = 0; s < 64; s++ )
            {
                X[s][ odiff[s] ] = 1;
                fwt ( X[s] ); 
            }

            for ( int r = 0; r < ROUND; r++ )
            {
                // Sbox layer
                if ( r >= 2 )
                    X = updateFullSbox_Zero( LAT, X ); 

                // Linear layer
                if ( ( r < ROUND - 1 ) && ( r >= 2 ) )
                    X = updateMatrix_Zero( preparedMask, X ); 
            }

            for ( int s = 0; s < 64; s++ )
                for ( int j = 0; j < 32; j++ )
                    COR[s][j] += pro * X[s][j];
        }
    }

    return COR;
}



int main()
{

    int pos = 0;

    word32 diff = 0xd;

    vector<word32> Diff(64, 0);

    Diff[0] = diff;

    auto X = getBias_Permutation_Zero_Opt3_MT( 6, pos, diff );

    double max = 0;
    int I, J;

    for ( int i = 0; i < 64; i++ )
    {
        for ( int j = 8; j < 32; j += 8 )
        {
            cout << dec << i << " " << hex << j << " " << log2( abs( X[i][j] ) ) << endl; 

            if ( abs( X[i][j] ) > max )
            {
                I = i; J = j;
                max = abs( X[i][j] );
            }
        }
    }

    cout << "Max " << endl;

    cout << " Offset " << " " << dec << I << " " << hex << J << " " << max << " " << log2( max ) << endl;
}


