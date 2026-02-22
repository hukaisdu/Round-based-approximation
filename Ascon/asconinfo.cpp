#include<vector>
#include"asconinfo.h"

using namespace std;

word32 Sbox[32] = {0x4, 0xb, 0x1f, 0x14, 0x1a, 0x15, 0x9, 0x2, 0x1b, 0x5, 0x8, 0x12, 0x1d, 0x3, 0x6, 0x1c, 0x1e, 0x13, 0x7, 0xe, 0x0, 0xd, 0x11, 0x18, 0x10, 0xc, 0x1, 0x19, 0x16, 0xa, 0xf, 0x17};

word32 Const[] = { 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2b, 0x1a };

vector< vector< word32 >> genAsconLinearMatrix()
{
    vector< vector<word32> > Mat ( 320, vector<word32> (320, 0) );
    int ROT[5][2] = { { 19, 28 }, { 61, 39 }, { 1, 6 }, { 10, 17 }, { 7, 41 } };
    for (int i = 0; i < 5; i++ )
    {
        for ( int j = 0; j < 64; j++ )
        {
            Mat[64 * i + j] [64 * i + j] = 1;
            Mat[64 * i + j] [64 * i + ( ( j + 64 - ROT[i][0] ) % 64 ) ] = 1;
            Mat[64 * i + j] [64 * i + ( ( j + 64 - ROT[i][1] ) % 64 ) ] = 1;
        }
    }
    return Mat;
}

vector< vector< vector< word32 > > > getPreparedMask( const vector< vector< word32> > & Mat ) 
{
    vector< vector< vector< word32 > > > Mask ( 64, vector< vector<word32> > ( 32 ) );

    for ( int s = 0; s < 64; s++ )
        for ( word32 v = 0; v < 32; v++ )
            Mask[s][v] = getMask( s, v, Mat );
    return Mask;
}

vector< word32 > getMask( int s, word32 v, const vector< vector<word32> > & Mat )
{
    vector< word32 > V ( 320, 0 );
    auto bits = int_2_bit( v, 5 );
    for ( int j = 0; j < 5; j++ )
        V[ s + 64 * j] = bits[j];

    vector< word32 > U ( 320, 0 );

    for ( int o = 0; o < 320; o++ )
        for ( int i = 0; i < 320; i++ )
            U[o] ^= Mat[i][o] * V[i]; // transpose of mat
    return U;
}


vector<word32> int_2_bit( word32 x, int N  )
{
    vector<word32> bits;

    for ( int i = 0; i < N; i++ )
        bits.push_back( x >> (N-1-i) & 0x1 );
    return bits;
}

word32 bit_2_int( const vector<word32> & bits, int N  )
{
    word32 x = 0;
    for ( int i = 0; i < N; i++  )
        x += bits[i] << ( N - 1 - i ); 
    return x;
}

word32 dot( word32 x, word32 y, int N )
{
    auto z = x & y;
    word32 res = 0;
    for ( int i = 0; i < 32; i++ )
        res ^= z >> i & 0x1;
    return res;
}

vector< vector<double> > genLAT( word32 Sbox[], int N ) 
{
    vector< vector<double> > LAT(N, vector<double> (N, 0) ); 

    for ( int maskin = 0; maskin < N; maskin ++ )
        for ( int maskout = 0; maskout < N; maskout ++ )
            for ( int x = 0; x < N; x++ )
            {
                if ( dot( maskin, x ) == dot( maskout, Sbox[x] ) )
                    LAT[ maskout ][ maskin ] += 1;
                else
                    LAT[ maskout ][ maskin ] -= 1;
            }
    for ( int i = 0; i < N; i++ )
        for ( int j = 0; j < N; j++ )
            LAT[i][j] /= (1.0 * N);
    return LAT;
}

vector< double > kron_vec( const vector< double > & A, const vector< double > & B )
{
    int N = A.size();

    vector < double > C ( A.size() * B.size(), 0 );
    for ( int u0 = 0; u0 < N; u0++ )
        for ( int u1 = 0; u1 < N; u1++ )
            C[ u0 * N + u1 ] = A[u0] * B[u1];
    return C;
}

vector< vector< double > > kron_matrix( vector< vector< double > >  & A, vector< vector< double > > & B )
{
    int N = A.size();

    int rowsize = A.size() * B.size();
    int colsize = A[0].size() * B[0].size();

    vector< vector< double > > C ( rowsize, vector< double> ( colsize, 0 ) );

    for ( int v0 = 0; v0 < N; v0++ )
        for ( int v1 = 0; v1 < N; v1++ )
            for ( int u0 = 0; u0 < N; u0++ )
                for ( int u1 = 0; u1 < N; u1++ )
                    C[ v0 * N + v1 ][ u0 * N + u1] = A[v0][u0] * B[v1][u1];
    return C;
}

void fwt(std::vector<double>& data ) 
{
    const size_t n = data.size();
    
    for (size_t len = 1; len < n; len <<= 1) 
    {
        for (size_t i = 0; i < n; i += 2 * len) 
        {
            for (size_t j = i; j < i + len; ++j) 
            {
                const double a = data[j];
                const double b = data[j + len];
                data[j] = a + b;     
                data[j + len] = a - b;
            }
        }
    }
}

vector< vector<double> > genDDT ( word32 Sbox[] )
{
    vector< vector<double> > DDT( 32, vector<double> ( 32, 0 ) );

    for ( word32 ind = 0; ind < 32; ind++ )
        for ( word32 x = 0; x < 32; x++ )
            DDT[ Sbox[x] ^ Sbox[ x ^ ind ] ][ ind ] += 1 / 32.0;
    return DDT;
}

