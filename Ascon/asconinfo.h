#ifndef __ASCONINFO_H__
#define __ASCONINFO_H__

#include<vector>
using namespace std;

typedef unsigned char word8;
typedef unsigned int word32;
typedef unsigned long long word64;

extern word32 Sbox[];
extern word32 Const[];

vector<word32> int_2_bit( word32 x, int N = 5  );
word32 bit_2_int( const vector<word32> & bits, int N = 5  );
word32 dot( word32 x, word32 y, int N = 32 );
vector< vector<double> > genLAT( word32 Sbox[], int N ); 
vector< double > kron_vec( const vector< double > & A, const vector< double > & B );
vector< vector< double > > kron_matrix( vector< vector< double > >  & A, vector< vector< double > > & B );
void fwt(std::vector<double>& data );
vector< vector<double> > genDDT ( word32 Sbox[] );
vector< vector< word32 >> genAsconLinearMatrix();
vector< word32 > getMask( int s, word32 v, const vector< vector<word32> > & Mat );
vector< vector< vector< word32 > > > getPreparedMask( const vector< vector< word32> > & Mat ); 

#endif
