#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
//#include <omp.h>
#include <stdlib.h>

using namespace std;

class SWalgorithm{
  public:
    // constructor
    SWalgorithm(string _secuenciaA_, string _secuenciaB_, string empty_);
    // metodo split
    vector <string> split(string sequence);
    // metodo alineamiento
    void alignment ();
    void backtracking();
    void strings();
    // atributos
    vector <string> secuenciaA;
    vector <string> secuenciaB;

    string _empty;

    int x;
    int y;
    //vector<vector<int> > matriz;
    //vector<vector<int> > tracking;
    int ** matriz;
    int ** tracking;
    int score_insertion;
    int score_deletion;
    int score_mismatch ;
    //int score_match = 1; // se actualiza abajo
    vector <string> alignedTokensA;
    vector <string> alignedTokensB;
    vector <int> posMax;// = (0, 0)
    vector <string> aTokens;
    vector <string> uaTokens_A;
    vector <string> uaTokens_B;
    int DELETION;
    int INSERTION;
    int MATCH;
    int maxTemporal;
    int posMaxX;
    int posMaxY;
    int scoreSW;
};

 SWalgorithm::SWalgorithm(string _secuenciaA_, string _secuenciaB_, string empty_){
    secuenciaA = split(_secuenciaA_);
    secuenciaB = split(_secuenciaB_);
    x = secuenciaA.size();
    y = secuenciaB.size();
    matriz = new int *[x+1];//(int **) malloc(sizeof(int *)*(x+1));
    tracking = new int *[x+1];//(int **) malloc(sizeof(int *)*(x+1));
    for(int i = 0;i<x+1;i++){
        matriz[i] = new int [y+1];//(int *) malloc(sizeof(int)*(y+1));
        tracking[i] = new int [y+1];//(int *) malloc(sizeof(int)*(y+1));
        for(int j=0;j<y+1;j++){
            matriz[i][j] = 0;
            tracking[i][j] = 0;
        }
    }
    score_insertion = -1;
    score_deletion = -1;
    score_mismatch = -1;
    DELETION = 1;
    INSERTION = 2;
    MATCH = 3;
    maxTemporal = 0;
    posMaxX = 0;
    posMaxY = 0;
    scoreSW = 0;
    _empty = empty_;
 }

vector <string> SWalgorithm::split(string sequence){
    vector <string> tokens;

    string token = "";

    for (int i = 0; i< sequence.size(); i++ ){
        if( sequence[i] == ' ' ){
            if (token.size()>0)
                tokens.push_back(token);
            token = "";
        }
        else{
            token+=sequence[i];
        }
    }
    if (token.size()>0)
        tokens.push_back(token);
    return tokens;
}

void SWalgorithm::alignment(){
    for(int mi = 1; mi<x+1; mi++){
        for(int mj = 1; mj<y+1; mj++){
            int scores_matriz[4] = {0,0,0,0};
            int scores_tracking[4] = {0,0,0,0};
            scores_matriz[1] = matriz[mi-1][mj]+score_deletion;
            scores_tracking[1] = DELETION;
            scores_matriz[2] = matriz[mi][mj-1]+score_insertion;
            scores_tracking[2] = INSERTION;
            if( secuenciaA[mi-1] == secuenciaB[mj-1] ){
                int score_match = secuenciaA[mi-1].size();
                scores_matriz[3] = matriz[mi-1][mj-1]+score_match;
                scores_tracking[3] = MATCH;
            }
            else{
                scores_matriz[3] = matriz[mi-1][mj-1]+score_mismatch;
                scores_tracking[3] = MATCH;
            }
            int max_ = 0;
            int max_tracking = 0;
            int pos_i = 0;
            for(int i=0;i<4;i++){
                if( scores_matriz[i]>max_ || (scores_matriz[i]==max_ && scores_tracking[i]>max_tracking) ){
                    max_ = scores_matriz[i];
                    max_tracking = scores_tracking[i];
                    pos_i = i;
                }
            }
            matriz[mi][mj] = scores_matriz[pos_i];
            tracking[mi][mj] = scores_tracking[pos_i];
            if( matriz[mi][mj]>maxTemporal ){
                maxTemporal = matriz[mi][mj];
                posMaxX = mi;
                posMaxY = mj;
            }
        }
    }
}

void SWalgorithm::backtracking(){
    int ti = posMaxX;
    int tj = posMaxY;
    while ( tracking[ti][tj] !=0 ){
        if( tracking[ti][tj] == DELETION ){
            ti-=1;
            alignedTokensA.insert(alignedTokensA.begin(), secuenciaA[ti]);
            alignedTokensB.insert(alignedTokensB.begin(), "");
        }
        else if( tracking[ti][tj] == INSERTION ){
            tj-=1;
            alignedTokensA.insert(alignedTokensA.begin(), "");
            alignedTokensB.insert(alignedTokensB.begin(), secuenciaB[tj]);
        }
        else if( tracking[ti][tj] == MATCH ){
            ti-=1;
            tj-=1;
            alignedTokensA.insert(alignedTokensA.begin(), secuenciaA[ti]);
            alignedTokensB.insert(alignedTokensB.begin(), secuenciaB[tj]);
        }
    }
    int i = 0;
    for(i=0;i<x;++i){
        delete[] matriz[i];//free(matriz[i]);
        delete[] tracking[i];//free(tracking[i]);
    }
    delete[] matriz;//free(matriz);
    delete[] tracking;//free(tracking);
}

void SWalgorithm::strings(){
    int index_t = 0;
    //string empty_ = to_string(posA)+to_string(posB);
    string string_ = "";
    //string ustring_A = "";
    //string ustring_B = "";
    while( index_t < alignedTokensA.size() ) {
        string tokenA = alignedTokensA[index_t];
        string tokenB = alignedTokensB[index_t];
        if( tokenA == tokenB ){
            if( !tokenA.empty() && !tokenB.empty() ){
                //aTokens.push_back( _empty+tokenA );
                string_ += " "+tokenA;
                scoreSW += tokenA.size();
                //printf("%s\n", tokenA);
            }
            /*
            if ( !ustring_A.empty() ){
                uaTokens_A.push_back( ustring_A.substr(1, ustring_A.length()) );
                ustring_A = "";
            }
            if ( !ustring_B.empty() ){
                uaTokens_B.push_back( ustring_B.substr(1, ustring_B.length()) );
                ustring_B = "";
            }
            */
        }

        else{
            scoreSW-=1;
            if( !string_.empty() ){
                //mew
                string_ = _empty+string_;
                //aTokens.push_back( string_.substr(1, string_.length()) );
                aTokens.push_back( string_.substr(0, string_.length()) );
                string_ = "";
            }
            /*
            if( !tokenA.empty() )
                ustring_A += " "+tokenA;
            if( !tokenB.empty() )
                ustring_B += " "+tokenB;
            */
        }

        index_t+=1;
    }

    if ( !string_.empty() ){
        //aTokens.push_back( string_.substr(1, string_.length()) );
        //new
        string_ = _empty+string_;
        aTokens.push_back( string_.substr(0, string_.length()) );
        }
    /*
    if( !ustring_A.empty() )
        uaTokens_A.push_back( ustring_A.substr(1, ustring_A.length()) );
    if( !ustring_B.empty() )
        uaTokens_B.push_back( ustring_B.substr(1, ustring_B.length()) );
    */

}

class Combinations{
  public:
    // constructor
    Combinations( string nombre_ );
    void extract_tokens();
    void extend_tokens(string A, string B, vector<string> &tokens, int posA, int posB);
    // atributos
    vector <string> datosX;
    vector <int> clasesX;
    string nombre;
    string pos;
    string sep = "XYZ";
};

 Combinations::Combinations( string nombre_){
    nombre = nombre_;
    ifstream file;
    string filename = "./out/DATOSX_"+nombre+".txt";


    file.open(filename.c_str());
    string str;
    while (getline(file, str))
    {
        datosX.push_back(str);
    }
    file.close();

    string filename_ = "./out/CLASESX_"+nombre+".txt";
    file.open(filename_.c_str());
    int str_;
    while (file>>str_)
    {
        clasesX.push_back(str_);
    }
    file.close();
 }

void Combinations::extend_tokens(string A, string B, vector<string> &tokens, int posA, int posB){
    pos = to_string(posA)+"-"+to_string(posB)+sep;
    SWalgorithm objeto(A,B, pos);
    objeto.alignment();
    objeto.backtracking();
    objeto.strings();
    tokens.insert(tokens.end(),objeto.aTokens.begin(),objeto.aTokens.end());
    //return objeto.aTokens;
 }

 void Combinations::extract_tokens(){
     int i = 0, j = 0, k = 0, index = 0;
     vector<int> posA;
     vector<int> posB;
     vector<string> tokens;
     ofstream file;
     string filename = "./out/TOKENS_"+nombre+".txt";

     for(i=0;i<clasesX.size();i++){
        for(j=i+1;j<clasesX.size();j++){
            if(clasesX[i]==clasesX[j]){
                if(datosX[i].length()>datosX[j].length()){
                    posA.push_back(i);
                    posB.push_back(j);
                }
                else{
                    posA.push_back(j);
                    posB.push_back(i);
                }
            }
        }
     }

     file.open(filename.c_str());

    //#pragma omp target
    //#pragma omp teams distribute parallel for
     for(index=0;index<posA.size();index++){
        /*
         string A = datosX[posA[index]];
         string B = datosX[posB[index]];
         SWalgorithm objeto(A,B);
         objeto.alignment();
         objeto.backtracking();
         objeto.strings();
         #pragma omp critical
         tokens.insert(tokens.end(),objeto.aTokens.begin(),objeto.aTokens.end());
         //#pragma omp critical
         //for(k=0;k<objeto.aTokens.size();k++)
         //   tokens.push_back()
         //   file << objeto.aTokens[k]+"\n";
         */
         //#pragma omp critical
         extend_tokens(datosX[posA[index]], datosX[posB[index]], tokens, posA[index], posB[index]);
     }
    for(k=0;k<tokens.size();k++)
        //file << to_string(clasesX[posA[k]]) << tokens[k]<<endl;
        file << tokens[k]<<endl;
    file.close();
 }

int main(int argc, char **argv)
{

    //string nombre = "FUMADOR";
    //Combinations combination( nombre );
    //combination.extract_tokens();


    Combinations combination( argv[1] );
    combination.extract_tokens();

    return 0;
}
