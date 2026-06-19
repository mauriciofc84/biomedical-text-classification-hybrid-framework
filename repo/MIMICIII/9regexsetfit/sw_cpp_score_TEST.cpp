#include <iostream>
#include <string>
#include <vector>
#include <fstream>
//#include <omp.h>
#include <stdlib.h>

using namespace std;

class SWalgorithm{
  public:
    // constructor
    SWalgorithm(string _secuenciaA_, string _secuenciaB_);
    // metodo split
    vector <string> split(string sequence);
    // metodo alineamiento
    void alignment ();
    void backtracking();
    void strings();
    // atributos
    vector <string> secuenciaA;
    vector <string> secuenciaB;
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

 SWalgorithm::SWalgorithm(string _secuenciaA_, string _secuenciaB_){
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
    string string_ = "";
    string ustring_A = "";
    string ustring_B = "";
    while( index_t < alignedTokensA.size() ) {
        string tokenA = alignedTokensA[index_t];
        string tokenB = alignedTokensB[index_t];
        if( tokenA == tokenB ){
            if( !tokenA.empty() && !tokenB.empty() ){
                string_ += " "+tokenA;
                scoreSW += tokenA.size();
            }
            if ( !ustring_A.empty() ){
                uaTokens_A.push_back( ustring_A.substr(1, ustring_A.length()) );
                ustring_A = "";
            }
            if ( !ustring_B.empty() ){
                uaTokens_B.push_back( ustring_B.substr(1, ustring_B.length()) );
                ustring_B = "";
            }
        }
        else{
            scoreSW-=1;
            if( !string_.empty() ){
                aTokens.push_back( string_.substr(1, string_.length()) );
                string_ = "";
            }
            if( !tokenA.empty() )
                ustring_A += " "+tokenA;
            if( !tokenB.empty() )
                ustring_B += " "+tokenB;
        }
        index_t+=1;
    }
    if ( !string_.empty() )
        aTokens.push_back( string_.substr(1, string_.length()) );
    if( !ustring_A.empty() )
        uaTokens_A.push_back( ustring_A.substr(1, ustring_A.length()) );
    if( !ustring_B.empty() )
        uaTokens_B.push_back( ustring_B.substr(1, ustring_B.length()) );
}

class Combinations{
  public:
    // constructor
    Combinations( string nombre_ );
    void extract_tokens();
    // atributos
    void max_similarity(string A, string B, vector<int> &max_values, vector<int> &max_length, vector<int> &max_classes, int classe, int pos);
    //training
    vector <string> datosX;
    vector <int> clasesX;
    //test
    vector <string> datosU;
    string nombre;
};

 Combinations::Combinations( string nombre_){
    nombre = nombre_;
    ifstream file;

    //training (texts)
    string filename = "./out/DATOSX_"+nombre+".txt";
    file.open(filename.c_str());
    string str;
    while (getline(file, str))
    {
        datosX.push_back(str);
    }
    file.close();
    //training (labels)
    string filename_ = "./out/CLASESX_"+nombre+".txt";
    file.open(filename_.c_str());
    int str_;
    while (file>>str_)
    {
        clasesX.push_back(str_);
    }
    file.close();

    //test
    string filename__ = "./out/DATOSU_"+nombre+".txt";
    file.open(filename__.c_str());
    string str__;
    while (getline(file, str__))
    {
        datosU.push_back(str__);
    }
    file.close();
 }

void Combinations::max_similarity(string A, string B, vector<int> &max_values, vector<int> &max_length, vector<int> &max_classes, int classe, int pos){
    SWalgorithm objeto( A, B );
    objeto.alignment();

    string B_aux;
    for (int i=0;i<B.size();i++){
        if(B[i]!=' '){
            B_aux += B[i];
        }
    }

    int val = objeto.maxTemporal;

    if (val > max_values[pos]){
        max_values[pos] = val;
        max_classes[pos] = classe;
        max_length[pos] = B_aux.size();
    }
 }

 void Combinations::extract_tokens(){
     int i = 0, j = 0, k = 0, index = 0;
     vector<int> posA;
     vector<int> posB;
     ofstream file;
     string filename = "./out/CLASESU_"+nombre+".txt";
     string filename_ = "./out/SCORESU_"+nombre+".txt";

     vector<int> max_values;
     vector<int> max_length;
     vector<int> max_classes;

     for(i=0;i<datosU.size();i++){
        max_values.push_back(-1);
        max_length.push_back(-1);
        max_classes.push_back(-1);
        for(j=0;j<datosX.size();j++){
            posA.push_back(i);
            posB.push_back(j);
        }
     }

     for(i=0;i<posA.size();i++){
        max_similarity(datosU[posA[i]], datosX[posB[i]], max_values, max_length, max_classes, clasesX[posB[i]], posA[i]);
     }

    file.open(filename.c_str());
    for(k=0;k<max_classes.size();k++)
        file << max_classes[k]<<endl;
    file.close();

    file.open(filename_.c_str());
    for(k=0;k<max_values.size();k++)
        file <<1000*max_values[k]/max_length[k]<<endl; //3 decimals
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
