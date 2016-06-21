/*
 * read_data.cpp
 *
 *  Created on: Mar 10, 2015
 *      Author: mou
 */


#include"read_data.h"
#include <stdio.h>
#include <stdlib.h>
#include"FFNN.h"
#include"activation.h"

#define sizeofint 4 // sizeof(int);
#define sizeoffloat 4
#define sizeofchar 1
#define MAX_BUF 1500000000
char buf[MAX_BUF];
int count[3];
inline void PermuteLayer(int num, Layer *** source, Layer *** target, int *mapping){
	for ( int i = 0; i<num; i++){
		target[i] = source[ mapping[i] ];
	}
}

inline void PermuteInt(int num, int* source, int* target, int *mapping){
	for ( int i = 0; i<num; i++){
		target[i] = source[ mapping[i] ];
	}
}

void DeleteClass(Layer ** layers, int len){

	if(layers == NULL)
		return;
	for (int i = 0; i< len; ++i){
		Layer * lay = layers[i];
		for (int j = 0; j< lay->numDown; ++j){
			delete lay->connectDown[j];
		}
		if (lay->dE_dy != NULL) delete lay->dE_dy;
		if (lay->dE_dz != NULL && lay->dE_dz != lay->dE_dy) delete lay->dE_dz;
		if (lay->dy_dz != NULL ) delete lay->dy_dz;
		if (lay->y != NULL)     delete lay->y;
		if (lay->z!=NULL && lay->y!=lay->z)	delete lay->z;
		//if (lay->connectUp != NULL) delete lay->connectUp;
		if (lay->connectDown != NULL) delete lay->connectDown;
		delete lay;

	}
	delete layers;


}


Layer** ReadOneXFromBuf(int i ,int number,char *& cursor){
	Layer ** one_net = NULL;
	if(i == 0)
		cursor = buf+count[number];
	int num = *(int*) cursor;
	if (num != i)
		cout << "ERROR: Training data not aligned, # = " << i << endl;

	cursor += sizeofint;
	ConstructNetFromBuf( one_net, i,number, cursor);
	return one_net;
}

void ReadToBuf(const char * filename ){
	FILE *infile = fopen(filename, "rb");

	if (infile == NULL){
		cout << "ERR: cannot open: " << filename << endl;
		exit(1);
	}

	int count=fread( buf, 1,  MAX_BUF, infile);
	cout<<"readtobuf"<<count<<endl;
	fclose(infile);

}
void ReadOneFile(const char * filename, int num_sample, Layer *** &net_array, int *& len){
	FILE *infile = fopen(filename, "rb");

	if (infile == NULL){
		cout << "ERR: cannot open: " << filename << endl;
		exit(1);
	}

	fread( buf, 1,  MAX_BUF, infile);
	char * cursor = buf;
	
	//cout << num_sample << endl;
	// for all samples
	for( int i_sample = 0; i_sample < num_sample; ++i_sample){
		//cout << i_sample << endl;
		if ( *(int *) cursor != i_sample)
				cout << "ERR: samples are not aligned, id: " << i_sample << endl;
		else
			cursor += sizeofint;

	//	ConstructNetFromBuf(net_array[i_sample], len[i_sample], cursor);
//		cout << len[i_sample] << endl;
	}


	fclose(infile);
	return;
}


void ReadLabels(const char * labelfile){

	FILE *infile = fopen(labelfile, "r");
	if( infile == NULL)
		cout << "ERROR: cannot load labels, file: " << labelfile << endl;
	char tmpstring[20];
	fgets( tmpstring, 100, infile);
	num_train = atoi(tmpstring);
	fgets( tmpstring, 100, infile);
	num_CV = atoi(tmpstring);
	fgets( tmpstring, 100, infile);
	num_test = atoi(tmpstring);

//	int * tmp_y_train = new int [num_train];
//	int * tmp_y_CV    = new int [num_CV];
//	int * tmp_y_test  = new int [num_test];
	cout << "number of train/CV/test " << num_train << " " << num_CV << " " << num_test << endl;
	y_train = new int[num_train];
	y_CV    = new int[num_CV];
	y_test  = new int[num_test];

	// y train
	for ( int i = 0; i < num_train; ++i){
		fgets( tmpstring, 100, infile);
		y_train[i] = atoi(tmpstring);
	}
	// y cv
	for ( int i = 0; i < num_CV; ++i){
		fgets( tmpstring, 100, infile);
		y_CV[i] = atoi(tmpstring);
	}
	// y test
	for ( int i = 0; i < num_test; ++i){
		fgets( tmpstring, 100, infile);
		y_test[i] = atoi(tmpstring);
	}

//	PermuteInt(num_train, tmp_y_train, y_train, file_train);
//	PermuteInt(num_CV,    tmp_y_CV,    y_CV,    file_CV);
//	PermuteInt(num_test,  tmp_y_test,  y_test,  file_test);
//
//	delete tmp_y_train;
//	delete tmp_y_CV;
//	delete tmp_y_test;

//	cout << file_train[0] << " " << file_CV[0] << " "<< file_test[0]<<endl;
//	cout << y_train[0] << " " << y_CV[0] << " " << y_test[0] << endl;
//	cout << num_train << " " << num_CV << " " << num_test << endl;
	fclose(infile);
}


void ReadAllData( ){

	char * f_train = "../construct_nn/QC/preprocessed/train_nets_QC_LSTM";
	char * f_CV =    "../construct_nn/QC/preprocessed/CV_nets_QC_LSTM";
	char * f_test =  "../construct_nn/QC/preprocessed/test_nets_QC_LSTM";

	FILE *infile;

	X_train = new Layer **[num_train];
	X_CV    = new Layer **[num_CV];
	X_test  = new Layer **[num_test];


	len_X_train = new int [num_train];
	len_X_CV    = new int [num_CV];
	len_X_test  = new int [num_test];
	
	count[0]=0;
	
	infile=fopen(f_train,"r");
	if (not infile){
		cout << "ERR: " << f_train << " does not exist" << endl;
	}
	X_train_cursor=buf;
	count[1]=fread( buf, 1,  MAX_BUF, infile);
	cout<<"read to buf"<<count[1]<<endl;
	fclose(infile);
	
	infile=fopen(f_CV,"r");
	if (not infile){
		cout << "ERR: " << f_CV << " does not exist" << endl;
	}
	X_CV_cursor=buf+count[1];
	count[2]=count[1]+fread( buf+count[1], 1,  MAX_BUF, infile);
	cout<<"read to buf"<<count[2]<<endl;
	fclose(infile);
	
	infile=fopen(f_test,"r");
	if (not infile){
		cout << "ERR: " << f_test << " does not exist" << endl;
	}
	X_test_cursor=buf+count[2];
	int counttotal = count[2]+fread( buf+count[2], 1,  MAX_BUF, infile);
	cout<<"read to buf"<<counttotal<<endl;
	fclose(infile);
	// read nets
/*
//	ReadOneFile( f_train, num_train, X_train, len_X_train);
//	ReadOneFile( f_test,  num_test,  X_test,  len_X_test);
	if (num_CV > 0)
 //           ReadOneFile( f_CV,    num_CV,    X_CV,    len_X_CV);
		ReadToBuf(f_CV,X_CV_cursor);

	ReadToBuf(f_train,X_train_cursor);
	ReadToBuf(f_test,X_test_cursor);
	
	// permutation
*/
	return;
}


void ConstructNetFromBuf( Layer ** & one_net, int  i,int number, char * & cursor){
	int numlay, numcon;


	numlay = *((int *) cursor);     cursor += sizeofint;
	numcon = *((int *) cursor);     cursor += sizeofint;
//	fread( &numlay, sizeofint, 1, infile);
//	fread( &numcon, sizeofint, 1, infile);
	if (number==0)
		len_X_train[i] = numlay;
	if (number==1)
		len_X_CV[i]	   = numlay;
	if (number==2)
		len_X_test[i]  = numlay;
	//cout << "# layer:" << numlay << " # numcon:" << numcon << endl;
	///////////////////////////////////////////////////////////
	// read layers
	///////////////////////////////////////////////////////////
	
	one_net = new Layer*[numlay];

	memset( one_net, 0, sizeof(Layer *) * numlay);

	for(int i = 0; i < numlay; i++){
		char type;
		type = *((char *) cursor); cursor += sizeofchar;
		
		//cout << "type:" << type << " ";
		if (type == 'p'){
			int numUnit = *((int *) cursor);  cursor += sizeofint;
			int numDown = *((int *) cursor);  cursor += sizeofint;
			char pool_type = * cursor;    cursor += sizeofchar;
			char remark = *cursor; cursor += sizeofchar;
			
			//cout << "numUnit, down " << numUnit << " " << numDown 
			//	<< " pool_type " << pool_type << " " << " remark " << remark << endl;
			// only max pooling is support temporarily
			one_net[i] = new PositivePoolLayer( "pool", numUnit, numDown);
			one_net[i]->p_drop = 0;
		}else if (type == 'a') { // type == a
			int numUnit = *((int *) cursor);  cursor += sizeofint;
			int numDown = *((int *) cursor);  cursor += sizeofint;
			char activation = * cursor; cursor += sizeofchar;
			int bidx = *((int *) cursor);  cursor += sizeofint;
			char remark= *cursor; cursor += sizeofchar;
			//cout << "numUnit:" << numUnit << " numDown:"<< numDown << " bidx:"<< bidx
			//	<< " activation " << activation << " " << " remark " << remark << endl;
				
			if( activation == 'r' ){ // relu
				one_net[i] = new DropoutLayer( "relu", numUnit, bidx, numDown, ReLU, ReLUPrime);
				one_net[i]->p_drop = p_drop_hid;
				// ReLU by default
			} else if( activation == 's') { // softmax
				one_net[i] = new Layer( "softmax", numUnit, bidx, numDown, Softmax, NULL);

			} else if( activation == '0'){
				one_net[i] = new DropoutLayer( "embed", numUnit, bidx, numDown, NULL, NULL);
				delete one_net[i]->z;
				delete one_net[i]->dE_dz;
				one_net[i]->z = one_net[i]->y;
				one_net[i]->dE_dz = one_net[i]->dE_dy;
                one_net[i]->p_drop = p_drop_embed;
			}else if ( activation == 'l'){ //
				one_net[i] = new DropoutLayer( "sigmoid", numUnit, bidx, numDown, sigmoid, sigmoidPrime);
				one_net[i]->p_drop = p_drop_hid;
			}else if ( activation == 't'){ //
				one_net[i] = new DropoutLayer( "t", numUnit, bidx, numDown, Tanh, TanhPrime);
				one_net[i]->p_drop = p_drop_hid;
			}else {
				cout << "ERR: activation not recognized: " << activation << endl;
				exit(1);
			}
			
			//if (remark == 'c'){
			//	one_net[i]->p_drop = 0;
			//}
		} else{
			cout << "ERR: layer type not recognized: " << type << endl;
		}
		
		


	}
	//cout << "bingo here" << endl;
	// read connections
	for(int i = 0; i < numcon; i++){
		char type;
		type = *((char *) cursor); cursor += sizeofchar;
		//cout << "con type:" << type << " " << flush;
		if (type == 'l'){ // linear transform
			int xid = *((int *) cursor);   cursor += sizeofint;
			int yid = *((int *) cursor);   cursor += sizeofint;
			int ydownid = *((int *) cursor); cursor+=sizeofint;
			int Widx = *((int *) cursor);  cursor += sizeofint;
			float Wcoef = 1.0;
			Wcoef = *((float *) cursor); cursor+=sizeoffloat;
			//cout << "xid: " << xid << " yid: " << yid << " ydownid: "<< ydownid << " Widx: " << Widx << " Wcoef: " << Wcoef << flush;
			Connection * con = new Connection(one_net[xid], one_net[yid],
						one_net[xid]->numUnit, one_net[yid]->numUnit,
						Widx, Wcoef);
			one_net[yid]->connectDown[ydownid] = con;
		
		} else if (type == 'b'){ // bilinear 
			int xid = *((int *) cursor);   cursor += sizeofint;
			int xid2 = *((int *) cursor);   cursor += sizeofint;
			int yid = *((int *) cursor);   cursor += sizeofint;
			int ydownid = *((int *) cursor); cursor+=sizeofint;
			int Widx = *((int *) cursor);  cursor += sizeofint;
			float Wcoef = *((float *) cursor); cursor+= sizeoffloat;
			//cout << "xid: " << xid << "xid2: " << xid2 << " yid: " << yid << " ydownid: "<< ydownid << " Widx: " << Widx << flush;
			one_net[xid]->p_drop = 0;
			one_net[xid2]->p_drop = 0;

			Connection * con = new BilinearConnection(one_net[xid], one_net[xid2], one_net[yid],
					one_net[xid]->numUnit, Widx, Wcoef);
				// Layer* _x1, Layer* _x2, Layer* _y, int num
			one_net[yid]->connectDown[ydownid] = con;
			
		}else if (type == '-'){ // minus 
			int xid = *((int *) cursor);   cursor += sizeofint;
			int xid2 = *((int *) cursor);   cursor += sizeofint;
			int yid = *((int *) cursor);   cursor += sizeofint;
			int ydownid = *((int *) cursor); cursor+=sizeofint;
			int Widx = *((int *) cursor);  cursor += sizeofint;
			float Wcoef = *((float *) cursor); cursor+= sizeoffloat;
			//cout << "xid: " << xid << "xid2: " << xid2 << " yid: " << yid << " ydownid: "<< ydownid << " Widx: " << Widx << flush;
			one_net[xid]->p_drop = 0;
			one_net[xid2]->p_drop = 0;

			Connection * con = new MinusConnection(one_net[xid], one_net[xid2], one_net[yid],
					one_net[xid]->numUnit, Widx, Wcoef);
				// Layer* _x1, Layer* _x2, Layer* _y, int num
			one_net[yid]->connectDown[ydownid] = con;
			
		}
		else if (type == 'm'){ // merge
			int xid = *((int *) cursor);   cursor += sizeofint;
			int xid2 = *((int *) cursor);   cursor += sizeofint;
			int yid = *((int *) cursor);   cursor += sizeofint;
			int ydownid = *((int *) cursor); cursor+=sizeofint;
			int Widx = *((int *) cursor);  cursor += sizeofint;
			float Wcoef = *((float *) cursor); cursor+= sizeoffloat;
			//cout << "xid: " << xid << "xid2: " << xid2 << " yid: " << yid << " ydownid: "<< ydownid << " Widx: " << Widx << flush;
			one_net[xid]->p_drop = 0;
			one_net[xid2]->p_drop = 0;

			Connection * con = new MergeConnection(one_net[xid], one_net[xid2], one_net[yid],
					one_net[xid]->numUnit, Widx, Wcoef);
				// Layer* _x1, Layer* _x2, Layer* _y, int num
			one_net[yid]->connectDown[ydownid] = con;
			
		}
		 else if ( type == 'p') { // 'pooling'
			int xid = *((int *) cursor);   cursor += sizeofint;
			int yid = *((int *) cursor);   cursor += sizeofint;
			int ydownid = *((int *) cursor); cursor+=sizeofint;
			//cout << "xid: " << xid << " yid: " << yid << " ydownid: " << ydownid; 
			Connection * con = new PositivePoolConnection( one_net[xid], one_net[yid],
					                  one_net[yid]->numUnit , 1);
	                one_net[yid]->connectDown[ydownid] = con;
		} else {
			cout << "ERR: connection type not recognized: " << int(type) << endl;
			exit(1);
		}
		//cout << endl;
		

	}
	return ;
}

