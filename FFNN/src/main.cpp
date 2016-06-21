/*
 * main.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#include "global.h"
#include <string>
#include "FFNN.h"
#include <cmath>
#include"read_data.h"
#include"train.h"
#include<stdlib.h>
#include <time.h> 
using namespace std;


int main(int argc, char * argv[]){
	cout << "look" << endl;

	srand( 200 ); 
	isTraining = true;

	if (argc != 9){
		alpha = 3;
		beta = 1;
		C_weights = 0;
		C_embed = 0;
		batch_size = 50;
		momentum = 0;
		p_drop_hid = 0;
	        p_drop_embed = 0;
		

	}else{
	
		alpha = atof(argv[1]);
		beta = atof(argv[2]);
		C_weights = atof(argv[3]);
		C_embed = atof(argv[4]);
		batch_size = atoi(argv[5]);
		p_drop_hid = atof(argv[6]);
                p_drop_embed = atof(argv[7]);
                momentum = atof(argv[8]);
		//leak_rate = atof(argv[9]);
		
	}

	max_epoch = 300;
	num_label = 6;
	cout << "alpha = " << alpha << "; beta = " << beta <<"; C_weights = " << C_weights
		<< "; C_embed = " << C_embed << "; batch_size = "  << batch_size
		<< "; momentum = " << momentum << "; p(drop_hid) = " << p_drop_hid 
                << "; p(drop_embed) = " << p_drop_embed << "; leak_rate = "<<leak_rate<<endl;

	
	beta = (1.0/beta - 1.0) / (4852.0/batch_size);

	init_epoch = 0;



	ReadParam("../construct_nn/QC/preprocessed/para_QC_LSTM");

	ReadLabels("../construct_nn/QC/preprocessed/labels");
	

	ReadAllData();
	cout << "here" << endl;
	cout << num_weights << " " << num_biases << endl;
	cout << weights[0] << " " << biases[0] << endl;

	
	cout << "INFO: Data loaded" << endl;

	
	train(NULL);

	cout << "done"<<endl;
	return 0;

}
