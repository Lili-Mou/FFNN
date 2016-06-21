/*
 * global.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#include"global.h"
#include <stdio.h>
#include <stdlib.h>

float * weights, * biases;
float * gradWeights, * gradBiases;
int num_weights, num_biases;
int num_train, num_CV, num_test;
int num_label;
pthread_t threads[NUM_THREADS];
pthread_mutex_t mutex_param;
float alpha;
float beta;
float learn_rate;
float C_weights;
float C_embed;
float weight_decay;
float embed_decay;
float momentum;
int batch_size;
int max_epoch;
float J;
int epoch;
int init_epoch;
float p_drop_hid;
float p_drop_embed;
float leak_rate;
int num_embed;

char* X_train_cursor;
char* X_CV_cursor;
char* X_test_cursor;
void RandomInitParam(){
	int NUM_W= 1000, NUM_B = 1000;

	weights = new float[NUM_W];
	biases = new float[NUM_B];

	float bound = 1;
	float bound2 = bound * 2;

	for(int i = 0; i < NUM_W; i++){
		weights[i] = i%10 * bound2 - bound;
	}
	for(int i = 0; i < NUM_B; i++){
		biases[i] = i % 10 * bound2 - bound;
	}
	cout << "INFO: Weights random initialized." << endl;
}

void ReadAllParam(const char * filepath){

	FILE *infile = fopen(filepath, "rb");

	if( infile == NULL){
		cout << "ERR: Can't load parameters; file name: " << filepath << endl;
		exit(1);
	}

	fread((void *) &num_weights, sizeof(int), 1, infile);
	fread((void *) &num_biases, sizeof(int), 1, infile);

	cout << num_weights << " " << num_biases << endl;

	weights = (float *) malloc( sizeof(float) * num_weights);
	biases  = (float *) malloc( sizeof(float) * (num_biases ) );
	gradWeights = (float *) malloc( sizeof(float) * num_weights);
	gradBiases = (float *) malloc( sizeof(float) * (num_biases ));

	
	
	memset( (void *) gradWeights, 0, sizeof(float) * num_weights);
	memset( (void *) gradBiases,  0, sizeof(float) * (num_biases ) );

	fread((void *)weights, sizeof(float), num_weights, infile);
	fread((void *)(biases), sizeof(float), num_biases, infile);

	fclose(infile);

	return;

}
void ReadParam(const char * filepath){
	FILE *infile = fopen(filepath, "rb");
	FILE * embed_file = fopen("embed100", "rb");

	if( infile == NULL){
		cout << "ERR: Can't load parameters; file name: " << filepath << endl;
		exit(1);
	}
	if (embed_file == NULL){
		cout << "ERR: Can't load embeddings; file name: " << embed_file << endl;
	}

	//num_embed = 62817600;
	num_embed = 209392 * 100;
	fread((void *) &num_weights, sizeof(int), 1, infile);
	fread((void *) &num_biases, sizeof(int), 1, infile);

	cout << num_weights << " " << num_biases << endl;

	weights = (float *) malloc( sizeof(float) * num_weights);
	biases  = (float *) malloc( sizeof(float) * (num_biases + num_embed) );
	gradWeights = (float *) malloc( sizeof(float) * num_weights);
	gradBiases = (float *) malloc( sizeof(float) * (num_biases + num_embed));

	
	
	//////////////////////////////////////////
	//fread( weights, sizeof(float), num_weights, infile);
	//fread( biases,  sizeof(float), num_biases,  infile);
	//return;
	
	/////////////////////////////////////////

	memset( (void *) gradWeights, 0, sizeof(float) * num_weights);
	memset( (void *) gradBiases,  0, sizeof(float) * (num_biases + num_embed) );

	fread((void *)weights, sizeof(float), num_weights, infile);
	fread((void *)biases,  sizeof(float), num_embed, embed_file);
	fread((void *)(biases+num_embed), sizeof(float), num_biases, infile);
	num_biases += num_embed;

	fclose(infile);
	fclose(embed_file);

//	cout << weights[0] << " " << weights[1] << " " << weights[num_weights-1] << endl;
//	cout << biases[0]  << " " << biases[1]  << " " << biases[num_biases-1]  << endl;
	return;
}

void WriteEmbed(const char * filepath){

	FILE *infile = fopen(filepath, "rb");
	FILE *outfile = fopen("embed", "wb");

	num_embed = 209392 * 100;

	fread((void *) &num_weights, sizeof(int), 1, infile);
	fread((void *) &num_biases, sizeof(int), 1, infile);

	cout << num_weights << " " << num_biases << endl;

	weights = (float *) malloc( sizeof(float) * num_weights);
	biases  = (float *) malloc( sizeof(float) * (num_biases) );

	fread((void *)weights, sizeof(float), num_weights, infile);
	fread((void *)biases,  sizeof(float), num_embed,   infile);
	cout << "here" << endl;
	cout << biases[0] << " "<< biases[1] << endl;
	fwrite( biases,  sizeof(float), num_embed,  outfile);
	fclose(outfile);

}
void SaveParam(const char * filepath){
	FILE * outfile = fopen(filepath, "wb");
	if (outfile == NULL){
		cout << "ERR: Can't save parameters; file name: " << filepath << endl;
	}
	fwrite( &num_weights, sizeof(int), 1, outfile);
	fwrite( &num_biases,  sizeof(int), 1, outfile);

	fwrite( weights, sizeof(float), num_weights, outfile);
	fwrite( biases,  sizeof(float), num_biases,  outfile);
	fflush( outfile );
	return;
}
