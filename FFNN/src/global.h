/*
 * global.h
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include<iostream>
#include<pthread.h>
#include<string.h>
extern"C"
{
    #include<cblas.h>
}
using namespace std;
////////////////////////////////////////////////////
// hyperparameters

extern float alpha;
extern float beta;
extern float C_weights;
extern float C_embed;
extern float weight_decay;
extern float embed_decay;
extern float momentum;
extern int batch_size;
extern int max_epoch;
extern float learn_rate;
extern float J;
extern int epoch;
extern int init_epoch;
extern float p_drop_hid;
extern float p_drop_embed;
extern float leak_rate;
extern bool isTraining;
////////////////////////////////////////////////////
// model parameters

extern float * weights, * biases;
extern float * gradWeights, *gradBiases;
extern int num_weights, num_biases;

extern int num_train;
extern int num_CV;
extern int num_test;
extern int num_label;
extern int num_embed;

extern char* X_train_cursor;
extern char* X_CV_cursor;
extern char* X_test_cursor;
////////////////////////////////////////////////////
// multi-threads
#define NUM_THREADS 1

extern pthread_t threads[];

// helper functions

void RandomInitParam();

void WriteEmbed(const char * filepath);
void ReadParam(const char * filepath);
void ReadAllParam(const char * filepath);
void SaveParam(const char * filepath);

#endif /* GLOBAL_H_ */
