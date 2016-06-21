#include "global.h"
#include <iostream>
#include "FFNN.h"
#include <cmath>
#include "train.h"
#include"read_data.h"
#include<stdio.h>
#include<stdlib.h>
using namespace std;

int sample_size = 5000;

inline int maxid(float array[], int len){
	float tmp_max = -10000;
	int max_id = 0;
	for(int i = 0; i < len; i++){
		if ( array[i] > tmp_max){
			tmp_max = array[i];
			max_id = i;
		}
	}
	return max_id;
}
int train_acc_num = 0;
int train_count=0;
inline void * Grad( void * num){

	int i = *(int *)num;
//	Layer ** Xnet = X_train[i];
	Layer ** Xnet = ReadOneXFromBuf( i ,0 ,X_train_cursor);
	
	
	int len = len_X_train[i];
	
	int t = y_train[i];

	train_count++;
	//cout<<"train number "<<i<<endl;
	FeedForward(Xnet, len);

	//cout<<"train number "<<i<<endl;
	int lastidx = len_X_train[i] - 1;

	float * h = Xnet[ lastidx ] -> y;

	J -= log(h[t]);
	//cout << i <<" "<< h[0]<<" "<<h[1]<<" "<<h[2]<<" "<< J << endl;

	for(int x = 0; x < num_label; x++){
		Xnet[ lastidx ]->dE_dz[x] = Xnet[ lastidx ]->y[x];
	}
	Xnet[ lastidx ] -> dE_dz[ t ] -= 1;

	CleanDerivative(Xnet, len);
	BackPropagation(Xnet, len);

	int predicted = maxid( h,  num_label);
	if ( t == predicted)
		train_acc_num ++;

	DeleteClass(Xnet, len_X_train[i]);
	return (void *) &J;
}
void train(FILE * flog){

	int n_sample = 0;
	void * status;

//	time_t t1 = time(NULL);
	int n_miniGD = 0;

   	float cv_acc = 0;
	float test_acc = 0;
	
	sample_size = num_train;
	int small_sample = sample_size/5;
	for (epoch = init_epoch; epoch < max_epoch; ++ epoch) {

		for (int i = 0;i < num_train; i += NUM_THREADS) {
			//cout<< epoch<<" "<<i<<endl;
			
			Grad( &i );
			++n_sample;
			//////////	////////////////
			// Multi-threading
			// Don't use it, if your networks are not large enough.
			/*
			int sampleid[NUM_THREADS];
			for ( int t = 0; t < NUM_THREADS && i + t < num_train; t++){
				++n_sample;

				sampleid[t] = i + t;
				int rc = pthread_create(threads+t, NULL, Grad, (void *) (sampleid+t) );
				if (rc){
					cout << "Error " << rc << endl;
					return;
				}

			}
			for( int t = 0; t < NUM_THREADS && i + t < num_train; t++){
				pthread_join(threads[t], &status);
				J +=  * ( float *) status;
			}
			*/
			// End of multi-threading
			//////////////////////////////////////////
			if (n_sample % batch_size==0){
				// minibatch GD
				//cout <<epoch<<" "<< i <<" J= "<< J/n_sample << endl;
				learn_rate = alpha/(1+beta* n_miniGD) + 0.001;
				n_miniGD += 1;
				
				if (C_weights != 0){
					float weight_decay = 1 - learn_rate * C_weights;
				//float biases_decay = 1 - learn_rate * C_weights;
					cblas_sscal(num_weights, weight_decay, weights, 1);
                                }
				if(C_embed != 0){
					float bias_decay = 1 - learn_rate * C_embed;
                                	cblas_sscal(num_biases, bias_decay, biases, 1);
				}
			/*	for(int i=1;i<=10;i++)
				cout<<gradWeights[i]<<" ";
				cout<<endl;
				for(int i=1;i<=10;i++)
				cout<<weights[i]<<" ";
				cout<<endl;
				 */
				GradDescent(num_weights, learn_rate , gradWeights, weights);
				GradDescent(num_biases,  learn_rate,  gradBiases,  biases);
				
//				GradDescent(num_biases - num_embed,  learn_rate,  gradBiases + num_embed,  biases + num_embed);


				cblas_sscal(num_weights, momentum, gradWeights, 1);
				cblas_sscal(num_biases, momentum, gradBiases, 1);

//				memset( (void *) gradBiases, 0, sizeof(float) * num_biases);


			}

			///////////////////////////////////////
			// CV and test
			if ( n_sample % sample_size == 0 || n_sample % small_sample == 0){
				if (num_CV > 0){
					cv_acc = predict(X_CV, y_CV, num_CV, len_X_CV,1);
				}
				test_acc = predict(X_test, y_test, num_test, len_X_test,2);

				cout << "epoch = " << epoch << "; train acc = " << train_acc_num / (float) train_count
					 <<"    n_sample: " << n_sample << "; J = " << J / sample_size
					 << " cv acc = " << cv_acc
  					 << " test acc = " << test_acc << endl;
				J = 0;
				train_acc_num = 0;
				train_count=0;
			}


		}
//		time_t t2 = time(NULL);
//		cout << t2 - t1 << endl;


//		cout << "=======================\nEpoch = "<< epoch <<  endl;//"; Training J = " <<  J / num_train<< endl;
//		if (num_CV > 0)
//                    predict(X_CV, y_CV, num_CV, len_X_CV);
//		predict(X_test, y_test, num_test, len_X_test);
//		cout << endl;

	}
}
float predict(Layer *** Xtest, int * ytest, int test_num, int * len_test,int number) {

	isTraining = false;
	int test_count=0;
	int correct = 0;
	int len;
	float J = 0.0, avg = 0.0;
	for (int i = 0;i < test_num; ++ i) 
	{
		Layer ** Xnet ;
		if (number==1) 
		{
			Xnet = ReadOneXFromBuf( i ,number ,X_CV_cursor);
			len = len_X_CV[i];
		}
		if (number==2)
		{
			Xnet=ReadOneXFromBuf( i ,number ,X_test_cursor);
			len = len_X_test[i];
		}
		int t = ytest[i];

		test_count++;
		FeedForward(Xnet, len_test[i]);
		int lastidx = len_test[i] - 1;
		float * h = Xnet[ lastidx ] -> y;
		J -= log(h[t]);
		avg += h[t];

		int predict = 0;
		float max_prob = 0.0;

		for(int j = 0; j < num_label; j++){
			if (h[j] > max_prob) {
				max_prob = h[j];
				predict = j;
			}

		}
		DeleteClass(Xnet, len);
//		if (i % 500 == 0){
//			cout<< "test case ("<< i<< "), h = "<< predict<< " (predicted "<< h[predict];
//			cout<< ") "<< t<< " (actural " << h[t]<<")"<<endl;
//		}
		if (predict == t)
			correct += 1;


	//	int len = len_test[i];
		//cout<<predict<<" "<<t<<endl;
		
	}
//	cout << "       average target output" << avg / test_num <<endl;
//	cout << "Cost " << J <<"; Acc " << (correct + 0.0) / test_num << endl;
	float acc = (correct + 0.0)/test_count;

	


	isTraining = true;
	return acc;
}
