/*
 * predict.h
 *
 *  Created on: Mar 13, 2015
 *      Author: mou
 */

#ifndef PREDICT_H_
#define PREDICT_H_


float predict(Layer *** X_test, int * y_test, int test_num, int * len_test,int number);

void train(FILE * flog);

#endif /* PREDICT_H_ */
