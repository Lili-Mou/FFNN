/*
 * activation.h
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

void ReLU(float * z, float * y, int n);
void ReLUPrime(float * y, float * dy_dz, int n);

void LReLU(float * z, float * y, int n);
void LReLUPrime(float * y, float * dy_dz, int n);

void Tanh(float * z, float * y, int n);
void TanhPrime(float * z, float * y, int n);

void sigmoid(float * z, float * y, int n);
void sigmoidPrime(float * z, float * y, int n);


void Softmax(float * z, float * y, int n);




#endif /* ACTIVATION_H_ */
