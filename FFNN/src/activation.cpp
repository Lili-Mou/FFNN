/*
 * activation.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */


#include"global.h"
#include"activation.h"
#include"math.h"

void sigmoid(float *z, float*y, int n){
	n--;
	while( n>= 0){
		y[n] = 1.0/(1+exp(-z[n]));
		n--;
	}
	return;
}

void sigmoidPrime(float *y, float * dy_dz, int n){
	n--;
	while( n>=0 ){
		dy_dz[n] = y[n] * (1 - y[n]);
		n--;
	}
	return;
}

void Tanh(float * z, float * y, int n){
	// in our implementation, we define
	// tanh(x) = sigmoid(x)
	// there is a scaling of 2 to the input
	n--;
	while( n>= 0 ){
        	y[n] = 2.0/(1.0+exp(-z[n]))-1;
		float t = exp(z[n]*2.0);//, t2 = exp(- z[n]);
		y[n] = 1.0-2.0/(1.0 + t);
		n--;
	}
        return;
}

void TanhPrime(float *y, float * dy_dz, int n){
	n--;
	while( n >= 0){
	        dy_dz[n] = 1 - y[n] * y[n];
		n--;
	}
	return;
}

// y = ReLU(z)
void ReLU(float * z, float * y, int n){
	n--;
	while( n >= 0 ){
		if ( z[n] > 0 )
			y[n] = z[n];
		else
			y[n] = 0.0;
		n--;
	}
	return ;
}

// dy_dz =  dy / dz | evaluated at y
void ReLUPrime(float * y, float * dy_dz, int n){
	n--;
	while( n >= 0){
		if ( y[n] > 0)
			dy_dz[n] = 1.0;
		else
			dy_dz[n] = 0.0;
		n--;
	}
	return ;
}


void LReLU(float * z, float * y, int n){
	n--;
	while( n >= 0 ){
		if ( z[n] > 0 )
			y[n] = z[n];
		else
			y[n] = z[n] / leak_rate;
		n--;
	}
	return ;
}

// dy_dz =  dy / dz | evaluated at y
void LReLUPrime(float * y, float * dy_dz, int n){
	n--;
	while( n >= 0){
		if ( y[n] > 0)
			dy_dz[n] = 1.0;
		else
			dy_dz[n] = 1.0 / leak_rate;
		n--;
	}
	return ;
}


void Softmax(float * z, float * y, int n){
	n--;
	int num = n;
	// compute the maximum of z
	float z_max = -1e10;
	while( num >= 0){
		if ( z[num] > z_max ){
			z_max = z[num];
		}
		num--;
	}

	// extract the maximum, exponentiate, and compute the denominator

	float den = 0;
	num = n;
	while( num >= 0 ){
		float tmp = z[num] - z_max;
		tmp = exp( tmp );
		den += tmp;
		z[num] = tmp;
		num--;
	}

	// compute y
	while( n >= 0 ){
		y[n] = z[n] / den;
		n--;
	}



}
