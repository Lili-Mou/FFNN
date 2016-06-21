/*
 * FFNN.h
 *
 *  Created on: Mar 7, 2015
 *      Author: mou
 */

#ifndef FFNN_H_
#define FFNN_H_

#include"global.h"
#include<iostream>
using namespace std;


class Layer;
class Connection;

class Layer{

public:
	// neuron properties
	string name;
	int bidx; // the beginning index of biases
	int numUnit;
	void (* f)( float *z, float *y, int n); // activation function, y = f(z)
	void (* fprime)( float * y, float * dy_dz, int n);// derivative

	// connections
	int numDown;
//	Connection **connectUp;
	Connection ** connectDown;

	// neuron states
	float * z;
	float * y;
	float * dE_dz;
	float * dE_dy;
	float * dy_dz;
	float p_drop;

	// construction function
	Layer(string name, int numUnit, int bidx, int numDown,
			void (* f)(float *z, float *y, int n),
			void (* fprime)(float *y, float * dy_dz, int n)
	);
	virtual ~Layer(){}

	// forward
	virtual void computeY();

	virtual void updateB();
};

class DropoutLayer: public Layer{
public:
	int * indicator;
	DropoutLayer(string name, int numUnit, int bidx, int numDown,
				void (* f)(float *z, float *y, int n),
				void (* fprime)(float *y, float * dy_dz, int n)
		);
	~DropoutLayer();
	virtual void computeY();
	virtual void updateB();

};


class PositivePoolLayer: public Layer{
public:
	int * indicator;
	PositivePoolLayer(string name, int numUnit, int numDown);
	~PositivePoolLayer();
	virtual void computeY();
	virtual void updateB();
};

class Connection{
public:
	Layer * xlayer;
	Layer * ylayer;
	int xnum;
	int ynum;
	int Widx;
	float Wcoef;

	// construction function
	Connection();
	Connection(Layer* _x, Layer* _y, int _xnum, int _ynum, int _Widx, float _Wcoef);
	virtual ~Connection(){}
	// compute Z
	virtual void computeZ();
	virtual void updateW();
};

class BilinearConnection: public Connection{
public:
	Layer * xlayer2;
	float * pointwise_prod;
	BilinearConnection(Layer* _x1, Layer* _x2, Layer* _y, int _xnum, int _Widx, float _Wcoef);
	~BilinearConnection();
	virtual void computeZ();
	virtual void updateW();
};

class MinusConnection: public Connection{
public:
	Layer * xlayer2;
	float * minus_prod;
	MinusConnection(Layer* _x1, Layer* _x2, Layer* _y, int _xnum, int _Widx, float _Wcoef);
	~MinusConnection();
	virtual void computeZ();
	virtual void updateW();
};

class MergeConnection: public Connection{
public:
	Layer * xlayer2;
	//float * pointwise_prod;
	MergeConnection(Layer* _x1, Layer* _x2, Layer* _y, int _xnum, int _Widx, float _Wcoef);
	~MergeConnection();
	virtual void computeZ();
	virtual void updateW();
};

class PositivePoolConnection: public Connection{
public:
	int type; // max pooling = 0, sum pooling = 1
	PositivePoolConnection(Layer * _x, Layer* _y, int _num, int _type);
	~PositivePoolConnection(){};
	virtual void computeZ();
	virtual void updateW();

};

void FeedForward(Layer **, int n);
void BackPropagation( Layer **, int n);
void CleanDerivative( Layer ** layer, int n);


//extern int num_train, num_CV, num_test;

extern int* file_train;
extern int* file_CV;
extern int* file_test;

extern Layer *** X_train;
extern Layer *** X_CV;
extern Layer *** X_test;

extern int * len_X_train;
extern int * len_X_CV;
extern int * len_X_test;

extern int * y_train;
extern int * y_CV;
extern int * y_test;

/////////////////////////
// blas wrapper
inline void icopy(int n, float * x, float * y);
inline void iXpY(int n,  float * x, float * y);
inline void alphaXpY(int n, float alpha, float * x, float * y);
inline void matrix_dot_vector(int M, int N, float coef, float * A, enum CBLAS_TRANSPOSE trans, float * x, float * y);
inline void selfplus_matrix_dot_vector(int M, int N, float coef, float * A, enum CBLAS_TRANSPOSE trans, float * x, float * y);
inline void pointwise_dot(float * x, float * y, float *z, int n);
inline void selfplus_matrix_dot_matrix(int M, int N, int K,
							float ceof, float * A, enum CBLAS_TRANSPOSE transA,
							float * B, enum CBLAS_TRANSPOSE transB, float * C);
void GradDescent(int n, float learn_rate, float * grad, float * param);
#endif /* FFNN_H_ */
